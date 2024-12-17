#include "mlir/IR/Verifier.h"

#ifdef QWERTY_USE_QIREE
#include "qiree/Module.hh"
#include "qiree/Executor.hh"
#include "qirxacc/XaccQuantum.hh"
#endif

#include "defs.hpp"
#include "ast.hpp"
#include "ast_visitor.hpp"
#include "mlir_handle.hpp"
#include "qir_qrt.hpp"

namespace {
using QireeResults = std::unordered_map<std::string, KernelResult>;

struct QireeParser {
    inline static const std::string tuple_tok = "tuple ";
    inline static const std::string length_tok = "length ";
    inline static const std::string distinct_tok = "distinct results ";
    inline static const std::string result_tok = "result ";
    inline static const std::string count_tok = "count ";

    std::string line;
    size_t off;

    bool gobble_tok(const std::string &tok) {
        if (line.size() < tok.size()
                || off + tok.size() > line.size()
                || line.substr(off, tok.size()) != tok) {
            return false;
        }
        off += tok.size();
        return true;
    }

    bool gobble_next(std::string &token_out) {
        bool found_space = true;
        size_t end;
        if ((end = line.find(' ', off)) == std::string::npos) {
            if (line.empty()) {
                return false;
            } else {
                found_space = false;
                end = line.size();
            }
        }
        token_out = line.substr(off, end-off);
        off = end;
        if (found_space) {
            off++; // Skip space
        }
        return true;
    }

    bool conv_uint(const std::string &token, size_t &uint_out) {
        size_t result;
        try {
            result = std::stoul(token);
        } catch (std::invalid_argument const&) {
            return false;
        } catch (std::out_of_range const&) {
            return false;
        }
        uint_out = result;
        return true;
    }

    bool gobble_uint(size_t &uint_out) {
        std::string uint_str;
        if (!gobble_next(uint_str)) {
            return false;
        }
        return conv_uint(uint_str, uint_out);
    }

    bool at_eol() const {
        return off == line.size();
    }

    bool parse(std::istream &is, QireeResults &results) {
        size_t tuple_rem = 0;
        std::string cur_tag;
        while (std::getline(is, line)) {
            off = 0;
            if (!line.empty()
                    && (line[0] == '{'
                        || line[0] == ' '
                        || line[0] == '}')) {
                // This is the accelerator buffer printout for humans to read, ignore it
                continue;
            }
            if (!gobble_tok(tuple_tok)) {
                goto failure;
            }
            std::string tag;
            if (!gobble_next(tag)) {
                goto failure;
            }
            if (gobble_tok(length_tok)) {
                size_t length;
                if (!gobble_uint(length)) {
                    goto failure;
                }
                if (!gobble_tok(distinct_tok)) {
                    goto failure;
                }
                size_t n_distinct;
                if (!gobble_uint(n_distinct)) {
                    goto failure;
                }
                if (!at_eol()) {
                    goto failure;
                }
                if (tuple_rem) {
                    goto failure;
                }
                if (results.count(tag)) {
                    goto failure;
                }
                cur_tag = tag;
                tuple_rem = n_distinct;
            } else if (gobble_tok(result_tok)) {
                std::string bits_str;
                if (!gobble_next(bits_str)) {
                    goto failure;
                }
                if (!gobble_tok(count_tok)) {
                    goto failure;
                }
                size_t count;
                if (!gobble_uint(count)) {
                    goto failure;
                }
                if (!at_eol()) {
                    goto failure;
                }
                if (!tuple_rem || tag != cur_tag) {
                    goto failure;
                }
                tuple_rem--;

                std::vector<bool> bits;
                bits.reserve(bits_str.size());
                for (char c : bits_str) {
                    if (c != '0' && c != '1') {
                        goto failure;
                    }
                    bits.push_back(c-'0');
                }
                if (results[tag].count(bits)) {
                    goto failure;
                }
                results[tag][bits] = count;
            } else {
                goto failure;
            }
        }

        return true;

        failure:
        return false;
    }
};
} // namespace

void QpuKernel::compile(MlirHandle &handle) {
    Kernel::compile(handle);
    QpuLoweringVisitor visitor(handle, getFuncOpName(), capture_objs);
    walk(visitor);

    assert(funcOp && "QpuLoweringVisitor did not set Kernel funcOp field. Is it broken?");

    const FuncType &func_type = dynamic_cast<const FuncType &>(getType());
    bool has_non_classical_inputs = !func_type.lhs->isClassical()
                                    || !func_type.rhs->isClassical();
    // Useful if this is some kernel we've compiled with some specific type
    // parameters used in 1 place. (In that case, we want inlining to remove it.)
    // ...or, as a separate consideration, if this function takes qubits as
    // inputs or outputs, it should never be called directly, so it doesn't
    // need its own LLVM function after lowering. So it should be private in
    // that case, too. However, we can't roll this logic into funcOp_private,
    // since that also indicates it was generated just for another kernel to be
    // used and thus can be erased when that caller is erased (see
    // eraseIfPrivate())
    if (funcOp_private || has_non_classical_inputs) {
        funcOp.setPrivate();
    }

    if (mlir::failed(mlir::verify(funcOp))) {
#ifndef NDEBUG
        funcOp->dump();
#endif
        throw CompileException("Could not verify " + ast_kind_name(AST_QPU)
                               + " FuncOp. See errors on stderr",
                               std::move(dbg->copy()));
    }
}

// Call into the JIT'd code with the simulation runtime from qir-runner, unless
// built wih QIR-EE (see docs/qiree.md) and `accelerator' is nonempty.
std::unique_ptr<KernelResult> QpuKernel::call(MlirHandle &handle,
                                              std::string accelerator,
                                              size_t n_shots) {
    const FuncType &func_type = dynamic_cast<const FuncType &>(getType());

    const TupleType *arg_tuple;
    if (!(arg_tuple = dynamic_cast<const TupleType *>(func_type.lhs.get()))
            || !arg_tuple->isUnit()) {
        throw JITException("I don't know how to invoke a kernel that takes "
                           "arguments e.g. " + func_type.lhs->toString());
    }

    const BitType *output_type = nullptr;
    const TupleType *ret_tuple;
    if (((ret_tuple = dynamic_cast<const TupleType *>(func_type.rhs.get()))
                && ret_tuple->isUnit())
            || (output_type =
                    dynamic_cast<const BitType *>(func_type.rhs.get()))) {
#ifdef QWERTY_USE_QIREE
        if (accelerator.empty()) {
            return callLocalSim(handle, n_shots, output_type);
        } else {
            return callQiree(handle, accelerator, n_shots, output_type);
        }
#else
        if (!accelerator.empty()) {
            throw JITException("To use accelerators other than the built-in "
                               "simulator, you need to rebuild with QIR-EE "
                               "support. See docs/qiree.md.");
        }
        return callLocalSim(handle, n_shots, output_type);
#endif
    } else {
        throw JITException("I don't know how to invoke a kernel with return "
                           "type " + func_type.rhs->toString());
    }

}

#ifdef QWERTY_USE_QIREE
std::unique_ptr<KernelResult> QpuKernel::callQiree(
        MlirHandle &handle,
        std::string accelerator,
        size_t n_shots,
        const BitType *output_type) {
    std::unique_ptr<llvm::Module> mod =
        handle.get_qir_module(/*to_base_profile=*/true);

    // Clear previous XACC output
    // per https://stackoverflow.com/a/5288044/321301
    handle.xacc_ss.str("");
    handle.xacc_ss.clear();

    handle.xacc.set_accelerator_and_shots(accelerator, n_shots);

    {
        qiree::Executor execute{qiree::Module{std::move(mod),
                                              getFuncOpName()}};
        execute(handle.xacc, handle.xacc_rt);
    }

    if (qwerty_debug) {
        std::ofstream stream(MlirHandle::qiree_log_filename, std::ios::app);
        stream << handle.xacc_ss.str() << "\n\n";
        stream.close();
    }

    if (!output_type) { // Returns Unit
        return nullptr;
    } else { // Returns bit[N]
        QireeResults results;
        QireeParser parser;
        #define QIREE_LOG_HELP "Set the $QWERTY_DEBUG environment variable " \
                               "to 1 and check qiree.log for the full " \
                               "output from QIR-EE."
        if (!parser.parse(handle.xacc_ss, results)) {
            throw JITException("Output from QIR-EE is malformed. " QIREE_LOG_HELP);
        }
        if (!results.count("ret")) {
            throw JITException("Output from QIR-EE is missing a ret tuple. " QIREE_LOG_HELP);
        }
        #undef QIREE_LOG_HELP
        std::unique_ptr<KernelResult> ret =
            std::make_unique<KernelResult>(results.at("ret"));
        return ret;
    }
}
#endif

std::unique_ptr<KernelResult> QpuKernel::callLocalSim(
        MlirHandle &handle,
        size_t n_shots,
        const BitType *output_type) {
    handle.jit_if_needed();

    std::string funcop_name = getFuncOpName();
    if (!output_type) { // Returns Unit
        for (size_t i = 0; i < n_shots; i++) {
            __quantum__rt__initialize(nullptr);
            if (handle.exec->invokePacked(funcop_name)) {
                throw JITException("Invoking kernel failed");
            }
        }
        return nullptr;
    } else { // Returns bit[N]
        std::unique_ptr<KernelResult> result =
                std::make_unique<KernelResult>();
        for (size_t i = 0; i < n_shots; i++) {
            __quantum__rt__initialize(nullptr);
            QirArray *arr;
            std::vector<void *> args{&arr};
            if (handle.exec->invokePacked(funcop_name, args)) {
                throw JITException("Invoking kernel failed");
            }
            size_t n_bits = (size_t)__quantum__rt__array_get_size_1d(arr);
            std::vector<bool> bits(n_bits);
            for (size_t i = 0; i < n_bits; i++) {
                bits[i] = *(char *)__quantum__rt__array_get_element_ptr_1d(arr, (int64_t)i);
            }
            __quantum__rt__array_update_reference_count(arr, -1);
            (*result)[bits]++;
        }
        return result;
    }
}

std::string QpuKernel::qasm(MlirHandle &handle, bool print_locs) {
    return handle.qasm(getFuncOpName(), print_locs);
}
