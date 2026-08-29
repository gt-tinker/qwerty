#include "CCirc/IR/CCircOps.h"
#include "CCirc/Synth/CCircSynth.h"

namespace {

std::pair<mlir::Value, mlir::Value> fullAdder1(
        mlir::OpBuilder &builder,
        mlir::Location loc,
        mlir::Value a,
        mlir::Value b,
        mlir::Value carry_in) {

    // 1 bit adder
    // sum = a ^ b ^ carry_in
    // carry out = (a & b) | (a & carry_in) | (b & carry_in)

    mlir::Value sum = ccirc::ParityOp::create(builder, loc, std::initializer_list<mlir::Value>{
        a, b, carry_in}).getResult();
    mlir::Value carry_out1 = ccirc::OrOp::create(builder, loc,
        ccirc::AndOp::create(builder, loc, a, b).getResult(),
        ccirc::AndOp::create(builder, loc, a, carry_in).getResult());
    mlir::Value carry_out2 = ccirc::OrOp::create(builder, loc,
        ccirc::AndOp::create(builder, loc, b, carry_in).getResult(), carry_out1);
    return {sum, carry_out2};
}

mlir::Value fullAdderN(
        mlir::OpBuilder &builder,
        mlir::Location loc,
        llvm::SmallVectorImpl<mlir::Value> &wires_a,
        llvm::SmallVectorImpl<mlir::Value> &wires_b,
        mlir::Value carry_in,
        llvm::SmallVectorImpl<mlir::Value> &wires_sum) {

    assert(wires_a.size() == wires_b.size() && "a and b must be same size");
    size_t n_bits = wires_a.size();

    mlir::Value carry = carry_in;
    wires_sum.clear();
    wires_sum.append(n_bits, nullptr);

    for (size_t i = 0; i < n_bits; i++){
        auto [sum, cnext] = fullAdder1(builder, loc, wires_a[n_bits-1-i], wires_b[n_bits-1-i], carry);
        wires_sum[n_bits-1-i] = sum;
        carry = cnext;
    }
    return carry;
}

// Add the constant b to the wires a, returning the carry out. Since b is known
// at synthesis time, there is no need for a full adder: fixing b collapses the
// 1-bit adder above to two gates per bit.
//
//     b == 0: sum = a ^ carry_in,    carry_out = a & carry_in
//     b == 1: sum = ~(a ^ carry_in), carry_out = a | carry_in
//
// Passing ~N as b with a carry_in of 1 calculates a - N, since
// a - N == a + ~N + 1 in two's complement.
mlir::Value constAdderN(
        mlir::OpBuilder &builder,
        mlir::Location loc,
        llvm::SmallVectorImpl<mlir::Value> &wires_a,
        llvm::APInt b,
        mlir::Value carry_in,
        llvm::SmallVectorImpl<mlir::Value> &wires_sum) {

    size_t n_bits = wires_a.size();
    assert(b.getBitWidth() == n_bits && "a and b must be same size");

    mlir::Value carry = carry_in;
    wires_sum.clear();
    wires_sum.append(n_bits, nullptr);

    for (size_t i = 0; i < n_bits; i++) {
        // wires_a is big endian, but APInt indexes bits little endian
        mlir::Value a = wires_a[n_bits-1-i];
        mlir::Value sum = ccirc::XorOp::create(builder, loc, a, carry).getResult();
        mlir::Value cnext;
        if (b[i]) {
            sum = ccirc::NotOp::create(builder, loc, sum).getResult();
            cnext = ccirc::OrOp::create(builder, loc, a, carry).getResult();
        } else {
            cnext = ccirc::AndOp::create(builder, loc, a, carry).getResult();
        }
        wires_sum[n_bits-1-i] = sum;
        carry = cnext;
    }
    return carry;
}

} // namespace

namespace ccirc {

void synthAdd(
        mlir::OpBuilder &builder,
        mlir::Location loc,
        llvm::SmallVectorImpl<mlir::Value> &wires_a,
        llvm::SmallVectorImpl<mlir::Value> &wires_b,
        llvm::SmallVectorImpl<mlir::Value> &wires_sum) {
    mlir::Value zero = ccirc::ConstantOp::create(builder,
        loc, llvm::APInt(/*numBits=*/1, /*val=*/0)).getResult();
    fullAdderN(builder, loc, wires_a, wires_b, zero, wires_sum);
}

void synthSub(
        mlir::OpBuilder &builder,
        mlir::Location loc,
        llvm::SmallVectorImpl<mlir::Value> &wires_a,
        llvm::SmallVectorImpl<mlir::Value> &wires_b,
        llvm::SmallVectorImpl<mlir::Value> &wires_diff) {
    // Two's complement: a - b = a + ~b + 1
    llvm::SmallVector<mlir::Value> wires_not_b;
    for (mlir::Value b : wires_b) {
        wires_not_b.push_back(ccirc::NotOp::create(builder, loc, b).getResult());
    }
    mlir::Value one = ccirc::ConstantOp::create(builder,
        loc, llvm::APInt(/*numBits=*/1, /*val=*/1)).getResult();
    fullAdderN(builder, loc, wires_a, wires_not_b, one, wires_diff);
}

// Pseudocode:
// doubleMod(wires_a, modN) {
//     bitsize = wires_a.size()
//     assert(modN.getBitWidth() == bitsize);
//     shifted = [wires_a, constant(0)] (aka wires_a << 1 in C syntax)
//     // Below, [1]+ accounts for the extra bit shifted in above
//     not_n = [1] + [NOT(modN[bitsize-1-i]) for i in range(bitsize)]
//     // 2*a - N
//     diff, carry_out = synthesize adder(a=shifted, b=not_n,
//                                        carry_in=constant(1))
//     // A carry out means no borrow, i.e. 2*a >= N, so the difference is the
//     // reduced result. Remove the MSB in order to return bitsize bits.
//     return carry_out? diff[1:] : shifted[1:]
// }
// Reference:
// https://github.com/gt-tinker/tweedledum/blob/a041ef41d1763f19f0a76592ef4b79fae6203240/external/mockturtle/mockturtle/generators/modular_arithmetic.hpp#L385
void synthDoubleMod(
        mlir::OpBuilder &builder,
        mlir::Location loc,
        llvm::APInt modN,
        llvm::SmallVectorImpl<mlir::Value> &wires_a,
        llvm::SmallVectorImpl<mlir::Value> &wires_out) {
    size_t n_bits = wires_a.size();
    assert(n_bits && "a is zero bits???");
    assert(modN.getBitWidth() == n_bits && "Modulus must be as wide as a");

    // 2*a needs an extra bit to avoid overflowing
    llvm::SmallVector<mlir::Value> wires_shifted(wires_a.begin(), wires_a.end());
    wires_shifted.push_back(ccirc::ConstantOp::create(builder,
        loc, llvm::APInt(/*numBits=*/1, /*val=*/0)).getResult());

    // 2*a - N
    mlir::Value one = ccirc::ConstantOp::create(builder,
        loc, llvm::APInt(/*numBits=*/1, /*val=*/1)).getResult();
    llvm::SmallVector<mlir::Value> wires_diff;
    mlir::Value carry_out = constAdderN(builder, loc, wires_shifted,
                                        ~modN.zext(n_bits+1), one, wires_diff);

    // A carry out means no borrow, i.e. 2*a >= N, so the difference is the
    // reduced result. Since a < N, both candidates fit in n_bits bits, so the
    // extra bit added above is dropped either way.
    llvm::SmallVector<mlir::Value> wires_then(wires_diff.begin()+1,
                                              wires_diff.end());
    llvm::SmallVector<mlir::Value> wires_else(wires_shifted.begin()+1,
                                              wires_shifted.end());
    synthMux(builder, loc, carry_out, wires_then, wires_else, wires_out);
}

// Pseudocode:
// addMod(wires_a, wires_b, modN) {
//     bitsize = wires_a.size()
//     assert(modN.getBitWidth() == bitsize && wires_b.size() == bitsize);
//     sum, carry_out = synthesize adder(a=wires_a, b=wires_b,
//                                       carry_in=constant(0))
//     bigsum = [carry_out] + sum // bigsum is bitsize+1 bits
//     // Below, [1]+ accounts for the extra bit of bigsum
//     not_n = [1] + [NOT(modN[bitsize-1-i]) for i in range(bitsize)]
//     diff, carry_out = synthesize adder(a=bigsum, b=not_n,
//                                        carry_in=constant(1)) // (a+b)-N
//     // A carry out means no borrow, i.e. a+b >= N, so the difference is the
//     // reduced result. Remove the MSB of diff in order to return bitsize
//     // bits. Otherwise a+b never overflowed bitsize bits in the first place,
//     // so the truncated sum is already correct.
//     return carry_out? diff[1:] : sum
// }
// Reference:
// https://github.com/gt-tinker/tweedledum/blob/a041ef41d1763f19f0a76592ef4b79fae6203240/external/mockturtle/mockturtle/generators/modular_arithmetic.hpp#L125
void synthAddMod(
        mlir::OpBuilder &builder,
        mlir::Location loc,
        llvm::APInt modN,
        llvm::SmallVectorImpl<mlir::Value> &wires_a,
        llvm::SmallVectorImpl<mlir::Value> &wires_b,
        llvm::SmallVectorImpl<mlir::Value> &wires_out) {
    size_t n_bits = wires_a.size();
    assert(n_bits && "a is zero bits???");
    assert(wires_b.size() == n_bits && "a and b must be same size");
    assert(modN.getBitWidth() == n_bits && "Modulus must be as wide as a");

    mlir::Value zero = ccirc::ConstantOp::create(builder,
        loc, llvm::APInt(/*numBits=*/1, /*val=*/0)).getResult();
    llvm::SmallVector<mlir::Value> wires_sum;
    mlir::Value carry = fullAdderN(builder, loc, wires_a, wires_b, zero,
                                   wires_sum);

    // a + b needs an extra bit to avoid overflowing
    llvm::SmallVector<mlir::Value> wires_bigsum;
    wires_bigsum.push_back(carry);
    wires_bigsum.append(wires_sum.begin(), wires_sum.end());

    // (a + b) - N
    mlir::Value one = ccirc::ConstantOp::create(builder,
        loc, llvm::APInt(/*numBits=*/1, /*val=*/1)).getResult();
    llvm::SmallVector<mlir::Value> wires_diff;
    mlir::Value carry_out = constAdderN(builder, loc, wires_bigsum,
                                        ~modN.zext(n_bits+1), one, wires_diff);

    // A carry out means no borrow, i.e. a + b >= N, so the difference is the
    // reduced result. Otherwise a + b did not overflow n_bits bits in the
    // first place, so the truncated sum is already correct.
    llvm::SmallVector<mlir::Value> wires_then(wires_diff.begin()+1,
                                              wires_diff.end());
    synthMux(builder, loc, carry_out, wires_then, wires_sum, wires_out);
}

// Double-and-add, starting from the most significant bit of x. Since x is a
// constant, its bits are tested at synthesis time instead of by the circuit,
// so only the modular reductions cost any gates.
//
// Pseudocode:
// modMul(x, modN, wires_y) {
//     bitsize = wires_y.size();
//     assert(bitsize > 0);
//     x_idx = bitsize-1
//     if x[x_idx] == 1 {
//       acc = y
//     } else {
//       acc = 0
//     }
//
//     while (--x_idx >= 0) {
//       doubled = doubleMod(acc, modN)
//       if x[x_idx] == 1 {
//         acc = addMod(doubled, y, modN)
//       } else {
//         acc = doubled
//       }
//     }
//     return acc
// }
// Reference:
// https://github.com/gt-tinker/tweedledum/blob/a041ef41d1763f19f0a76592ef4b79fae6203240/external/mockturtle/mockturtle/generators/modular_arithmetic.hpp#L486
void synthModMul(
        mlir::OpBuilder &builder,
        mlir::Location loc,
        llvm::APInt x,
        llvm::APInt modN,
        llvm::SmallVectorImpl<mlir::Value> &wires_y,
        llvm::SmallVectorImpl<mlir::Value> &wires_out) {
    size_t n_bits = wires_y.size();
    assert(n_bits && "y is zero bits???");
    assert(modN.getBitWidth() == n_bits && "Modulus must be as wide as y");
    assert(x.getBitWidth() == n_bits && "x must be as wide as y");

    llvm::SmallVector<mlir::Value> wires_acc;
    if (x[n_bits-1]) {
        wires_acc.append(wires_y.begin(), wires_y.end());
    } else {
        mlir::Value zero = ccirc::ConstantOp::create(builder,
            loc, llvm::APInt(/*numBits=*/1, /*val=*/0)).getResult();
        wires_acc.append(n_bits, zero);
    }

    for (ssize_t i = (ssize_t)n_bits-2; i >= 0; i--) {
        llvm::SmallVector<mlir::Value> wires_doubled;
        synthDoubleMod(builder, loc, modN, wires_acc, wires_doubled);

        if (x[i]) {
            synthAddMod(builder, loc, modN, wires_doubled, wires_y, wires_acc);
        } else {
            wires_acc = std::move(wires_doubled);
        }
    }

    wires_out.clear();
    wires_out.append(wires_acc.begin(), wires_acc.end());
}

} // namespace ccirc
