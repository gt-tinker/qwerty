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

// Build ~N as wires, where N is zero extended by one bit first. Passing this
// to fullAdderN() as b with a carry_in of 1 calculates a - N, since
// a - N == a + ~N + 1 in two's complement.
void notModN(
        mlir::OpBuilder &builder,
        mlir::Location loc,
        llvm::APInt modN,
        llvm::SmallVectorImpl<mlir::Value> &wires_not_n) {
    mlir::Value not_modN = ccirc::ConstantOp::create(builder,
        loc, ~modN.zext(modN.getBitWidth()+1)).getResult();
    mlir::ValueRange not_n_wires = ccirc::WireUnpackOp::create(builder,
        loc, not_modN).getWires();
    wires_not_n.assign(not_n_wires.begin(), not_n_wires.end());
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

    llvm::SmallVector<mlir::Value> wires_not_n;
    notModN(builder, loc, modN, wires_not_n);

    // 2*a - N
    mlir::Value one = ccirc::ConstantOp::create(builder,
        loc, llvm::APInt(/*numBits=*/1, /*val=*/1)).getResult();
    llvm::SmallVector<mlir::Value> wires_diff;
    mlir::Value carry_out = fullAdderN(builder, loc, wires_shifted,
                                       wires_not_n, one, wires_diff);

    // A carry out means no borrow, i.e. 2*a >= N, so the difference is the
    // reduced result. Since a < N, both candidates fit in n_bits bits, so the
    // extra bit added above is dropped either way.
    llvm::SmallVector<mlir::Value> wires_then(wires_diff.begin()+1,
                                              wires_diff.end());
    llvm::SmallVector<mlir::Value> wires_else(wires_shifted.begin()+1,
                                              wires_shifted.end());
    synthMux(builder, loc, carry_out, wires_then, wires_else, wires_out);
}

void synthModMul(
        mlir::OpBuilder &builder,
        mlir::Location loc,
        llvm::APInt x,
        llvm::APInt modN,
        llvm::SmallVectorImpl<mlir::Value> &wires_y,
        llvm::SmallVectorImpl<mlir::Value> &wires_out) {
    // TODO: implement me
    assert(0 && "Mod mul not implemented");

    // Suggested pseudocode:
    // ---------------------
    // size_t bitsize = wires_y.size();
    // assert(bitsize > 0);
    // ssize_t x_idx = bitsize-1
    // if x[x_idx] == 1 {
    //   acc = y
    // } else {
    //   acc = 0
    // }
    //
    // while (--x_idx >= 0) {
    //   doubled = doubleMod(acc, modN)
    //   if x[x_idx] == 1 {
    //     acc = addMod(doubled, y, modN)
    //   }
    // }
    // Reference:
    // https://github.com/gt-tinker/tweedledum/blob/a041ef41d1763f19f0a76592ef4b79fae6203240/external/mockturtle/mockturtle/generators/modular_arithmetic.hpp#L486

    // TODO: Verify the carry_out condition on the muxes at the end of both
    //       helper pseudocode snippets below. Do I have them backwards?
    // Helpers:
    // --------
    // doubleMod(wires_a, modN) {
    //     bitsize = wires_a.size()
    //     assert(modN.getBitWidth() == bitsize);
    //     shifted = [wires_a, constant(0)] (aka wires_a << 1 in C syntax)
    //     // Below, [1]+ is sign extension
    //     not_n_wires = [1] + [NOT(constant(modN[bitsize-1-i]))
    //                          for i in range(bitsize)]
    //     // 2*a - N
    //     sum, carry_out = synthesize adder(a=shifted, b=not_n_wires,
    //                                       carry_in=constant(1))
    //     // Remove MSB in order to return bitsize bits
    //     return carry_out? shifted[1:] : sum[1:]
    // }
    // Reference:
    // https://github.com/gt-tinker/tweedledum/blob/a041ef41d1763f19f0a76592ef4b79fae6203240/external/mockturtle/mockturtle/generators/modular_arithmetic.hpp#L385
    //
    // addMod(wires_a, wires_b, modN) {
    //     bitsize = wires_a.size()
    //     assert(modN.getBitWidth() == bitsize && wires_b.size() == bitsize);
    //     sum, carry_out = synthesize adder(a=wires_a, b=wires_b,
    //                                       carry_in=constant(0))
    //     bigsum = [carry_out] + sum // bigsum is bitsize+1 bits
    //     // Below, [1]+ is sign extension
    //     not_n_wires = [1] + [NOT(constant(modN[bitsize-1-i]))
    //                          for i in range(bitsize)]
    //     diff, carry_out = synthesize adder(a=bigsum, b=not_n_wires,
    //                                        carry_in=constant(1)) // (a+b)-N
    //     // Remove MSB of diff in order to return bitsize bits
    //     return carry_out? sum : diff[1:]
    // }
    // Reference:
    // https://github.com/gt-tinker/tweedledum/blob/a041ef41d1763f19f0a76592ef4b79fae6203240/external/mockturtle/mockturtle/generators/modular_arithmetic.hpp#L125
}

} // namespace ccirc
