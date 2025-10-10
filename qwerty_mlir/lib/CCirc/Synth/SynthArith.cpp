#include "CCirc/IR/CCircOps.h"
#include "CCirc/Synth/CCircSynth.h"

namespace {
    std::pair<mlir::Value, mlir::Value> fullAddr1(
            mlir::OpBuilder &builder, mlir::Location loc,
            mlir::Value a,
            mlir::Value b, 
            mlir::Value carry_in){
            
            // 1 bit adder
            // sum = a ^ b ^ carry_in
            // carry out = (a & b) | (a & carry_in) | (b & carry_in)

            mlir::Value sum = builder.create<ccirc::ParityOp>(loc, std::initializer_list<mlir::Value>{
                a, b, carry_in}).getResult();
            mlir::Value carry_out1 = builder.create<ccirc::OrOp>(loc,
                builder.create<ccirc::AndOp>(loc, a, b).getResult(),
                builder.create<ccirc::AndOp>(loc, a, carry_in).getResult());
            mlir::Value carry_out2 = builder.create<ccirc::OrOp>(loc,
                builder.create<ccirc::AndOp>(loc, b, carry_in).getResult(), carry_out1);
            return {sum, carry_out2};
        }

    mlir::Value fullAddrN(
            mlir::OpBuilder &builder, mlir::Location loc,
            llvm::SmallVectorImpl<mlir::Value> &a,
            llvm::SmallVectorImpl<mlir::Value> &b, 
            mlir::Value carry_in,
            llvm::SmallVectorImpl<mlir::Value> &returnSum){

        // n bit addr of a and b
        // loop through every bit returning full added value and carry out
        assert(a.size() == b.size() && "a and b must be same size");

        mlir::Value carry = carry_in;
        returnSum.clear();
        returnSum.append(a.size(), nullptr);

        for(size_t i = 0; i < a.size(); i++){
            auto [sum, cnext] = fullAddr1(builder, loc, a[a.size()-1-i], b[b.size()-1-i], carry);
            returnSum[a.size()-1-i] = sum;
            carry = cnext;
        }
        return carry;
    }
}
namespace ccirc {

void synthAdd(
        mlir::OpBuilder &builder,
        mlir::Location loc,
        llvm::SmallVectorImpl<mlir::Value> &wires_a,
        llvm::SmallVectorImpl<mlir::Value> &wires_b,
        llvm::SmallVectorImpl<mlir::Value> &wires_sum) {
    // TODO: implement me
    assert(0 && "Add not implemented");
}


    // make it take wires_y, does mod doubling on wires y and put on wires out.
    void synthModMul(
            mlir::OpBuilder &builder,
            mlir::Location loc,
            llvm::APInt x,
            llvm::APInt modN,
            llvm::SmallVectorImpl<mlir::Value> &wires_y,
            llvm::SmallVectorImpl<mlir::Value> &wires_out) {
            
                

                // for testing fullAddrN
                // wires_y = cin, a, b


                llvm::SmallVector<mlir::Value> aBits;
                llvm::SmallVector<mlir::Value> bBits;
                mlir::Value cin = wires_y[15];
                mlir::Value zero = builder.create<ccirc::ConstantOp>(loc, llvm::APInt(1, 0)).getResult();
                mlir::Value one = builder.create<ccirc::ConstantOp>(loc, llvm::APInt(1, 1)).getResult();

                // set aBits and bBits
                for (size_t i = 0; i < 8; i++){
                    aBits.push_back(wires_y[i+16]);
                    bBits.push_back(wires_y[i+24]);
                }


                for(size_t i = 0; i < wires_y.size() - 9; i++){
                    wires_out.push_back(zero);
                }
            
                // call full addr
                llvm::SmallVector<mlir::Value> sumBits;
                auto carryOut = fullAddrN(builder, loc, aBits, bBits, cin, sumBits);

                // clear output and push sum then carryout
                wires_out.push_back(carryOut);
                wires_out.append(sumBits.begin(), sumBits.end());


                /*
                for(size_t i = 0; i < 7; i++){
                    wires_out.push_back(zero);
                }
                wires_out.push_back(one);
                wires_out.push_back(zero);
                */
                
                
            }

} // namespace ccirc
