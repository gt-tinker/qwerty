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

    /*
    void synthModMul(
        mlir::OpBuilder &builder,
        mlir::Location loc,
        llvm::APInt x,
        llvm::APInt modN,
        llvm::SmallVectorImpl<mlir::Value> &wires_y,
        llvm::SmallVectorImpl<mlir::Value> &wires_out) {


        // x, modN are constant
        // wires_y is input y_wires_rev
        // wires_out is x_wires input
        // we go MSB to LSB

        // width from modN, everything must be this length
        size_t bitsize = modN.getBitWidth();

        // checks
        assert(wires_y.size() >= bitsize && "wires_y must contain at least modulus bitwidth");
        assert(wires_y.size() == x.getBitWidth() && "wires_y and x don't have same wire size");
        
        // trim from (width to LSB)
        llvm::SmallVector<mlir::Value> y_trim(wires_y.end() - bitsize, wires_y.end());
        
        // if x is 0, fill  wires_out with 0 wires
        if(x.isZero()){
            wires_out.clear();
            mlir::Value z = builder.create<ccirc::ConstantOp>(loc, builder.getIntegerAttr(builder.getI1Type(), 0)).getResult();
            for (size_t i = 0; i < bitsize; i++){
                wires_out.push_back(z);
            }
            return;
        }


        // define value to go through loop with size of modN 
        llvm::SmallVector<mlir::Value> acc;
        
        // get MSB of x
        if x[bitsize - 1] == 0{
            mlir::Value z = builder.create<ccirc::ConstantOp>(loc, builder.getIntegerAttr(builder.getI1Type(), 0)).getResult();
            for (size_t i = 0; i < bitsize; i++){
                acc.push_back(z);
            }
        } else {
            acc = y_trim;
        }
        
        // loop through the rest of x
        for(size_t i = 1; i < bitsize; i++){

            // double and take modN
            double_mod(builder, loc, acc, modN);

            // if the current bit of x is 1, add y_trim modN
            if(x[bitsize - 1 - i]){
                add_mod(builder, loc, acc, y_trim, modN);
            }
        }
        

        wires_out.clear();
        wires_out.append(acc.begin(), acc.end());
    }
    

    void double_mod( 
        mlir::OpBuilder &builder,
        mlir::Location loc,
        llvm::SmallVectorImpl<mlir::Value> &acc,
        llvm::APInt &modN)
    {
    
        // things to check:
        // std::copy/copy_n both work
        // builder.getBoolAttr works

        size_t bitsize = acc.size();
        assert(bitsize == modN.getBitWidth() && "modN width must match acc");

        std::copy(acc.begin() + 1u, acc.end(), acc.begin());
        acc[bitsize - 1] = cZero();

        llvm::SmallVector<mlir::Value> shifted(bitsize + 1u, cZero());
        std::copy_n(shifted.begin(), bitsize, acc.begin());


        llvm::SmallVector<mlir::Value> word(bitsize + 1, cZero());

        mlir::Value modulus = builder.create<ccirc::ConstantOp>(loc, builder.getIntegerAttr(builder.getIntegerType(bitsize+1), modN.zext(bitsize+1))).getResult();
        llvm::SmallVector<mlir::Value> modulus_bits(builder.create<ccirc::WireUnpackOp>(loc, modulus).getWires());

        mlir::Value carry_inv =  cZero();
        
        // carry_ripple_subtractor_inplace(ntk, shifted, word, carry_inv);

        // mux_inplace( ntk, ntk.create_not( carry_inv ), acc, std::vector<signal<Ntk>>(shifted.begin(), shifted.begin() + bitsize));
        // synthMux(builder, loc, wires_k[wires_k.size()-1-i], shifted, unshifted, wires_out);         
    }

    inline void carry_ripple_adder_inplace( Ntk& ntk, std::vector<signal<Ntk>>& a, std::vector<signal<Ntk>> const& b, signal<Ntk>& carry )
    {
        static_assert( is_network_type_v<Ntk>, "Ntk is not a network type" );

        assert( a.size() == b.size() );

        auto pa = a.begin();
        for ( auto pb = b.begin(); pa != a.end(); ++pa, ++pb )
        {
            std::tie( *pa, carry ) = full_adder( ntk, *pa, *pb, carry );
        }
        }

        inline void modular_adder_inplace( Ntk& ntk, std::vector<signal<Ntk>>& a, std::vector<signal<Ntk>> const& b )
        {
        auto carry = ntk.get_constant( false );
        carry_ripple_adder_inplace( ntk, a, b, carry );
    }

    

    mlir::Value carry_ripple_subtractor_inplace(
        mlir::OpBuilder &b, mlir::Location loc,
        llvm::SmallVectorImpl<mlir::Value> &a,
        llvm::ArrayRef<mlir::Value> bvec )
    {
        static_assert( is_network_type_v<Ntk>, "Ntk is not a network type" );
        static_assert( has_create_not_v<Ntk>, "Ntk does not implement the create_not method" );

        assert( a.size() == b.size() );

        auto pa = a.begin();
        for ( auto pb = b.begin(); pa != a.end(); ++pa, ++pb )
        {
            std::tie( *pa, carry ) = full_adder( ntk, *pa, ntk.create_not( *pb ), carry );
        }
    }
    
    */




} // namespace ccirc
