#include "CCirc/IR/CCircOps.h"
#include "CCirc/Synth/CCircSynth.h"

namespace ccirc {

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

    // 
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




} // namespace ccirc
