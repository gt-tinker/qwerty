//===- CCircOps.cpp - CCirc dialect ops --------------------*- C++ -*-===//
//===----------------------------------------------------------------------===//

#include <unordered_set>
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "CCirc/IR/CCircOps.h"
#include "CCirc/IR/CCircDialect.h"
#include "CCirc/Transforms/CCircPasses.h"

#define GET_OP_CLASSES
#include "CCirc/IR/CCircOps.cpp.inc"

namespace {
struct DoubleNegation : public mlir::OpRewritePattern<ccirc::NotOp> { 
     using OpRewritePattern<ccirc::NotOp>::OpRewritePattern; 
  
     mlir::LogicalResult matchAndRewrite(ccirc::NotOp op, 
                                         mlir::PatternRewriter &rewriter) const override { 
         ccirc::NotOp upstream_notop = 
             op.getCallee().getDefiningOp<ccirc::NotOp>(); 
  
         if (!upstream_notop) { 
             return mlir::failure(); 
         } 
  
         rewriter.replaceOp(op, upstream_notop.getCallee()); 
         return mlir::success(); 
     } 
 }; 
 } //namespace end

 void NotOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                        MLIRContext *context) {
  results.add<DoubleNegationPattern>(context);
}