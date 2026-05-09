use crate::{
    ast::{AfterRewrite, classical::BinaryOpKind, try_log2, ubig_with_n_lower_bits_set},
    error::{LowerError, LowerErrorKind},
    meta::{
        DimExpr, Progress,
        classical::{MetaExpr, MetaStmt, meta_expr, meta_stmt},
        expand::{Expandable, MacroEnv},
    },
};
use qwerty_ast_macros::rebuild;

impl MetaExpr {
    pub fn expand(self, env: &mut MacroEnv) -> Result<(MetaExpr, Progress), LowerError> {
        rebuild!(MetaExpr, self, expand, env)
    }

    pub(crate) fn expand_rewriter(
        self,
        env: &mut MacroEnv,
        children_progress: Progress,
    ) -> Result<(Self, Progress, AfterRewrite), LowerError> {
        match self {
            MetaExpr::Mod {
                dividend,
                divisor,
                dbg,
            } => {
                if matches!(children_progress, Progress::Full) {
                    let divisor_int = divisor.extract()?;
                    let pow = try_log2(divisor_int).ok_or_else(|| {
                        // For now, the divisor needs to be a power of 2
                        LowerError {
                            kind: LowerErrorKind::ModNotPowerOf2 {
                                bad_divisor: divisor_int,
                            },
                            dbg: dbg.clone(),
                        }
                    })?;
                    let and_op = MetaExpr::BinaryOp {
                        kind: BinaryOpKind::And,
                        left: dividend,
                        right: Box::new(MetaExpr::BitLiteral {
                            val: ubig_with_n_lower_bits_set(pow),
                            // We have no way of knowing at this stage how
                            // big this register should be. Best we can do
                            // is allocate a placeholder dimvar that we pray
                            // inference will sort out.
                            n_bits: DimExpr::DimVar {
                                var: env.allocate_internal_dim_var(),
                                dbg: dbg.clone(),
                            },
                            dbg: dbg.clone(),
                        }),
                        dbg,
                    };
                    // Try to expand just in case, since we just changed things
                    Ok((and_op, children_progress, AfterRewrite::Retry))
                } else {
                    Ok((
                        MetaExpr::Mod {
                            dividend,
                            divisor,
                            dbg,
                        },
                        children_progress,
                        AfterRewrite::Done,
                    ))
                }
            }

            other @ (MetaExpr::Variable { .. }
            | MetaExpr::Slice { .. }
            | MetaExpr::UnaryOp { .. }
            | MetaExpr::BinaryOp { .. }
            | MetaExpr::ReduceOp { .. }
            | MetaExpr::ModMul { .. }
            | MetaExpr::BitLiteral { .. }
            | MetaExpr::Repeat { .. }
            | MetaExpr::Concat { .. }) => Ok((other, children_progress, AfterRewrite::Done)),
        }
    }
}

impl Expandable for MetaStmt {
    fn expand(self, env: &mut MacroEnv) -> Result<(MetaStmt, Progress), LowerError> {
        rebuild!(MetaStmt, self, expand, env)
    }
}
