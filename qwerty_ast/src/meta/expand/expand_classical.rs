use crate::{
    error::ExtractError,
    meta::{
        Progress,
        classical::{MetaExpr, MetaStmt},
        expand::{Expandable, MacroEnv},
    },
};

impl MetaExpr {
    pub fn expand(&self, env: &MacroEnv) -> Result<(MetaExpr, Progress), ExtractError> {
        match self {
            MetaExpr::Slice {
                val,
                lower,
                upper,
                dbg,
            } => val.expand(env).and_then(|(expanded_val, val_prog)| {
                lower.expand(env).and_then(|(expanded_lower, lower_prog)| {
                    upper.expand(env).map(|(expanded_upper, upper_prog)| {
                        (
                            MetaExpr::Slice {
                                val: Box::new(expanded_val),
                                lower: expanded_lower,
                                upper: expanded_upper,
                                dbg: dbg.clone(),
                            },
                            val_prog.join(lower_prog).join(upper_prog),
                        )
                    })
                })
            }),

            MetaExpr::UnaryOp { kind, val, dbg } => {
                val.expand(env).map(|(expanded_val, val_prog)| {
                    (
                        MetaExpr::UnaryOp {
                            kind: *kind,
                            val: Box::new(expanded_val),
                            dbg: dbg.clone(),
                        },
                        val_prog,
                    )
                })
            }

            MetaExpr::BinaryOp {
                kind,
                left,
                right,
                dbg,
            } => left.expand(env).and_then(|(expanded_left, left_prog)| {
                right.expand(env).map(|(expanded_right, right_prog)| {
                    (
                        MetaExpr::BinaryOp {
                            kind: *kind,
                            left: Box::new(expanded_left),
                            right: Box::new(expanded_right),
                            dbg: dbg.clone(),
                        },
                        left_prog.join(right_prog),
                    )
                })
            }),

            MetaExpr::ReduceOp { kind, val, dbg } => {
                val.expand(env).map(|(expanded_val, val_prog)| {
                    (
                        MetaExpr::ReduceOp {
                            kind: *kind,
                            val: Box::new(expanded_val),
                            dbg: dbg.clone(),
                        },
                        val_prog,
                    )
                })
            }

            MetaExpr::BitLiteral { val, n_bits, dbg } => {
                n_bits.expand(env).map(|(expanded_n_bits, n_bits_prog)| {
                    (
                        MetaExpr::BitLiteral {
                            val: val.clone(),
                            n_bits: expanded_n_bits,
                            dbg: dbg.clone(),
                        },
                        n_bits_prog,
                    )
                })
            }

            MetaExpr::Variable { .. } => Ok((self.clone(), Progress::Full)),
        }
    }
}

impl Expandable for MetaStmt {
    fn expand(&self, env: &mut MacroEnv) -> Result<(MetaStmt, Progress), ExtractError> {
        match self {
            MetaStmt::Expr { expr } => expr.expand(env).map(|(expanded_expr, progress)| {
                (
                    MetaStmt::Expr {
                        expr: expanded_expr,
                    },
                    progress,
                )
            }),

            MetaStmt::Assign { lhs, rhs, dbg } => {
                rhs.expand(env).map(|(expanded_expr, progress)| {
                    (
                        MetaStmt::Assign {
                            lhs: lhs.to_string(),
                            rhs: expanded_expr,
                            dbg: dbg.clone(),
                        },
                        progress,
                    )
                })
            }

            MetaStmt::UnpackAssign { lhs, rhs, dbg } => {
                rhs.expand(env).map(|(expanded_expr, progress)| {
                    (
                        MetaStmt::UnpackAssign {
                            lhs: lhs.clone(),
                            rhs: expanded_expr,
                            dbg: dbg.clone(),
                        },
                        progress,
                    )
                })
            }

            MetaStmt::Return { val, dbg } => val.expand(env).map(|(expanded_expr, progress)| {
                (
                    MetaStmt::Return {
                        val: expanded_expr,
                        dbg: dbg.clone(),
                    },
                    progress,
                )
            }),
        }
    }
}
