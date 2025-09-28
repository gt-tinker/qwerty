use crate::{
    ast::{classical::BinaryOpKind, try_log2, ubig_with_n_lower_bits_set},
    error::{LowerError, LowerErrorKind},
    meta::{
        DimExpr, Progress,
        classical::{MetaExpr, MetaStmt},
        expand::{Expandable, MacroEnv},
    },
};

impl MetaExpr {
    pub fn expand(&self, env: &mut MacroEnv) -> Result<(MetaExpr, Progress), LowerError> {
        match self {
            MetaExpr::Mod {
                dividend,
                divisor,
                dbg,
            } => dividend
                .expand(env)
                .and_then(|(expanded_dividend, dividend_prog)| {
                    divisor
                        .expand(env)
                        .and_then(|(expanded_divisor, divisor_prog)| {
                            if let (DimExpr::DimConst { dbg: dim_dbg, .. }, Progress::Full) =
                                (&expanded_divisor, divisor_prog)
                            {
                                expanded_divisor.extract().and_then(|divisor_int| {
                                    try_log2(divisor_int)
                                        .ok_or_else(|| {
                                            // For now, the divisor needs to be a power of 2
                                            LowerError {
                                                kind: LowerErrorKind::Malformed,
                                                dbg: dbg.clone(),
                                            }
                                        })
                                        .and_then(|pow| {
                                            MetaExpr::BinaryOp {
                                                kind: BinaryOpKind::And,
                                                left: Box::new(expanded_dividend),
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
                                                    dbg: dim_dbg.clone(),
                                                }),
                                                dbg: dbg.clone(),
                                            }
                                            .expand(env)
                                        })
                                })
                            } else {
                                Ok((
                                    MetaExpr::Mod {
                                        dividend: Box::new(expanded_dividend),
                                        divisor: expanded_divisor,
                                        dbg: dbg.clone(),
                                    },
                                    dividend_prog.join(divisor_prog),
                                ))
                            }
                        })
                }),

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

            MetaExpr::ModMul {
                x,
                j,
                y,
                mod_n,
                dbg,
            } => {
                let (expanded_x, x_prog) = x.expand(env)?;
                let (expanded_j, j_prog) = j.expand(env)?;
                let (expanded_y, y_prog) = y.expand(env)?;
                let (expanded_mod_n, mod_n_prog) = mod_n.expand(env)?;
                let progress = x_prog.join(j_prog).join(y_prog).join(mod_n_prog);
                Ok((
                    MetaExpr::ModMul {
                        x: expanded_x,
                        j: expanded_j,
                        y: Box::new(expanded_y),
                        mod_n: expanded_mod_n,
                        dbg: dbg.clone(),
                    },
                    progress,
                ))
            }

            MetaExpr::Repeat { val, amt, dbg } => {
                let (expanded_val, val_prog) = val.expand(env)?;
                let (expanded_amt, amt_prog) = amt.expand(env)?;
                let progress = val_prog.join(amt_prog);
                Ok((
                    MetaExpr::Repeat {
                        val: Box::new(expanded_val),
                        amt: expanded_amt,
                        dbg: dbg.clone(),
                    },
                    progress,
                ))
            }

            MetaExpr::Concat { left, right, dbg } => {
                let (expanded_left, left_prog) = left.expand(env)?;
                let (expanded_right, right_prog) = right.expand(env)?;
                let progress = left_prog.join(right_prog);
                Ok((
                    MetaExpr::Concat {
                        left: Box::new(expanded_left),
                        right: Box::new(expanded_right),
                        dbg: dbg.clone(),
                    },
                    progress,
                ))
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
    fn expand(&self, env: &mut MacroEnv) -> Result<(MetaStmt, Progress), LowerError> {
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
