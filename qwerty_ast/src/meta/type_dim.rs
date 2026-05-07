use crate::{
    ast::{self, RegKind},
    dbg::DebugLoc,
    error::{LowerError, LowerErrorKind},
    meta::{MacroEnv, Progress},
};
use dashu::{base::Signed, integer::IBig};
use qwerty_ast_macros::{
    gen_rebuild, rebuild, rewrite_match, rewrite_ty, visitor_expr, visitor_write,
};
use std::{
    fmt,
    ops::{AddAssign, MulAssign},
};

/// A dimension variable. The distinction between the two variants exists
/// because macro parameters are distinct from function dimension variables
/// (even if they have the same name), and even dimension variables with the
/// same name across different functions must be distinct.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum DimVar {
    /// A macro parameter whose scope is only the definition of a macro.
    /// Example:
    /// ```text
    /// fourier[N] = fourier[N-1] // std.revolve
    /// ```
    /// Another example:
    /// ```text
    /// (op[[i]] for i in range(M))
    ///      ^
    /// ```
    MacroParam { var_name: String },
    /// A function dimension variable. Example:
    /// ```text
    /// @qpu[[N]]
    /// def foo():
    ///     ...
    /// ```
    FuncVar { var_name: String, func_name: String },

    /// A global dimension variable. Currently only used in the REPL to allow
    /// for temporary dimension variable allocation during expansion.
    GlobalVar { var_name: String },
}

impl fmt::Display for DimVar {
    /// Returns a representation of a dimension variable expression that
    /// matches the syntax in the Python DSL.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            // TODO: disambiguate these two variants?
            DimVar::MacroParam { var_name }
            | DimVar::FuncVar { var_name, .. }
            | DimVar::GlobalVar { var_name } => {
                write!(f, "{}", var_name)
            }
        }
    }
}

#[gen_rebuild {
    substitute_dim_var(
        rewrite(substitute_dim_var_rewriter),
        more_copied_args(dim_var: &DimVar, new_dim_expr: &DimExpr),
    ),
    expand(
        rewrite(expand_rewriter),
        progress(Progress),
        more_copied_args(env: &MacroEnv),
        result_err(LowerError),
    ),
    strip_dbg(
        rewrite(strip_dbg_rewriter),
    ),
    canonicalize(
        rewrite(canonicalize_rewriter),
    ),
}]
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum DimExpr {
    /// Dimension variable. Example syntax:
    /// ```text
    /// N
    /// ```
    DimVar { var: DimVar, dbg: Option<DebugLoc> },

    /// A constant dimension variable value. Example syntax:
    /// ```text
    /// 2
    /// ```
    DimConst { val: IBig, dbg: Option<DebugLoc> },

    /// Sum of dimension variable values. Example syntax:
    /// ```text
    /// N + 1
    /// ```
    DimSum {
        left: Box<DimExpr>,
        right: Box<DimExpr>,
        dbg: Option<DebugLoc>,
    },

    /// Product of dimension variable values. Example syntax:
    /// ```text
    /// 2*N
    /// ```
    DimProd {
        left: Box<DimExpr>,
        right: Box<DimExpr>,
        dbg: Option<DebugLoc>,
    },

    /// Raise one dimension variable values to the power of another. Example
    /// syntax:
    /// ```text
    /// 2**J
    /// ```
    DimPow {
        base: Box<DimExpr>,
        pow: Box<DimExpr>,
        dbg: Option<DebugLoc>,
    },

    /// Negation of a dimension variable value. Example syntax:
    /// ```text
    /// -N
    /// ```
    DimNeg {
        val: Box<DimExpr>,
        dbg: Option<DebugLoc>,
    },
}

impl DimExpr {
    /// Returns true if this is a constant with the provided value.
    fn is_constant(&self, value: &IBig) -> bool {
        if let DimExpr::DimConst { val, .. } = self {
            val == value
        } else {
            false
        }
    }

    /// Returns true if this is a constant 0.
    pub fn is_constant_zero(&self) -> bool {
        self.is_constant(&IBig::ZERO)
    }

    /// Returns true if this is a constant 1.
    pub fn is_constant_one(&self) -> bool {
        self.is_constant(&IBig::ONE)
    }

    /// Return the same dimension variable expression but without any debug
    /// symbols. Useful for comparing expressions.
    pub fn strip_dbg(self) -> DimExpr {
        rebuild!(DimExpr, self, strip_dbg)
    }

    pub(crate) fn strip_dbg_rewriter(self) -> DimExpr {
        match self {
            DimExpr::DimVar { var, dbg: _ } => DimExpr::DimVar { var, dbg: None },

            DimExpr::DimConst { val, dbg: _ } => DimExpr::DimConst { val, dbg: None },

            DimExpr::DimSum {
                left,
                right,
                dbg: _,
            } => DimExpr::DimSum {
                left,
                right,
                dbg: None,
            },

            DimExpr::DimProd {
                left,
                right,
                dbg: _,
            } => DimExpr::DimProd {
                left,
                right,
                dbg: None,
            },

            DimExpr::DimPow { base, pow, dbg: _ } => DimExpr::DimPow {
                base,
                pow,
                dbg: None,
            },

            DimExpr::DimNeg { val, dbg: _ } => DimExpr::DimNeg { val, dbg: None },
        }
    }

    /// Recursively flatten a tree of `DimSum`s into a list of operands.
    fn flatten_sum(self, out: &mut Vec<DimExpr>) {
        let mut stack = vec![self];

        while let Some(node) = stack.pop() {
            if let DimExpr::DimSum {
                left,
                right,
                dbg: _,
            } = node
            {
                stack.push(*right);
                stack.push(*left);
            } else {
                out.push(node);
            }
        }
    }

    /// Recursively flatten a tree of `DimProd`s into a list of operands.
    fn flatten_prod(self, out: &mut Vec<DimExpr>) {
        let mut stack = vec![self];

        while let Some(node) = stack.pop() {
            if let DimExpr::DimProd {
                left,
                right,
                dbg: _,
            } = node
            {
                stack.push(*right);
                stack.push(*left);
            } else {
                out.push(node);
            }
        }
    }

    /// Fold multiple constants in a sum or product into one constant. To be
    /// used after [`DimExpr::flatten_sum`] and [`DimExpr::flatten_prod`].
    fn fold_constants<F1, F2>(
        vals: Vec<DimExpr>,
        init_acc: IBig,
        acc_func: F1,
        is_ident_func: F2,
    ) -> Vec<DimExpr>
    where
        F1: Fn(&mut IBig, IBig),
        F2: Fn(&IBig) -> bool,
    {
        let mut acc = init_acc;
        let mut operands = vec![];
        for v in vals {
            if let DimExpr::DimConst { val, .. } = v {
                acc_func(&mut acc, val);
            } else {
                operands.push(v);
            }
        }
        if !is_ident_func(&acc) {
            operands.push(DimExpr::DimConst {
                val: acc,
                dbg: None,
            });
        }
        operands
    }

    /// Find duplicate entries in the provided list of terms in a sum `N+N+N+4`
    /// and consolidate them into `3*N + 4`.
    fn consolidate_dupes_in_sum(mut vals: Vec<DimExpr>) -> Vec<DimExpr> {
        vals.reverse();
        let mut output = Vec::new();
        let mut prev: Option<DimExpr> = None;
        let mut count = 0;
        loop {
            let maybe_next = vals.pop();

            if let (Some(previous), Some(next)) = (prev.as_ref(), maybe_next.as_ref())
                && previous.clone().strip_dbg() == next.clone().strip_dbg()
            {
                count += 1;
            } else {
                if let Some(previous) = prev {
                    let this_output = if count == 1 {
                        previous
                    } else {
                        DimExpr::DimProd {
                            left: Box::new(DimExpr::DimConst {
                                val: count.into(),
                                dbg: None,
                            }),
                            right: Box::new(previous),
                            dbg: None,
                        }
                    };
                    output.push(this_output);
                }

                if maybe_next.is_none() {
                    break;
                }

                count = 1;
                prev = maybe_next;
            }
        }
        output
    }

    /// Return a canon form of this expression in which:
    ///
    /// 1. Trees of `DimSum` and `DimProd` are rebuilt in a left-recursive
    ///    form where operands are sorted.
    /// 2. `DimProd` operands that are `DimConst` 1s are removed
    /// 3. `DimSum` operands that are `DimConst` 0s are removed
    ///
    /// Debug symbols are not removed. You should call [`DimExpr::strip_dbg`]
    /// for that first if you want it.
    pub fn canonicalize(self) -> DimExpr {
        rebuild!(DimExpr, self, canonicalize)
    }

    pub(crate) fn canonicalize_rewriter(self) -> DimExpr {
        match self {
            DimExpr::DimNeg { val, dbg } => {
                // --N => N
                if let DimExpr::DimNeg {
                    val: inner_val,
                    dbg: _,
                } = *val
                {
                    *inner_val
                } else {
                    DimExpr::DimNeg { val, dbg }
                }
            }

            DimExpr::DimSum { left, right, dbg } => {
                let mut vals = vec![];
                left.flatten_sum(&mut vals);
                right.flatten_sum(&mut vals);

                let mut operands =
                    Self::fold_constants(vals, IBig::ZERO, IBig::add_assign, IBig::is_zero);
                operands.sort();
                operands = Self::consolidate_dupes_in_sum(operands);
                operands.sort();

                operands
                    .into_iter()
                    .reduce(|left, right| DimExpr::DimSum {
                        left: Box::new(left),
                        right: Box::new(right),
                        dbg: dbg.clone(),
                    })
                    .unwrap_or_else(|| DimExpr::DimConst {
                        val: IBig::ZERO,
                        dbg,
                    })
            }

            DimExpr::DimProd { left, right, dbg } => {
                let mut vals = vec![];
                left.flatten_prod(&mut vals);
                right.flatten_prod(&mut vals);

                let mut operands =
                    Self::fold_constants(vals, IBig::ONE, IBig::mul_assign, IBig::is_one);
                operands.sort();
                operands
                    .into_iter()
                    .reduce(|left, right| DimExpr::DimProd {
                        left: Box::new(left),
                        right: Box::new(right),
                        dbg: dbg.clone(),
                    })
                    .unwrap_or_else(|| DimExpr::DimConst {
                        val: IBig::ONE,
                        dbg,
                    })
            }

            already_canon @ (DimExpr::DimVar { .. }
            | DimExpr::DimConst { .. }
            | DimExpr::DimPow { .. }) => already_canon,
        }
    }

    /// Returns `true` if this expression contains at least one dim var.
    pub fn contains_dim_var(&self) -> bool {
        visitor_expr! {DimExpr, self,
            DimExpr::DimVar { .. } => true,
            DimExpr::DimConst { .. } => false,
            DimExpr::DimSum { left, right, .. } => visit!(*left) || visit!(*right),
            DimExpr::DimProd { left, right, .. } => visit!(*left) || visit!(*right),
            DimExpr::DimPow { base, pow, .. } => visit!(*base) || visit!(*pow),
            DimExpr::DimNeg { val, .. } => visit!(*val),
        }
    }

    /// Extract an [`IBig`] from this dimension variable expression or
    /// return an error if it is not fully folded yet.
    pub fn extract_bigint(self) -> Result<IBig, LowerError> {
        match self {
            DimExpr::DimConst { val, dbg: _ } => Ok(val),
            DimExpr::DimVar { dbg, .. }
            | DimExpr::DimSum { dbg, .. }
            | DimExpr::DimProd { dbg, .. }
            | DimExpr::DimPow { dbg, .. }
            | DimExpr::DimNeg { dbg, .. } => Err(LowerError {
                kind: LowerErrorKind::NotFullyFolded,
                dbg: dbg.clone(),
            }),
        }
    }

    pub fn get_dbg(&self) -> Option<DebugLoc> {
        let (DimExpr::DimConst { dbg, .. }
        | DimExpr::DimVar { dbg, .. }
        | DimExpr::DimSum { dbg, .. }
        | DimExpr::DimProd { dbg, .. }
        | DimExpr::DimPow { dbg, .. }
        | DimExpr::DimNeg { dbg, .. }) = self;
        dbg.clone()
    }

    /// Extract a constant integer from this dimension variable expression or
    /// return an error if it is not fully folded yet.
    pub fn extract(self) -> Result<usize, LowerError> {
        let dbg = self.get_dbg();
        let val = self.extract_bigint()?;
        if val.is_negative() {
            Err(LowerError {
                kind: LowerErrorKind::NegativeInteger {
                    offender: val.clone(),
                },
                dbg,
            })
        } else {
            val.clone().try_into().map_err(|_err| LowerError {
                kind: LowerErrorKind::IntegerTooBig { offender: val },
                dbg,
            })
        }
    }
}

impl fmt::Display for DimExpr {
    /// Returns a representation of a dimension variable expression that
    /// matches the syntax in the Python DSL.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        visitor_write! {DimExpr, self,
            DimExpr::DimVar { var, .. } => write!(f, "{}", var),
            DimExpr::DimConst { val, .. } => write!(f, "{}", val),
            DimExpr::DimSum { left, right, .. } => write!(f, "({!})+({!})", *left, *right),
            DimExpr::DimProd { left, right, .. } => write!(f, "({!})*({!})", *left, *right),
            DimExpr::DimPow { base, pow, .. } => write!(f, "({!})**({!})", *base, *pow),
            DimExpr::DimNeg { val, .. } => write!(f, "-({!})", *val),
        }
    }
}

#[gen_rebuild {
    substitute_dim_var(
        more_copied_args(dim_var: &DimVar, new_dim_expr: &DimExpr),
        recurse_attrs,
    ),
    expand(
        progress(Progress),
        more_copied_args(env: &MacroEnv),
        result_err(LowerError),
        recurse_attrs,
    ),
    extract(
        rewrite(extract_rewriter),
        rewrite_to(
            MetaType => ast::Type,
            DimExpr => usize,
        ),
        result_err(LowerError),
        recurse_attrs,
    ),
}]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum MetaType {
    FuncType {
        in_ty: Box<MetaType>,
        out_ty: Box<MetaType>,
    },
    RevFuncType {
        in_out_ty: Box<MetaType>,
    },
    RegType {
        #[gen_rebuild::skip_recurse]
        elem_ty: RegKind,
        dim: DimExpr,
    },
    TupleType {
        tys: Vec<MetaType>,
    },
    UnitType,
}

impl MetaType {
    /// Extract an [`ast::Type`] from this MetaQwerty type or return an error
    /// if contained dimension variable expressions are not fully folded yet.
    pub fn extract(self) -> Result<ast::Type, LowerError> {
        rebuild!(MetaType, self, extract)
    }

    pub(crate) fn extract_rewriter(
        rewritten: rewrite_ty!(MetaType, extract),
    ) -> Result<ast::Type, LowerError> {
        Ok(rewrite_match! {MetaType, extract, rewritten,
            FuncType { in_ty, out_ty } => {
                ast::Type::FuncType {
                    in_ty: Box::new(in_ty),
                    out_ty: Box::new(out_ty),
                }
            }

            RevFuncType { in_out_ty } => {
                ast::Type::RevFuncType {
                    in_out_ty: Box::new(in_out_ty),
                }
            }

            RegType { elem_ty, dim } => ast::Type::RegType { elem_ty, dim },

            TupleType { tys } => ast::Type::TupleType { tys },

            UnitType => ast::Type::UnitType,
        })
    }
}

// TODO: Don't duplicate this with ast.rs
impl fmt::Display for MetaType {
    /// Returns a representation of a type that matches the syntax for the
    /// Python DSL.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        visitor_write! {MetaType, self,
            MetaType::FuncType { in_ty, out_ty }
                if matches!(
                    (&**in_ty, &**out_ty),
                    (
                        MetaType::RegType {
                            elem_ty: in_elem_ty @ (RegKind::Qubit | RegKind::Bit),
                            dim: in_dim
                        },
                        MetaType::RegType {
                            elem_ty: out_elem_ty @ (RegKind::Qubit | RegKind::Bit),
                            dim: out_dim
                        },
                    )
                    if in_elem_ty == out_elem_ty
                        && in_dim.clone().strip_dbg() == out_dim.clone().strip_dbg()) =>
            {
                write!(
                    f,
                    "{}func[{}]",
                    match &**in_ty {
                        MetaType::RegType {
                            elem_ty: RegKind::Qubit,
                            ..
                        } => "q",
                        MetaType::RegType {
                            elem_ty: RegKind::Bit,
                            ..
                        } => "c",
                        _ => unreachable!(),
                    },
                    match &**in_ty {
                        MetaType::RegType { dim, .. } => dim,
                        _ => unreachable!(),
                    }
                )
            }

            MetaType::FuncType { in_ty, out_ty }
                if matches!(
                    (&**in_ty, &**out_ty),
                    (
                        MetaType::RegType {
                            elem_ty: in_elem_ty @ (RegKind::Qubit | RegKind::Bit),
                            dim: in_dim
                        },
                        MetaType::RegType {
                            elem_ty: out_elem_ty @ (RegKind::Qubit | RegKind::Bit),
                            dim: out_dim
                        },
                    )
                    if in_elem_ty == out_elem_ty
                        && in_dim.clone().strip_dbg() != out_dim.clone().strip_dbg()) =>
            {
                write!(
                    f,
                    "{}func[{},{}]",
                    match &**in_ty {
                        MetaType::RegType {
                            elem_ty: RegKind::Qubit,
                            ..
                        } => "q",
                        MetaType::RegType {
                            elem_ty: RegKind::Bit,
                            ..
                        } => "c",
                        _ => unreachable!(),
                    },
                    match &**in_ty {
                        MetaType::RegType { dim, .. } => dim,
                        _ => unreachable!(),
                    },
                    match &**out_ty {
                        MetaType::RegType { dim, .. } => dim,
                        _ => unreachable!(),
                    },
                )
            }

            MetaType::FuncType { in_ty, out_ty } => write!(f, "func[{!},{!}]", *in_ty, *out_ty),

            MetaType::RevFuncType { in_out_ty }
                if matches!(
                    &**in_out_ty,
                    MetaType::RegType {
                        elem_ty: elem_ty @ (RegKind::Qubit | RegKind::Bit),
                        dim,
                    }
                ) =>
            {
                write!(
                    f,
                    "rev_{}func[{}]",
                    match &**in_out_ty {
                        MetaType::RegType {
                            elem_ty: RegKind::Qubit,
                            ..
                        } => "q",
                        MetaType::RegType {
                            elem_ty: RegKind::Bit,
                            ..
                        } => "c",
                        _ => unreachable!(),
                    },
                    match &**in_out_ty {
                        MetaType::RegType { dim, .. } => dim,
                        _ => unreachable!(),
                    },
                )
            }

            MetaType::RevFuncType { in_out_ty } => write!(f, "rev_func[{!}]", *in_out_ty),

            MetaType::RegType { elem_ty: RegKind::Qubit, dim } => write!(f, "qubit[{}]", dim),

            MetaType::RegType { elem_ty: RegKind::Bit, dim } => write!(f, "bit[{}]", dim),

            MetaType::RegType { elem_ty: RegKind::Basis, dim } => write!(f, "basis[{}]", dim),

            MetaType::TupleType { tys } => write!(f, "({!:,})", tys, ", "),

            MetaType::UnitType => write!(f, "None"),
        }
    }
}

#[cfg(test)]
mod test_display {
    use super::{DebugLoc, DimExpr, IBig, MetaType, RegKind};

    #[test]
    fn test_meta_type_qfunc_same_dim() {
        let dbg1 = Some(DebugLoc {
            file: "bubba.py".to_string(),
            line: 3,
            col: 4,
        });
        let dbg2 = Some(DebugLoc {
            file: "skippy.py".to_string(),
            line: 2,
            col: 5,
        });
        let ty = MetaType::FuncType {
            in_ty: Box::new(MetaType::RegType {
                elem_ty: RegKind::Qubit,
                dim: DimExpr::DimConst {
                    val: IBig::ONE,
                    dbg: dbg1,
                },
            }),
            out_ty: Box::new(MetaType::RegType {
                elem_ty: RegKind::Qubit,
                dim: DimExpr::DimConst {
                    val: IBig::ONE,
                    dbg: dbg2,
                },
            }),
        };
        assert_eq!(ty.to_string(), "qfunc[1]");
    }

    #[test]
    fn test_meta_type_qfunc_diff_dim() {
        let dbg = Some(DebugLoc {
            file: "bubba.py".to_string(),
            line: 3,
            col: 4,
        });
        let ty = MetaType::FuncType {
            in_ty: Box::new(MetaType::RegType {
                elem_ty: RegKind::Qubit,
                dim: DimExpr::DimConst {
                    val: IBig::ONE,
                    dbg: dbg.clone(),
                },
            }),
            out_ty: Box::new(MetaType::RegType {
                elem_ty: RegKind::Qubit,
                dim: DimExpr::DimConst {
                    val: IBig::ONE + 1,
                    dbg,
                },
            }),
        };
        assert_eq!(ty.to_string(), "qfunc[1,2]");
    }

    #[test]
    fn test_meta_type_cfunc_same_dim() {
        let dbg1 = Some(DebugLoc {
            file: "bubba.py".to_string(),
            line: 3,
            col: 4,
        });
        let dbg2 = Some(DebugLoc {
            file: "skippy.py".to_string(),
            line: 2,
            col: 5,
        });
        let ty = MetaType::FuncType {
            in_ty: Box::new(MetaType::RegType {
                elem_ty: RegKind::Bit,
                dim: DimExpr::DimConst {
                    val: IBig::ONE,
                    dbg: dbg1,
                },
            }),
            out_ty: Box::new(MetaType::RegType {
                elem_ty: RegKind::Bit,
                dim: DimExpr::DimConst {
                    val: IBig::ONE,
                    dbg: dbg2,
                },
            }),
        };
        assert_eq!(ty.to_string(), "cfunc[1]");
    }

    #[test]
    fn test_meta_type_cfunc_diff_dim() {
        let dbg = Some(DebugLoc {
            file: "bubba.py".to_string(),
            line: 3,
            col: 4,
        });
        let ty = MetaType::FuncType {
            in_ty: Box::new(MetaType::RegType {
                elem_ty: RegKind::Bit,
                dim: DimExpr::DimConst {
                    val: IBig::ONE,
                    dbg: dbg.clone(),
                },
            }),
            out_ty: Box::new(MetaType::RegType {
                elem_ty: RegKind::Bit,
                dim: DimExpr::DimConst {
                    val: IBig::ONE + 1,
                    dbg,
                },
            }),
        };
        assert_eq!(ty.to_string(), "cfunc[1,2]");
    }

    #[test]
    fn test_meta_type_func() {
        let dbg = Some(DebugLoc {
            file: "bubba.py".to_string(),
            line: 3,
            col: 4,
        });
        let ty = MetaType::FuncType {
            in_ty: Box::new(MetaType::RegType {
                elem_ty: RegKind::Qubit,
                dim: DimExpr::DimConst {
                    val: IBig::ONE + 2,
                    dbg: dbg.clone(),
                },
            }),
            out_ty: Box::new(MetaType::RegType {
                elem_ty: RegKind::Bit,
                dim: DimExpr::DimConst {
                    val: IBig::ONE + 2,
                    dbg,
                },
            }),
        };
        assert_eq!(ty.to_string(), "func[qubit[3],bit[3]]");
    }

    #[test]
    fn test_meta_type_rev_qfunc() {
        let dbg = Some(DebugLoc {
            file: "bubba.py".to_string(),
            line: 3,
            col: 4,
        });
        let ty = MetaType::RevFuncType {
            in_out_ty: Box::new(MetaType::RegType {
                elem_ty: RegKind::Qubit,
                dim: DimExpr::DimConst {
                    val: IBig::ONE + 3,
                    dbg,
                },
            }),
        };
        assert_eq!(ty.to_string(), "rev_qfunc[4]");
    }

    #[test]
    fn test_meta_type_rev_func_unit() {
        let ty = MetaType::RevFuncType {
            in_out_ty: Box::new(MetaType::UnitType),
        };
        assert_eq!(ty.to_string(), "rev_func[None]");
    }

    #[test]
    fn test_meta_type_qubit_reg() {
        let dbg = Some(DebugLoc {
            file: "bubba.py".to_string(),
            line: 3,
            col: 4,
        });
        let ty = MetaType::RegType {
            elem_ty: RegKind::Qubit,
            dim: DimExpr::DimConst {
                val: IBig::ONE + 3,
                dbg,
            },
        };
        assert_eq!(ty.to_string(), "qubit[4]");
    }

    #[test]
    fn test_meta_type_bit_reg() {
        let dbg = Some(DebugLoc {
            file: "bubba.py".to_string(),
            line: 3,
            col: 4,
        });
        let ty = MetaType::RegType {
            elem_ty: RegKind::Bit,
            dim: DimExpr::DimConst {
                val: IBig::ONE + 2,
                dbg,
            },
        };
        assert_eq!(ty.to_string(), "bit[3]");
    }

    #[test]
    fn test_meta_type_basis_reg() {
        let dbg = Some(DebugLoc {
            file: "bubba.py".to_string(),
            line: 3,
            col: 4,
        });
        let ty = MetaType::RegType {
            elem_ty: RegKind::Basis,
            dim: DimExpr::DimConst {
                val: IBig::ONE + 1,
                dbg,
            },
        };
        assert_eq!(ty.to_string(), "basis[2]");
    }

    #[test]
    fn test_meta_type_tuple() {
        let dbg = Some(DebugLoc {
            file: "bubba.py".to_string(),
            line: 3,
            col: 4,
        });
        let ty = MetaType::TupleType {
            tys: vec![
                MetaType::RegType {
                    elem_ty: RegKind::Qubit,
                    dim: DimExpr::DimConst {
                        val: IBig::ONE + 3,
                        dbg: dbg.clone(),
                    },
                },
                MetaType::RegType {
                    elem_ty: RegKind::Bit,
                    dim: DimExpr::DimConst {
                        val: IBig::ONE + 2,
                        dbg,
                    },
                },
            ],
        };

        assert_eq!(ty.to_string(), "(qubit[4], bit[3])");
    }

    #[test]
    fn test_meta_type_unit() {
        let ty = MetaType::UnitType;
        assert_eq!(ty.to_string(), "None");
    }
}
