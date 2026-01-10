use crate::{
    ast::{self, RegKind},
    dbg::DebugLoc,
    error::{LowerError, LowerErrorKind},
    meta::{MacroEnv, Progress},
};
use dashu::{base::Signed, integer::IBig};
use qwerty_ast_macros::{gen_rebuild, rebuild, rewrite_match, rewrite_ty, visitor_match};
use std::fmt;

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
                vals = vals.into_iter().filter(|v| !v.is_constant_zero()).collect();
                vals.sort();
                vals.into_iter()
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
                vals = vals.into_iter().filter(|v| !v.is_constant_one()).collect();
                vals.sort();
                vals.into_iter()
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
        visitor_match! {DimExpr, self,
            DimExpr::DimVar { .. } => {
                return true;
            }
            DimExpr::DimConst { .. } => {},
            DimExpr::DimSum { left, right, .. } => {
                visit!(*left);
                visit!(*right);
            }
            DimExpr::DimProd { left, right, .. } => {
                visit!(*left);
                visit!(*right);
            }
            DimExpr::DimPow { base, pow, .. } => {
                visit!(*base);
                visit!(*pow);
            }
            DimExpr::DimNeg { val, .. } => {
                visit!(*val);
            }
        }

        false
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
        visitor_match! {DimExpr, self,
            DimExpr::DimVar { var, .. } => write!(f, "{}", var)?,
            DimExpr::DimConst { val, .. } => write!(f, "{}", val)?,
            DimExpr::DimSum { left, right, .. } => {
                write!(f, "(")?;
                visit!(*left);
                write!(f, ")+(")?;
                visit!(*right);
                write!(f, ")")?;
            }
            DimExpr::DimProd { left, right, .. } => {
                write!(f, "(")?;
                visit!(*left);
                write!(f, ")*(")?;
                visit!(*right);
                write!(f, ")")?;
            }
            DimExpr::DimPow { base, pow, .. } => {
                write!(f, "(")?;
                visit!(*base);
                write!(f, ")**(")?;
                visit!(*pow);
                write!(f, ")")?;
            }
            DimExpr::DimNeg { val, .. } => {
                write!(f, "-(")?;
                visit!(*val);
                write!(f, ")")?;
            }
        }

        Ok(())
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
        visitor_match! {MetaType, self,
            MetaType::FuncType { in_ty, out_ty } => {
                match (&**in_ty, &**out_ty) {
                    (
                        MetaType::RegType {
                            elem_ty: in_elem_ty,
                            dim: in_dim,
                        },
                        MetaType::RegType {
                            elem_ty: out_elem_ty,
                            dim: out_dim,
                        },
                    ) if *in_elem_ty != RegKind::Basis && *out_elem_ty != RegKind::Basis => {
                        let prefix = match (in_elem_ty, out_elem_ty) {
                            (RegKind::Qubit, RegKind::Qubit) => "q",
                            (RegKind::Qubit, RegKind::Bit) => "qb",
                            (RegKind::Bit, RegKind::Qubit) => "bq",
                            (RegKind::Bit, RegKind::Bit) => "b",
                            (RegKind::Basis, _) | (_, RegKind::Basis) => {
                                unreachable!("bases cannot be function arguments/results")
                            }
                        };
                        write!(f, "{}func[", prefix)?;
                        if in_elem_ty == out_elem_ty && in_dim == out_dim {
                            write!(f, "{}]", in_dim)?;
                        } else {
                            write!(f, "{},{}]", in_dim, out_dim)?;
                        }
                    }
                    _ => {
                        write!(f, "func[")?;
                        // TODO: This visit!() isn't seen because it's not top-level
                        visit!(*in_ty);
                        write!(f, ",")?;
                        // TODO: This visit!() isn't seen because it's not top-level
                        visit!(*out_ty);
                        write!(f, "]")?;
                    }
                }
            }
            MetaType::RevFuncType { in_out_ty } => {
                match &**in_out_ty {
                    MetaType::RegType { elem_ty: RegKind::Qubit, dim } => {
                        write!(f, "rev_qfunc[{}]", dim)?;
                    }
                    MetaType::RegType { elem_ty: RegKind::Bit, dim } => {
                        write!(f, "rev_bfunc[{}]", dim)?;
                    }
                    // TODO: This visit!() isn't seen because it's not top-level
                    _ => {
                        write!(f, "rev_func[")?;
                        visit!(*in_out_ty);
                        write!(f, "]")?;
                    }
                }

            }
            MetaType::RegType { elem_ty, dim } => {
                match elem_ty {
                    RegKind::Qubit => write!(f, "qubit[{}]", dim)?,
                    RegKind::Bit => write!(f, "bit[{}]", dim)?,
                    RegKind::Basis => write!(f, "basis[{}]", dim)?,
                }
            }
            MetaType::TupleType { tys } => {
                write!(f, "(")?;
                for (i, ty) in tys.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    // TODO: This visit!() isn't seen because it's not top-level
                    visit!(ty);
                }
                write!(f, ")")?;
            }
            MetaType::UnitType => write!(f, "None")?,
        }

        Ok(())
    }
}
