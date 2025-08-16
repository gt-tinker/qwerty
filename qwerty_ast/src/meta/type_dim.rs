use crate::{
    ast::{self, RegKind},
    dbg::DebugLoc,
    error::{LowerError, LowerErrorKind},
};
use dashu::{base::Signed, integer::IBig};
use std::fmt;

/// A dimension variable. The distinction between the two variants exists
/// because macro parameters are distinct from function dimension variables
/// (even if they have the same name), and even dimension variables with the
/// same name across different functions must be distinct.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum DimVar {
    /// Example:
    /// ```text
    /// fourier[N] = fourier[N-1] // std.revolve
    ///                      ^
    /// ```
    /// Another example:
    /// ```text
    /// (op[[i]] for i in range(M))
    ///      ^
    /// ```
    MacroParam {
        var_name: String,
    },
    FuncVar {
        var_name: String,
        func_name: String,
    },
}

impl fmt::Display for DimVar {
    /// Returns a representation of a dimension variable expression that
    /// matches the syntax in the Python DSL.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            // TODO: disambiguate these two variants?
            DimVar::MacroParam { var_name } | DimVar::FuncVar { var_name, .. } => {
                write!(f, "{}", var_name)
            }
        }
    }
}

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
    /// Return the same dimension variable expression but without any debug
    /// symbols. Useful for comparing expressions.
    pub fn strip_dbg(&self) -> DimExpr {
        match self {
            DimExpr::DimVar { var, dbg: _ } => DimExpr::DimVar {
                var: var.clone(),
                dbg: None,
            },
            DimExpr::DimConst { val, dbg: _ } => DimExpr::DimConst {
                val: val.clone(),
                dbg: None,
            },
            DimExpr::DimSum {
                left,
                right,
                dbg: _,
            } => DimExpr::DimSum {
                left: Box::new(left.strip_dbg()),
                right: Box::new(right.strip_dbg()),
                dbg: None,
            },
            DimExpr::DimProd {
                left,
                right,
                dbg: _,
            } => DimExpr::DimProd {
                left: Box::new(left.strip_dbg()),
                right: Box::new(right.strip_dbg()),
                dbg: None,
            },
            DimExpr::DimNeg { val, dbg: _ } => DimExpr::DimNeg {
                val: Box::new(val.strip_dbg()),
                dbg: None,
            },
        }
    }

    /// Recursively flatten a tree of `DimSum`s into a list of operands.
    fn flatten_sum(&self, out: &mut Vec<DimExpr>) {
        if let DimExpr::DimSum {
            left,
            right,
            dbg: _,
        } = self
        {
            left.flatten_sum(out);
            right.flatten_sum(out);
        } else {
            out.push(self.clone());
        }
    }

    /// Recursively flatten a tree of `DimProd`s into a list of operands.
    fn flatten_prod(&self, out: &mut Vec<DimExpr>) {
        if let DimExpr::DimProd {
            left,
            right,
            dbg: _,
        } = self
        {
            left.flatten_prod(out);
            right.flatten_prod(out);
        } else {
            out.push(self.clone());
        }
    }

    /// Return a canon form of this expression in which:
    ///
    /// 1. Trees of `DimSum` and `DimProd` are rebuilt in a left-recursive
    ///    form where operands are sorted.
    ///
    /// Debug symbols are not removed. You should call [`DimExpr::strip_dbg`]
    /// for that first if you want it.
    pub fn canonicalize(&self) -> DimExpr {
        match self {
            // Base cases
            DimExpr::DimVar { .. } | DimExpr::DimConst { .. } => self.clone(),

            DimExpr::DimNeg { val, dbg } => DimExpr::DimNeg {
                val: Box::new(val.canonicalize()),
                dbg: dbg.clone(),
            },

            DimExpr::DimSum {
                left, right, dbg, ..
            } => {
                let mut vals = vec![];
                left.flatten_sum(&mut vals);
                right.flatten_sum(&mut vals);
                vals.sort();
                vals.into_iter()
                    .reduce(|left, right| DimExpr::DimSum {
                        left: Box::new(left),
                        right: Box::new(right),
                        dbg: dbg.clone(),
                    })
                    .unwrap()
            }

            DimExpr::DimProd {
                left, right, dbg, ..
            } => {
                let mut vals = vec![];
                left.flatten_prod(&mut vals);
                right.flatten_prod(&mut vals);
                vals.sort();
                vals.into_iter()
                    .reduce(|left, right| DimExpr::DimProd {
                        left: Box::new(left),
                        right: Box::new(right),
                        dbg: dbg.clone(),
                    })
                    .unwrap()
            }
        }
    }

    /// Returns `true` if this expression contains at least one dim var.
    pub fn contains_dim_var(&self) -> bool {
        match self {
            DimExpr::DimVar { .. } => true,
            DimExpr::DimConst { .. } => false,
            DimExpr::DimSum { left, right, .. } => {
                left.contains_dim_var() || right.contains_dim_var()
            }
            DimExpr::DimProd { left, right, .. } => {
                left.contains_dim_var() || right.contains_dim_var()
            }
            DimExpr::DimNeg { val, .. } => val.contains_dim_var(),
        }
    }

    /// Extract a constant integer from this dimension variable expression or
    /// return an error if it is not fully folded yet.
    pub fn extract(&self) -> Result<usize, LowerError> {
        match self {
            DimExpr::DimConst { val, dbg } => {
                if val.is_negative() {
                    Err(LowerError {
                        kind: LowerErrorKind::NegativeInteger {
                            offender: val.clone(),
                        },
                        dbg: dbg.clone(),
                    })
                } else {
                    val.try_into().map_err(|_err| LowerError {
                        kind: LowerErrorKind::IntegerTooBig {
                            offender: val.clone(),
                        },
                        dbg: dbg.clone(),
                    })
                }
            }
            DimExpr::DimVar { dbg, .. }
            | DimExpr::DimSum { dbg, .. }
            | DimExpr::DimProd { dbg, .. }
            | DimExpr::DimNeg { dbg, .. } => Err(LowerError {
                kind: LowerErrorKind::NotFullyFolded,
                dbg: dbg.clone(),
            }),
        }
    }
}

impl fmt::Display for DimExpr {
    /// Returns a representation of a dimension variable expression that
    /// matches the syntax in the Python DSL.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DimExpr::DimVar { var, .. } => write!(f, "{}", var),
            DimExpr::DimConst { val, .. } => write!(f, "{}", val),
            DimExpr::DimSum { left, right, .. } => write!(f, "({})+({})", left, right),
            DimExpr::DimProd { left, right, .. } => write!(f, "({})*({})", left, right),
            DimExpr::DimNeg { val, .. } => write!(f, "-({})", val),
        }
    }
}

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
    pub fn extract(&self) -> Result<ast::Type, LowerError> {
        match self {
            MetaType::FuncType { in_ty, out_ty } => in_ty.extract().and_then(|in_ast_ty| {
                out_ty.extract().map(|out_ast_ty| ast::Type::FuncType {
                    in_ty: Box::new(in_ast_ty),
                    out_ty: Box::new(out_ast_ty),
                })
            }),
            MetaType::RevFuncType { in_out_ty } => {
                in_out_ty
                    .extract()
                    .map(|in_out_ast_ty| ast::Type::RevFuncType {
                        in_out_ty: Box::new(in_out_ast_ty),
                    })
            }
            MetaType::RegType { elem_ty, dim } => dim.extract().map(|dim_val| ast::Type::RegType {
                elem_ty: *elem_ty,
                dim: dim_val,
            }),
            MetaType::TupleType { tys } => {
                let extracted_tys: Result<Vec<_>, LowerError> =
                    tys.iter().map(MetaType::extract).collect();
                extracted_tys.map(|ex_tys| ast::Type::TupleType { tys: ex_tys })
            }
            MetaType::UnitType => Ok(ast::Type::UnitType),
        }
    }
}

// TODO: Don't duplicate this with ast.rs
impl fmt::Display for MetaType {
    /// Returns a representation of a type that matches the syntax for the
    /// Python DSL.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MetaType::FuncType { in_ty, out_ty } => match (&**in_ty, &**out_ty) {
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
                        write!(f, "{}]", in_dim)
                    } else {
                        write!(f, "{},{}]", in_dim, out_dim)
                    }
                }
                _ => write!(f, "func[{},{}]", in_ty, out_ty),
            },
            MetaType::RevFuncType { in_out_ty } => match &**in_out_ty {
                MetaType::RegType {
                    elem_ty: RegKind::Qubit,
                    dim,
                } => write!(f, "rev_qfunc[{}]", dim),
                MetaType::RegType {
                    elem_ty: RegKind::Bit,
                    dim,
                } => write!(f, "rev_bfunc[{}]", dim),
                _ => write!(f, "rev_func[{}]", in_out_ty),
            },
            MetaType::RegType { elem_ty, dim } => match elem_ty {
                RegKind::Qubit => write!(f, "qubit[{}]", dim),
                RegKind::Bit => write!(f, "bit[{}]", dim),
                RegKind::Basis => write!(f, "basis[{}]", dim),
            },
            MetaType::TupleType { tys } => {
                write!(f, "(")?;
                for (i, ty) in tys.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", ty)?;
                }
                write!(f, ")")
            }
            MetaType::UnitType => write!(f, "None"),
        }
    }
}
