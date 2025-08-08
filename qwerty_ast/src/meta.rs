use crate::{
    ast::{self, RegKind},
    dbg::DebugLoc,
    error::{ExtractError, ExtractErrorKind},
};
use dashu::{base::Signed, integer::IBig};
use std::fmt;

pub mod classical;
pub mod qpu;

#[derive(Debug, Clone, PartialEq)]
pub enum DimExpr {
    /// Dimension variable. Example syntax:
    /// ```text
    /// N
    /// ```
    DimVar { name: String, dbg: Option<DebugLoc> },

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
    /// Extract a constant integer from this dimension variable expression or /
    /// return `None` if it is not fully folded yet.
    pub fn extract(&self) -> Result<usize, ExtractError> {
        match self {
            DimExpr::DimConst { val, dbg } => {
                if val.is_negative() {
                    Err(ExtractError {
                        kind: ExtractErrorKind::NegativeInteger {
                            offender: val.clone(),
                        },
                        dbg: dbg.clone(),
                    })
                } else {
                    val.try_into().map_err(|_err| ExtractError {
                        kind: ExtractErrorKind::IntegerTooBig {
                            offender: val.clone(),
                        },
                        dbg: dbg.clone(),
                    })
                }
            }
            DimExpr::DimVar { dbg, .. }
            | DimExpr::DimSum { dbg, .. }
            | DimExpr::DimProd { dbg, .. }
            | DimExpr::DimNeg { dbg, .. } => Err(ExtractError {
                kind: ExtractErrorKind::NotFullyFolded,
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
            DimExpr::DimVar { name, .. } => write!(f, "{}", name),
            DimExpr::DimConst { val, .. } => write!(f, "{}", val),
            DimExpr::DimSum { left, right, .. } => write!(f, "({})+({})", left, right),
            DimExpr::DimProd { left, right, .. } => write!(f, "({})*({})", left, right),
            DimExpr::DimNeg { val, .. } => write!(f, "-({})", val),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
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
    /// Extract an [`ast::Type`] from this MetaQwerty type or return `None` if
    /// contained dimension variable expressions are not fully folded yet.
    pub fn extract(&self) -> Result<ast::Type, ExtractError> {
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
                let extracted_tys: Result<Vec<_>, ExtractError> =
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

/// A function (kernel) definition.
///
/// Example syntax:
/// ```text
/// @qpu[[N]]
/// def get_zero() -> qubit[N]:
///     return '0'**N
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct MetaFunctionDef<S> {
    pub name: String,
    pub args: Vec<(MetaType, String)>,
    pub ret_type: MetaType,
    pub body: Vec<S>,
    pub is_rev: bool,
    pub dim_vars: Vec<String>,
    pub dbg: Option<DebugLoc>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum MetaFunc {
    /// A `@qpu` kernel.
    Qpu(MetaFunctionDef<qpu::MetaStmt>),
    /// A `@classical` function.
    Classical(MetaFunctionDef<classical::MetaStmt>),
}

/// The top-level node in a Qwerty program that holds all function defintiions.
///
/// In the current implementation, there is only one of these per Python
/// interpreter.
#[derive(Debug, Clone, PartialEq)]
pub struct MetaProgram {
    pub funcs: Vec<MetaFunc>,
    pub dbg: Option<DebugLoc>,
}

impl MetaProgram {
    pub fn expand(&self) -> ast::Program {
        todo!("MetaProgram::expand()")
    }
}
