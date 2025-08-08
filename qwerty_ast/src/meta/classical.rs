use crate::{
    ast::classical::{BinaryOpKind, UnaryOpKind},
    dbg::DebugLoc,
    meta::DimExpr,
};
use dashu::integer::UBig;
use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub enum MetaExpr {
    /// A variable name used in an expression. Example syntax:
    /// ```text
    /// my_var
    /// ```
    Variable { name: String, dbg: Option<DebugLoc> },

    /// A unary bitwise operation. Example syntax:
    /// ```text
    /// ~x
    /// ```
    UnaryOp {
        kind: UnaryOpKind,
        val: Box<MetaExpr>,
        dbg: Option<DebugLoc>,
    },

    /// A binary bitwise operation. Example syntax:
    /// ```text
    /// x & y
    /// ```
    BinaryOp {
        kind: BinaryOpKind,
        left: Box<MetaExpr>,
        right: Box<MetaExpr>,
        dbg: Option<DebugLoc>,
    },

    /// A logical reduction operation. Example syntax:
    /// ```text
    /// x.xor_reduce()
    /// ```
    ReduceOp {
        kind: BinaryOpKind,
        val: Box<MetaExpr>,
        dbg: Option<DebugLoc>,
    },

    /// A constant bit register. Example syntax:
    /// ```text
    /// bit[N](0b0)
    /// ```
    BitLiteral {
        val: UBig,
        n_bits: DimExpr,
        dbg: Option<DebugLoc>,
    },
}

// TODO: don't duplicate this code with classical.rs
impl fmt::Display for MetaExpr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MetaExpr::Variable { name, .. } => write!(f, "{}", name),
            MetaExpr::UnaryOp { kind, val, .. } => {
                let kind_str = match kind {
                    UnaryOpKind::Not => "~",
                };
                write!(f, "{}({})", kind_str, *val)
            }
            MetaExpr::BinaryOp {
                kind, left, right, ..
            } => {
                let kind_str = match kind {
                    BinaryOpKind::And => "&",
                    BinaryOpKind::Or => "|",
                    BinaryOpKind::Xor => "^",
                };
                write!(f, "({}) {} ({})", *left, kind_str, *right)
            }
            MetaExpr::ReduceOp { kind, val, .. } => {
                let kind_str = match kind {
                    BinaryOpKind::And => "and",
                    BinaryOpKind::Or => "or",
                    BinaryOpKind::Xor => "xor",
                };
                write!(f, "({}).{}_reduce()", kind_str, *val)
            }
            MetaExpr::BitLiteral { val, n_bits, .. } => {
                write!(f, "bit[{}](0b{:b})", n_bits, val)
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum MetaStmt {
    /// An expression statement. Example syntax:
    /// ```text
    /// f(x)
    /// ```
    Expr { expr: MetaExpr },

    /// An assignment statement. Example syntax:
    /// ```text
    /// q = '0'
    /// ```
    Assign {
        lhs: String,
        rhs: MetaExpr,
        dbg: Option<DebugLoc>,
    },

    /// A register-unpacking assignment statement. Example syntax:
    /// ```text
    /// q1, q2 = '01'
    /// ```
    UnpackAssign {
        lhs: Vec<String>,
        rhs: MetaExpr,
        dbg: Option<DebugLoc>,
    },

    /// A return statement. Example syntax:
    /// ```text
    /// return q
    /// ```
    Return {
        val: MetaExpr,
        dbg: Option<DebugLoc>,
    },
}

// TODO: don't duplicate with ast.rs
impl fmt::Display for MetaStmt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MetaStmt::Expr { expr, .. } => write!(f, "{}", expr),
            MetaStmt::Assign { lhs, rhs, .. } => write!(f, "{} = {}", lhs, rhs),
            MetaStmt::UnpackAssign { lhs, rhs, .. } => {
                for (i, name) in lhs.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", name)?;
                }
                write!(f, " = {}", rhs)
            }
            MetaStmt::Return { val, .. } => write!(f, "return {}", val),
        }
    }
}
