//! Expressions for `@classical` functions.

use super::{BitLiteral, Variable};
use crate::dbg::DebugLoc;
use std::fmt;

// ----- Classical Operators -----

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UnaryOpKind {
    Not,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BinaryOpKind {
    And,
    Or,
    Xor,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RotateOpKind {
    Rotr,
    Rotl,
}

// Structs for classical::Expr variants

/// See [`Expr::Slice`].
#[derive(Debug, Clone, PartialEq)]
pub struct Slice {
    pub val: Box<Expr>,
    pub lower: usize,
    pub upper: usize,
    pub dbg: Option<DebugLoc>,
}

/// See [`Expr::UnaryNot`].
#[derive(Debug, Clone, PartialEq)]
pub struct UnaryOp {
    pub kind: UnaryOpKind,
    pub val: Box<Expr>,
    pub dbg: Option<DebugLoc>,
}

/// See [`Expr::BinaryOp`].
#[derive(Debug, Clone, PartialEq)]
pub struct BinaryOp {
    pub kind: BinaryOpKind,
    pub left: Box<Expr>,
    pub right: Box<Expr>,
    pub dbg: Option<DebugLoc>,
}

/// See [`Expr::ReduceOp`].
#[derive(Debug, Clone, PartialEq)]
pub struct ReduceOp {
    pub kind: BinaryOpKind,
    pub val: Box<Expr>,
    pub dbg: Option<DebugLoc>,
}

/// See [`Expr::RotateOp`].
#[derive(Debug, Clone, PartialEq)]
pub struct RotateOp {
    pub kind: RotateOpKind,
    pub val: Box<Expr>,
    pub amt: Box<Expr>,
    pub dbg: Option<DebugLoc>,
}

/// See [`Expr::Concat`].
#[derive(Debug, Clone, PartialEq)]
pub struct Concat {
    pub left: Box<Expr>,
    pub right: Box<Expr>,
    pub dbg: Option<DebugLoc>,
}

/// See [`Expr::Repeat`].
#[derive(Debug, Clone, PartialEq)]
pub struct Repeat {
    pub val: Box<Expr>,
    pub amt: usize,
    pub dbg: Option<DebugLoc>,
}

/// See [`Expr::ModMul`].
#[derive(Debug, Clone, PartialEq)]
pub struct ModMul {
    pub x: usize,
    pub j: usize,
    pub y: Box<Expr>,
    pub mod_n: usize,
    pub dbg: Option<DebugLoc>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    /// A variable name used in an expression. Example syntax:
    /// ```text
    /// my_var
    /// ```
    Variable(Variable),

    /// Extract a subsequence of bits from a register. Example syntax:
    /// ```text
    /// my_bit_reg[2:4]
    /// ```
    Slice(Slice),

    /// A unary bitwise operation. Example syntax:
    /// ```text
    /// ~x
    /// ```
    UnaryOp(UnaryOp),

    /// A binary bitwise operation. Example syntax:
    /// ```text
    /// x & y
    /// ```
    BinaryOp(BinaryOp),

    /// A logical reduction operation. Example syntax:
    /// ```text
    /// x.xor_reduce()
    /// ```
    ReduceOp(ReduceOp),

    /// A bit rotation. Example syntax:
    /// ```text
    /// val_reg.rotl(amt_reg)
    /// ```
    RotateOp(RotateOp),

    /// Concatenate bit registers. Example syntax:
    /// ```text
    /// x.concat(y)
    /// ```
    Concat(Concat),

    /// Copy a bit n times. Example syntax:
    /// ```text
    /// x.repeat(37)
    /// ```
    Repeat(Repeat),

    /// Modular multiplication. Example syntax:
    /// ```text
    /// x**2**J * y % modN
    /// ```
    ModMul(ModMul),

    /// A constant bit register. Example syntax:
    /// ```text
    /// bit[4](0b1101)
    /// ```
    BitLiteral(BitLiteral),
}

impl fmt::Display for Expr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Expr::Variable(var) => write!(f, "{}", var),
            Expr::Slice(Slice {
                val, lower, upper, ..
            }) => write!(f, "{}[{}..{}]", *val, lower, upper),
            Expr::UnaryOp(UnaryOp { kind, val, .. }) => {
                let kind_str = match kind {
                    UnaryOpKind::Not => "~",
                };
                write!(f, "{}({})", kind_str, *val)
            }
            Expr::BinaryOp(BinaryOp {
                kind, left, right, ..
            }) => {
                let kind_str = match kind {
                    BinaryOpKind::And => "&",
                    BinaryOpKind::Or => "|",
                    BinaryOpKind::Xor => "^",
                };
                write!(f, "({}) {} ({})", *left, kind_str, *right)
            }
            Expr::ReduceOp(ReduceOp { kind, val, .. }) => {
                let kind_str = match kind {
                    BinaryOpKind::And => "&",
                    BinaryOpKind::Or => "|",
                    BinaryOpKind::Xor => "^",
                };
                write!(f, "{}({})", kind_str, *val)
            }
            Expr::RotateOp(RotateOp { kind, val, amt, .. }) => {
                let kind_str = match kind {
                    RotateOpKind::Rotr => "rotr",
                    RotateOpKind::Rotl => "rotl",
                };
                write!(f, "{}({}, {})", kind_str, *val, *amt)
            }
            Expr::Concat(Concat { left, right, .. }) => {
                write!(f, "({}) ++ ({})", *left, *right)
            }
            Expr::Repeat(Repeat { val, amt, .. }) => write!(f, "({}) * {}", *val, amt),
            Expr::ModMul(ModMul { x, j, y, mod_n, .. }) => {
                write!(f, "mod_mul({}, {}, {}, {})", x, j, *y, mod_n)
            }
            Expr::BitLiteral(blit) => write!(f, "{}", blit),
        }
    }
}
