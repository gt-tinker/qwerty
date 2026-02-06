//! Expressions for `@classical` functions.

use super::{Canonicalizable, ToPythonCode, Trivializable};
use crate::dbg::DebugLoc;
use dashu::integer::UBig;
use qwerty_ast_macros::{gen_rebuild_structs, rebuild, visitor_write};
use std::fmt;

// ----- Classical Operators -----

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryOpKind {
    Not,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryOpKind {
    And,
    Or,
    Xor,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RotateOpKind {
    Rotr,
    Rotl,
}

gen_rebuild_structs! {
    configs {
        canonicalize(
            rewrite(canonicalize_rewriter),
        ),
    }

    defs {
        /// See [`Expr::Variable`].
        #[derive(Debug, Clone, PartialEq)]
        pub struct Variable {
            pub name: String,
            pub dbg: Option<DebugLoc>,
        }

        /// See [`Expr::Slice`].
        #[derive(Debug, Clone, PartialEq)]
        pub struct Slice {
            pub val: Box<Expr>,
            pub lower: usize,
            pub upper: Option<usize>,
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

        /// See [`Expr::BitLiteral`].
        #[derive(Debug, Clone, PartialEq)]
        pub struct BitLiteral {
            pub val: UBig,
            pub n_bits: usize,
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
    }
}

impl Expr {
    /// Returns the debug location for this expression.
    pub fn get_dbg(&self) -> Option<DebugLoc> {
        let (Expr::Variable(Variable { dbg, .. })
        | Expr::Slice(Slice { dbg, .. })
        | Expr::UnaryOp(UnaryOp { dbg, .. })
        | Expr::BinaryOp(BinaryOp { dbg, .. })
        | Expr::ReduceOp(ReduceOp { dbg, .. })
        | Expr::RotateOp(RotateOp { dbg, .. })
        | Expr::Concat(Concat { dbg, .. })
        | Expr::Repeat(Repeat { dbg, .. })
        | Expr::ModMul(ModMul { dbg, .. })
        | Expr::BitLiteral(BitLiteral { dbg, .. })) = self;

        dbg.clone()
    }

    /// Called by `gen_rebuild` for the `canonicalize` config. The resulting
    /// rebuild code is called in the implementation of [`Canonicalize`] trait.
    pub(crate) fn canonicalize_rewriter(self) -> Self {
        match self {
            Expr::UnaryOp(UnaryOp {
                kind: UnaryOpKind::Not,
                val,
                dbg,
            }) => {
                match *val {
                    // ~~x --> x
                    Expr::UnaryOp(UnaryOp {
                        kind: UnaryOpKind::Not,
                        val: inner_val,
                        dbg: _,
                    }) => *inner_val,

                    other => Expr::UnaryOp(UnaryOp {
                        kind: UnaryOpKind::Not,
                        val: Box::new(other),
                        dbg,
                    }),
                }
            }

            already_canon => already_canon,
        }
    }
}

impl fmt::Display for Expr {
     fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
         visitor_write! { Expr, self,
			Expr::Variable(var) => write!(f, "{}", var),

			Expr::Slice(Slice {val, lower, upper: Some(upper_val), ..}) => {
				write!(f, "{!}[{}:{}]", *val, lower, upper_val)
			},
			Expr::Slice(Slice {val, lower, upper: None, ..}) => {
				write!(f, "{!}[{}:]", *val, lower)
			},

			Expr::UnaryOp(UnaryOp  {kind: UnaryOpKind::Not,  val, ..}) => {
				write!(f, "{}({!}", "~", *val)
			},

			Expr::BinaryOp(BinaryOp {kind: BinaryOpKind::And, left, right, ..}) => {
				write!(f, "({!}) {} ({!})", *left, "&", *right)
			},
			Expr::BinaryOp(BinaryOp {kind: BinaryOpKind::Or, left, right, ..}) => {
				write!(f, "({!}) {} ({!})", *left, "|", *right)
			},
			Expr::BinaryOp(BinaryOp {kind: BinaryOpKind::Xor, left, right, ..}) => {
				write!(f, "({!}) {} ({!})", *left, "^", *right)
			},

			Expr::ReduceOp(ReduceOp {kind: BinaryOpKind::And, val, ..}) => {
				write!(f, "({!}).{}_reduce()", *val, "and")
			},
			Expr::ReduceOp(ReduceOp {kind: BinaryOpKind::Or, val, ..}) => {
				write!(f, "({!}).{}_reduce()", *val, "or")
			},
			Expr::ReduceOp(ReduceOp {kind: BinaryOpKind::Xor, val, ..}) => {
				write!(f, "({!}).{}_reduce()", *val, "xor")
			},

			Expr::RotateOp(RotateOp {kind: RotateOpKind::Rotr, val, amt, ..}) => {
				write!(f, "{}({!}, {!})", "rotr", *val, *amt)
			}
			Expr::RotateOp(RotateOp {kind: RotateOpKind::Rotl, val, amt, ..}) => {
				write!(f, "{}({!}, {!})", "rotl", *val, *amt)
			}

			Expr::Concat(Concat { left, right, .. }) => {
                write!(f, "({!}).concat({!})", *left, *right)
            }

			Expr::Repeat(Repeat { val, amt, .. }) => write!(f, "({!}).repeat({})", *val, amt),

            Expr::ModMul(ModMul { x, j, y, mod_n, .. }) => {
                write!(f, "{}**2**{} * ({!}) % {}", x, j, *y, mod_n)
            }

            Expr::BitLiteral(blit) => write!(f, "{}", blit), // Defer to impl in ast.rs
         }
     }
}
// impl fmt::Display for Expr {
//     fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
//         match self {
//             Expr::Variable(var) => write!(f, "{}", var), // Defer to impl in ast.rs
//             Expr::Slice(Slice {
//                 val, lower, upper, ..
//             }) => {
//                 write!(f, "{}[{}:", *val, lower)?;
//                 if let Some(upper_val) = upper {
//                     write!(f, "{}", upper_val)?;
//                 }
//                 write!(f, "]")
//             }
//             Expr::UnaryOp(UnaryOp { kind, val, .. }) => {
//                 let kind_str = match kind {
//                     UnaryOpKind::Not => "~",
//                 };
//                 write!(f, "{}({})", kind_str, *val)
//             }
//             Expr::BinaryOp(BinaryOp {
//                 kind, left, right, ..
//             }) => {
//                 let kind_op = match kind {
//                     BinaryOpKind::And => "&",
//                     BinaryOpKind::Or => "|",
//                     BinaryOpKind::Xor => "^",
//                 };
//                 write!(f, "({}) {} ({})", *left, kind_op, *right)
//             }
//             Expr::ReduceOp(ReduceOp { kind, val, .. }) => {
//                 let kind_str = match kind {
//                     BinaryOpKind::And => "and",
//                     BinaryOpKind::Or => "or",
//                     BinaryOpKind::Xor => "xor",
//                 };
//                 write!(f, "({}).{}_reduce()", *val, kind_str)
//             }
//             Expr::RotateOp(RotateOp { kind, val, amt, .. }) => {
//                 let kind_str = match kind {
//                     RotateOpKind::Rotr => "rotr",
//                     RotateOpKind::Rotl => "rotl",
//                 };
//                 write!(f, "{}({}, {})", kind_str, *val, *amt)
//             }
//             Expr::Concat(Concat { left, right, .. }) => {
//                 write!(f, "({}).concat({})", *left, *right)
//             }
//             Expr::Repeat(Repeat { val, amt, .. }) => write!(f, "({}).repeat({})", *val, amt),
//             Expr::ModMul(ModMul { x, j, y, mod_n, .. }) => {
//                 write!(f, "{}**2**{} * ({}) % {}", x, j, *y, mod_n)
//             }
//             Expr::BitLiteral(blit) => write!(f, "{}", blit), // Defer to impl in ast.rs
//         }
//     }
// }

impl ToPythonCode for Expr {
    fn fmt_py(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Delegate to the `Display` implementation. For now, there are no
        // `@classical` macros
        write!(f, "{}", self)
    }
}

impl Trivializable for Expr {
    /// Closest thing we have to a canonical trivial expression is an unused
    /// bit literal.
    fn trivial(dbg: Option<DebugLoc>) -> Self {
        Self::BitLiteral(BitLiteral {
            val: UBig::ZERO,
            n_bits: 1,
            dbg,
        })
    }

    /// No expressions in the `@classical` DSL have side effects, so every
    /// expression is trivial.
    fn is_trivial(&self) -> bool {
        true
    }
}

impl Canonicalizable for Expr {
    fn canonicalize(self) -> Self {
        rebuild!(Expr, self, canonicalize)
    }
}
