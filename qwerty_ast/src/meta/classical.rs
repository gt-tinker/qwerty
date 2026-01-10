use crate::{
    ast::{
        self,
        classical::{BinaryOpKind, UnaryOpKind},
    },
    dbg::DebugLoc,
    error::{LowerError, LowerErrorKind},
    meta::{DimExpr, DimVar, Progress, expand::MacroEnv},
};
use dashu::integer::UBig;
use qwerty_ast_macros::{gen_rebuild, rebuild, rewrite_match, rewrite_ty, visitor_match};
use std::fmt;

#[gen_rebuild {
    substitute_dim_var(
        more_copied_args(dim_var: &DimVar, new_dim_expr: &DimExpr),
        recurse_attrs,
    ),
    expand(
        rewrite(expand_rewriter),
        progress(Progress),
        more_copied_args(env: &mut MacroEnv),
        result_err(LowerError),
        recurse_attrs,
    ),
    extract(
        rewrite(extract_rewriter),
        rewrite_to(
            MetaExpr => ast::classical::Expr,
            DimExpr => usize,
        ),
        result_err(LowerError),
        recurse_attrs,
    ),
}]
#[derive(Debug, Clone, PartialEq)]
pub enum MetaExpr {
    /// Take the modulus of a bit register. Example syntax:
    /// ```text
    /// x % 4
    /// ```
    /// If the divisor is a power of two, this will get expanded to a
    /// [`MetaExpr::BinaryOp`] of kind `And`.
    Mod {
        dividend: Box<MetaExpr>,
        divisor: DimExpr,
        dbg: Option<DebugLoc>,
    },

    /// A variable name used in an expression. Example syntax:
    /// ```text
    /// my_var
    /// ```
    Variable { name: String, dbg: Option<DebugLoc> },

    /// Extract a subsequence of bits from a register. Example syntax:
    /// ```text
    /// my_bit_reg[2:4]
    /// ```
    Slice {
        val: Box<MetaExpr>,
        lower: DimExpr,
        upper: Option<DimExpr>,
        dbg: Option<DebugLoc>,
    },

    /// A unary bitwise operation. Example syntax:
    /// ```text
    /// ~x
    /// ```
    UnaryOp {
        #[gen_rebuild::skip_recurse]
        kind: UnaryOpKind,
        val: Box<MetaExpr>,
        dbg: Option<DebugLoc>,
    },

    /// A binary bitwise operation. Example syntax:
    /// ```text
    /// x & y
    /// ```
    BinaryOp {
        #[gen_rebuild::skip_recurse]
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
        #[gen_rebuild::skip_recurse]
        kind: BinaryOpKind,
        val: Box<MetaExpr>,
        dbg: Option<DebugLoc>,
    },

    /// Modular multiplication. Example syntax:
    /// ```text
    /// x**2**J * y % modN
    /// ```
    ModMul {
        x: DimExpr,
        j: DimExpr,
        y: Box<MetaExpr>,
        mod_n: DimExpr,
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

    /// Repeat a bit. Example syntax:
    /// ```text
    /// val.repeat(amt)
    /// ```
    Repeat {
        val: Box<MetaExpr>,
        amt: DimExpr,
        dbg: Option<DebugLoc>,
    },

    /// Concatenate two bit registers. Example syntax:
    /// ```text
    /// x.concat(y)
    /// ```
    Concat {
        left: Box<MetaExpr>,
        right: Box<MetaExpr>,
        dbg: Option<DebugLoc>,
    },
}

impl MetaExpr {
    /// Returns the debug location for this expression.
    pub fn get_dbg(&self) -> Option<DebugLoc> {
        match self {
            MetaExpr::Mod { dbg, .. }
            | MetaExpr::Variable { dbg, .. }
            | MetaExpr::Slice { dbg, .. }
            | MetaExpr::UnaryOp { dbg, .. }
            | MetaExpr::BinaryOp { dbg, .. }
            | MetaExpr::ReduceOp { dbg, .. }
            | MetaExpr::ModMul { dbg, .. }
            | MetaExpr::BitLiteral { dbg, .. }
            | MetaExpr::Repeat { dbg, .. }
            | MetaExpr::Concat { dbg, .. } => dbg.clone(),
        }
    }

    /// Extracts a plain-AST `@classical` expression from a metaQwerty
    /// `@classical` expression.
    pub fn extract(self) -> Result<ast::classical::Expr, LowerError> {
        rebuild!(MetaExpr, self, extract)
    }

    pub(crate) fn extract_rewriter(
        rewritten: rewrite_ty!(MetaExpr, extract),
    ) -> Result<ast::classical::Expr, LowerError> {
        rewrite_match! {MetaExpr, extract, rewritten,
            // For now, this should be folded into a bitwise AND.
            Mod { dbg, .. } => Err(LowerError {
                kind: LowerErrorKind::NotFullyFolded,
                dbg,
            }),
            Variable { name, dbg } => {
                Ok(ast::classical::Expr::Variable(ast::classical::Variable { name, dbg }))
            }
            Slice { val, lower, upper, dbg, } => {
                Ok(ast::classical::Expr::Slice(ast::classical::Slice {
                    val: Box::new(val),
                    lower,
                    upper,
                    dbg,
                }))
            }
            UnaryOp { kind, val, dbg } => {
                Ok(ast::classical::Expr::UnaryOp(ast::classical::UnaryOp {
                    kind,
                    val: Box::new(val),
                    dbg,
                }))
            }
            BinaryOp { kind, left, right, dbg } => {
                Ok(ast::classical::Expr::BinaryOp(ast::classical::BinaryOp {
                    kind,
                    left: Box::new(left),
                    right: Box::new(right),
                    dbg,
                }))
            }
            ReduceOp { kind, val, dbg } => {
                Ok(ast::classical::Expr::ReduceOp(ast::classical::ReduceOp {
                    kind,
                    val: Box::new(val),
                    dbg,
                }))
            }
            ModMul { x, j, y, mod_n, dbg } => {
                Ok(ast::classical::Expr::ModMul(ast::classical::ModMul {
                    x,
                    j,
                    y: Box::new(y),
                    mod_n,
                    dbg,
                }))
            }
            BitLiteral { val, n_bits, dbg } => {
                Ok(ast::classical::Expr::BitLiteral(ast::classical::BitLiteral {
                    val,
                    n_bits,
                    dbg,
                }))
            }
            Repeat { val, amt, dbg } => {
                Ok(ast::classical::Expr::Repeat(ast::classical::Repeat {
                    val: Box::new(val),
                    amt,
                    dbg,
                }))
            }
            Concat { left, right, dbg } => {
                Ok(ast::classical::Expr::Concat(ast::classical::Concat {
                    left: Box::new(left),
                    right: Box::new(right),
                    dbg,
                }))
            }
        }
    }
}

// TODO: don't duplicate this code with classical.rs
impl fmt::Display for MetaExpr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        visitor_match! {MetaExpr, self,
            MetaExpr::Mod { dividend, divisor, .. } => {
                write!(f, "(")?;
                visit!(*dividend);
                write!(f, ") % ({})", divisor)?;
            }
            MetaExpr::Variable { name, .. } => write!(f, "{}", name)?,
            MetaExpr::Slice { val, lower, upper, .. } => {
                write!(f, "(")?;
                visit!(*val);
                write!(f, ")[{}:", lower)?;
                if let Some(upper_dimexpr) = upper {
                    write!(f, "{}", upper_dimexpr)?;
                }
                write!(f, "]")?;
            }
            MetaExpr::UnaryOp { kind, val, .. } => {
                let kind_str = match kind {
                    UnaryOpKind::Not => "~",
                };
                write!(f, "{}(", kind_str)?;
                visit!(*val);
                write!(f, ")")?;
            }
            MetaExpr::BinaryOp {
                kind, left, right, ..
            } => {
                write!(f, "(")?;
                visit!(*left);
                let kind_str = match kind {
                    BinaryOpKind::And => "&",
                    BinaryOpKind::Or => "|",
                    BinaryOpKind::Xor => "^",
                };
                write!(f, ") {} (", kind_str)?;
                visit!(*right);
                write!(f, ")")?;
            }
            MetaExpr::ReduceOp { kind, val, .. } => {
                write!(f, "(")?;
                visit!(*val);
                let kind_str = match kind {
                    BinaryOpKind::And => "and",
                    BinaryOpKind::Or => "or",
                    BinaryOpKind::Xor => "xor",
                };
                write!(f, ").{}_reduce()", kind_str)?;
            }
            MetaExpr::ModMul { x, j, y, mod_n, .. } => {
                write!(f, "({})**2**({}) * (", x, j)?;
                visit!(*y);
                write!(f, ") % ({})", mod_n)?;
            }
            MetaExpr::BitLiteral { val, n_bits, .. } => {
                write!(f, "bit[{}](0b{:b})", n_bits, val)?;
            }
            MetaExpr::Repeat { val, amt, .. } => {
                write!(f, "(")?;
                visit!(*val);
                write!(f, ").repeat({})", amt)?;
            }
            MetaExpr::Concat { left, right, .. } => {
                write!(f, "(")?;
                visit!(*left);
                write!(f, ").concat(")?;
                visit!(*right);
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
        more_copied_args(env: &mut MacroEnv),
        result_err(LowerError),
        recurse_attrs,
    ),
    extract(
        rewrite(extract_rewriter),
        rewrite_to(
            MetaStmt => ast::Stmt<ast::classical::Expr>,
            MetaExpr => ast::classical::Expr,
        ),
        result_err(LowerError),
        recurse_attrs,
    ),
}]
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

impl MetaStmt {
    /// Extracts a plain-AST `@classical` statement from a metaQwerty
    /// `@classical` statement.
    pub fn extract(self) -> Result<ast::Stmt<ast::classical::Expr>, LowerError> {
        rebuild!(MetaStmt, self, extract)
    }

    pub(crate) fn extract_rewriter(
        rewritten: rewrite_ty!(MetaStmt, extract),
    ) -> Result<ast::Stmt<ast::classical::Expr>, LowerError> {
        Ok(rewrite_match! {MetaStmt, extract, rewritten,
            Expr { expr } => {
                let dbg = expr.get_dbg();
                ast::Stmt::Expr(ast::StmtExpr { expr, dbg })
            }
            Assign { lhs, rhs, dbg } => {
                ast::Stmt::Assign(ast::Assign { lhs, rhs, dbg })
            }
            UnpackAssign { lhs, rhs, dbg } => {
                ast::Stmt::UnpackAssign(ast::UnpackAssign { lhs, rhs, dbg })
            }
            Return { val, dbg } => {
                ast::Stmt::Return(ast::Return { val, dbg })
            }
        })
    }
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
