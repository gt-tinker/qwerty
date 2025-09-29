use crate::{
    ast::{
        self,
        classical::{BinaryOpKind, UnaryOpKind},
    },
    dbg::DebugLoc,
    error::{LowerError, LowerErrorKind},
    meta::DimExpr,
};
use dashu::integer::UBig;
use std::fmt;

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
    /// x.repeat(y)
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
    pub fn extract(&self) -> Result<ast::classical::Expr, LowerError> {
        match self {
            // For now, this should be folded into a bitwise AND.
            MetaExpr::Mod { dbg, .. } => Err(LowerError {
                kind: LowerErrorKind::NotFullyFolded,
                dbg: dbg.clone(),
            }),
            MetaExpr::Variable { name, dbg } => Ok(ast::classical::Expr::Variable(ast::Variable {
                name: name.to_string(),
                dbg: dbg.clone(),
            })),
            MetaExpr::Slice {
                val,
                lower,
                upper,
                dbg,
            } => val.extract().and_then(|ast_val| {
                lower.extract().and_then(|lower_int| {
                    upper
                        .as_ref()
                        .map(DimExpr::extract)
                        .transpose()
                        .map(|upper_int_opt| {
                            ast::classical::Expr::Slice(ast::classical::Slice {
                                val: Box::new(ast_val),
                                lower: lower_int,
                                upper: upper_int_opt,
                                dbg: dbg.clone(),
                            })
                        })
                })
            }),
            MetaExpr::UnaryOp { kind, val, dbg } => val.extract().map(|ast_val| {
                ast::classical::Expr::UnaryOp(ast::classical::UnaryOp {
                    kind: *kind,
                    val: Box::new(ast_val),
                    dbg: dbg.clone(),
                })
            }),
            MetaExpr::BinaryOp {
                kind,
                left,
                right,
                dbg,
            } => left.extract().and_then(|ast_left| {
                right.extract().map(|ast_right| {
                    ast::classical::Expr::BinaryOp(ast::classical::BinaryOp {
                        kind: *kind,
                        left: Box::new(ast_left),
                        right: Box::new(ast_right),
                        dbg: dbg.clone(),
                    })
                })
            }),
            MetaExpr::ReduceOp { kind, val, dbg } => val.extract().map(|ast_val| {
                ast::classical::Expr::ReduceOp(ast::classical::ReduceOp {
                    kind: *kind,
                    val: Box::new(ast_val),
                    dbg: dbg.clone(),
                })
            }),
            MetaExpr::ModMul {
                x,
                j,
                y,
                mod_n,
                dbg,
            } => x.extract().and_then(|x_int| {
                j.extract().and_then(|j_int| {
                    y.extract().and_then(|ast_y| {
                        mod_n.extract().map(|mod_n_int| {
                            ast::classical::Expr::ModMul(ast::classical::ModMul {
                                x: x_int,
                                j: j_int,
                                y: Box::new(ast_y),
                                mod_n: mod_n_int,
                                dbg: dbg.clone(),
                            })
                        })
                    })
                })
            }),
            MetaExpr::BitLiteral { val, n_bits, dbg } => n_bits.extract().map(|n_bits_int| {
                ast::classical::Expr::BitLiteral(ast::BitLiteral {
                    val: val.clone(),
                    n_bits: n_bits_int,
                    dbg: dbg.clone(),
                })
            }),
            MetaExpr::Repeat { val, amt, dbg } => val.extract().and_then(|ast_val| {
                amt.extract().map(|amount_int| {
                    ast::classical::Expr::Repeat(ast::classical::Repeat {
                        val: Box::new(ast_val),
                        amt: amount_int,
                        dbg: dbg.clone(),
                    })
                })
            }),
            MetaExpr::Concat { left, right, dbg } => left.extract().and_then(|ast_left| {
                right.extract().map(|ast_right| {
                    ast::classical::Expr::Concat(ast::classical::Concat {
                        left: Box::new(ast_left),
                        right: Box::new(ast_right),
                        dbg: dbg.clone(),
                    })
                })
            }),
        }
    }
}

// TODO: don't duplicate this code with classical.rs
impl fmt::Display for MetaExpr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MetaExpr::Mod {
                divisor, dividend, ..
            } => write!(f, "({}) % ({})", divisor, dividend),
            MetaExpr::Variable { name, .. } => write!(f, "{}", name),
            MetaExpr::Slice {
                val, lower, upper, ..
            } => {
                write!(f, "({})[{}:", val, lower)?;
                if let Some(upper_dimexpr) = upper {
                    write!(f, "{}", upper_dimexpr)?;
                }
                write!(f, "]")
            }
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
            MetaExpr::ModMul { x, j, y, mod_n, .. } => {
                write!(f, "({})**2**({}) * ({}) % ({})", x, j, y, mod_n)
            }
            MetaExpr::Repeat { val, amt, .. } => {
                write!(f, "({}).repeat({})", val, amt)
            }
            MetaExpr::Concat { left, right, .. } => {
                write!(f, "({}).concat({})", left, right)
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

impl MetaStmt {
    /// Extracts a plain-AST `@classical` statement from a metaQwerty
    /// `@classical` statement.
    pub fn extract(&self) -> Result<ast::Stmt<ast::classical::Expr>, LowerError> {
        match self {
            MetaStmt::Expr { expr } => expr.extract().map(|ast_expr| {
                ast::Stmt::Expr(ast::StmtExpr {
                    expr: ast_expr,
                    dbg: expr.get_dbg(),
                })
            }),
            MetaStmt::Assign { lhs, rhs, dbg } => rhs.extract().map(|ast_rhs| {
                ast::Stmt::Assign(ast::Assign {
                    lhs: lhs.to_string(),
                    rhs: ast_rhs,
                    dbg: dbg.clone(),
                })
            }),
            MetaStmt::UnpackAssign { lhs, rhs, dbg } => rhs.extract().map(|ast_rhs| {
                ast::Stmt::UnpackAssign(ast::UnpackAssign {
                    lhs: lhs.clone(),
                    rhs: ast_rhs,
                    dbg: dbg.clone(),
                })
            }),
            MetaStmt::Return { val, dbg } => val.extract().map(|ast_val| {
                ast::Stmt::Return(ast::Return {
                    val: ast_val,
                    dbg: dbg.clone(),
                })
            }),
        }
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
