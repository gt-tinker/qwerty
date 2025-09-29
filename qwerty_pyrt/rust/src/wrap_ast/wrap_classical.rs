use crate::wrap_ast::{
    py_glue::UBigWrap,
    wrap_dim_expr::DimExpr,
    wrap_type::{DebugLoc, Type},
};
use pyo3::{prelude::*, types::PyType};
use qwerty_ast::{ast, meta};
use std::fmt;

#[pyclass(eq, hash, frozen)]
#[derive(Clone, PartialEq, Hash)]
pub enum UnaryOpKind {
    Not,
}

// Intentionally not annotated with #[pymethods]: this is only for use in Rust
impl UnaryOpKind {
    fn to_ast_kind(&self) -> ast::classical::UnaryOpKind {
        match self {
            UnaryOpKind::Not => ast::classical::UnaryOpKind::Not,
        }
    }
}

#[pyclass(eq, hash, frozen)]
#[derive(Clone, PartialEq, Hash)]
pub enum BinaryOpKind {
    And,
    Or,
    Xor,
}

// Intentionally not annotated with #[pymethods]: this is only for use in Rust
impl BinaryOpKind {
    fn to_ast_kind(&self) -> ast::classical::BinaryOpKind {
        match self {
            BinaryOpKind::And => ast::classical::BinaryOpKind::And,
            BinaryOpKind::Or => ast::classical::BinaryOpKind::Or,
            BinaryOpKind::Xor => ast::classical::BinaryOpKind::Xor,
        }
    }
}

#[pyclass(str, eq)]
#[derive(Clone, PartialEq)]
pub struct ClassicalExpr {
    pub expr: meta::classical::MetaExpr,
}

impl ClassicalExpr {
    pub fn new(expr: meta::classical::MetaExpr) -> Self {
        Self { expr }
    }
}

impl fmt::Display for ClassicalExpr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.expr)
    }
}

#[pymethods]
impl ClassicalExpr {
    #[classmethod]
    fn new_mod(
        _cls: &Bound<'_, PyType>,
        dividend: ClassicalExpr,
        divisor: DimExpr,
        dbg: Option<DebugLoc>,
    ) -> Self {
        Self {
            expr: meta::classical::MetaExpr::Mod {
                dividend: Box::new(dividend.expr),
                divisor: divisor.dim_expr,
                dbg: dbg.map(|dbg| dbg.dbg),
            },
        }
    }

    #[classmethod]
    fn new_variable(_cls: &Bound<'_, PyType>, name: String, dbg: Option<DebugLoc>) -> Self {
        Self {
            expr: meta::classical::MetaExpr::Variable {
                name,
                dbg: dbg.map(|dbg| dbg.dbg),
            },
        }
    }

    #[classmethod]
    fn new_slice(
        _cls: &Bound<'_, PyType>,
        val: ClassicalExpr,
        lower: DimExpr,
        upper: Option<DimExpr>,
        dbg: Option<DebugLoc>,
    ) -> Self {
        Self {
            expr: meta::classical::MetaExpr::Slice {
                val: Box::new(val.expr),
                lower: lower.dim_expr,
                upper: upper.map(|upper_dim_expr| upper_dim_expr.dim_expr),
                dbg: dbg.map(|dbg| dbg.dbg),
            },
        }
    }

    #[classmethod]
    fn new_unary_op(
        _cls: &Bound<'_, PyType>,
        kind: UnaryOpKind,
        val: ClassicalExpr,
        dbg: Option<DebugLoc>,
    ) -> Self {
        Self {
            expr: meta::classical::MetaExpr::UnaryOp {
                kind: kind.to_ast_kind(),
                val: Box::new(val.expr),
                dbg: dbg.map(|dbg| dbg.dbg),
            },
        }
    }

    #[classmethod]
    fn new_binary_op(
        _cls: &Bound<'_, PyType>,
        kind: BinaryOpKind,
        left: ClassicalExpr,
        right: ClassicalExpr,
        dbg: Option<DebugLoc>,
    ) -> Self {
        Self {
            expr: meta::classical::MetaExpr::BinaryOp {
                kind: kind.to_ast_kind(),
                left: Box::new(left.expr),
                right: Box::new(right.expr),
                dbg: dbg.map(|dbg| dbg.dbg),
            },
        }
    }

    #[classmethod]
    fn new_reduce_op(
        _cls: &Bound<'_, PyType>,
        kind: BinaryOpKind,
        val: ClassicalExpr,
        dbg: Option<DebugLoc>,
    ) -> Self {
        Self {
            expr: meta::classical::MetaExpr::ReduceOp {
                kind: kind.to_ast_kind(),
                val: Box::new(val.expr),
                dbg: dbg.map(|dbg| dbg.dbg),
            },
        }
    }

    #[classmethod]
    fn new_mod_mul(
        _cls: &Bound<'_, PyType>,
        x: DimExpr,
        j: DimExpr,
        y: ClassicalExpr,
        mod_n: DimExpr,
        dbg: Option<DebugLoc>,
    ) -> Self {
        Self {
            expr: meta::classical::MetaExpr::ModMul {
                x: x.dim_expr,
                j: j.dim_expr,
                y: Box::new(y.expr),
                mod_n: mod_n.dim_expr,
                dbg: dbg.map(|dbg| dbg.dbg),
            },
        }
    }

    #[classmethod]
    fn new_bit_literal(
        _cls: &Bound<'_, PyType>,
        val: UBigWrap,
        n_bits: usize,
        dbg: Option<DebugLoc>,
    ) -> Self {
        let dbg = dbg.map(|dbg| dbg.dbg);
        Self {
            expr: meta::classical::MetaExpr::BitLiteral {
                val: val.0,
                n_bits: meta::DimExpr::DimConst {
                    val: n_bits.into(),
                    dbg: dbg.clone(),
                },
                dbg,
            },
        }
    }

    #[classmethod]
    fn new_repeat(
        _cls: &Bound<'_, PyType>,
        val: ClassicalExpr,
        amt: DimExpr,
        dbg: Option<DebugLoc>,
    ) -> Self {
        Self {
            expr: meta::classical::MetaExpr::Repeat {
                val: Box::new(val.expr),
                amt: amt.dim_expr,
                dbg: dbg.map(|dbg| dbg.dbg),
            },
        }
    }

    #[classmethod]
    fn new_concat(
        _cls: &Bound<'_, PyType>,
        left: ClassicalExpr,
        right: ClassicalExpr,
        dbg: Option<DebugLoc>,
    ) -> Self {
        Self {
            expr: meta::classical::MetaExpr::Concat {
                left: Box::new(left.expr),
                right: Box::new(right.expr),
                dbg: dbg.map(|dbg| dbg.dbg),
            },
        }
    }

    /// Return the Debug form of this Expr from __repr__(). By contrast,
    /// __str__() returns the Display form.
    pub fn __repr__(&self) -> String {
        format!("{:?}", self.expr)
    }
}

#[pyclass(str, eq)]
#[derive(Clone, PartialEq)]
pub struct ClassicalStmt {
    pub stmt: meta::classical::MetaStmt,
}

impl fmt::Display for ClassicalStmt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.stmt)
    }
}

#[pymethods]
impl ClassicalStmt {
    #[classmethod]
    fn new_expr(_cls: &Bound<'_, PyType>, expr: ClassicalExpr) -> Self {
        Self {
            stmt: meta::classical::MetaStmt::Expr { expr: expr.expr },
        }
    }

    #[classmethod]
    fn new_assign(
        _cls: &Bound<'_, PyType>,
        lhs: String,
        rhs: ClassicalExpr,
        dbg: Option<DebugLoc>,
    ) -> Self {
        Self {
            stmt: meta::classical::MetaStmt::Assign {
                lhs,
                rhs: rhs.expr,
                dbg: dbg.map(|dbg| dbg.dbg),
            },
        }
    }

    #[classmethod]
    fn new_unpack_assign(
        _cls: &Bound<'_, PyType>,
        lhs: Vec<String>,
        rhs: ClassicalExpr,
        dbg: Option<DebugLoc>,
    ) -> Self {
        Self {
            stmt: meta::classical::MetaStmt::UnpackAssign {
                lhs,
                rhs: rhs.expr,
                dbg: dbg.map(|dbg| dbg.dbg),
            },
        }
    }

    #[classmethod]
    fn new_return(_cls: &Bound<'_, PyType>, val: ClassicalExpr, dbg: Option<DebugLoc>) -> Self {
        Self {
            stmt: meta::classical::MetaStmt::Return {
                val: val.expr,
                dbg: dbg.map(|dbg| dbg.dbg),
            },
        }
    }

    /// Return the Debug form of this Stmt from __repr__(). By contrast,
    /// __str__() returns the Display form.
    pub fn __repr__(&self) -> String {
        format!("{:?}", self.stmt)
    }
}

#[pyclass]
#[derive(Clone)]
pub struct ClassicalFunctionDef {
    pub function_def: meta::MetaFunctionDef<meta::classical::MetaStmt>,
}

#[pymethods]
impl ClassicalFunctionDef {
    #[new]
    fn new(
        name: String,
        args: Vec<(Option<Type>, String)>,
        ret_type: Option<Type>,
        body: Vec<ClassicalStmt>,
        is_rev: bool,
        dim_vars: Vec<String>,
        dbg: Option<DebugLoc>,
    ) -> Self {
        Self {
            function_def: meta::MetaFunctionDef {
                name,
                args: args
                    .into_iter()
                    .map(|(arg_ty, arg_name)| (arg_ty.map(|arg_ty| arg_ty.ty), arg_name))
                    .collect(),
                ret_type: ret_type.map(|ret_type| ret_type.ty.clone()),
                body: body.iter().map(|stmt| stmt.stmt.clone()).collect(),
                is_rev,
                dim_vars,
                dbg: dbg.map(|dbg| dbg.dbg),
            },
        }
    }

    fn get_name(&self) -> String {
        self.function_def.name.to_string()
    }
}
