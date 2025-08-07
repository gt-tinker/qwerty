use crate::wrap_ast::{
    py_glue::UBigWrap,
    ty::{DebugLoc, Type},
};
use pyo3::{prelude::*, types::PyType};
use qwerty_ast::ast;
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
    pub(crate) expr: ast::classical::Expr,
}

impl ClassicalExpr {
    pub fn new(expr: ast::classical::Expr) -> Self {
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
    fn new_variable(_cls: &Bound<'_, PyType>, name: String, dbg: Option<DebugLoc>) -> Self {
        Self {
            expr: ast::classical::Expr::Variable(ast::Variable {
                name,
                dbg: dbg.map(|dbg| dbg.dbg),
            }),
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
            expr: ast::classical::Expr::UnaryOp(ast::classical::UnaryOp {
                kind: kind.to_ast_kind(),
                val: Box::new(val.expr),
                dbg: dbg.map(|dbg| dbg.dbg),
            }),
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
            expr: ast::classical::Expr::BinaryOp(ast::classical::BinaryOp {
                kind: kind.to_ast_kind(),
                left: Box::new(left.expr),
                right: Box::new(right.expr),
                dbg: dbg.map(|dbg| dbg.dbg),
            }),
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
            expr: ast::classical::Expr::ReduceOp(ast::classical::ReduceOp {
                kind: kind.to_ast_kind(),
                val: Box::new(val.expr),
                dbg: dbg.map(|dbg| dbg.dbg),
            }),
        }
    }

    #[classmethod]
    fn new_bit_literal(
        _cls: &Bound<'_, PyType>,
        val: UBigWrap,
        n_bits: usize,
        dbg: Option<DebugLoc>,
    ) -> Self {
        Self {
            expr: ast::classical::Expr::BitLiteral(ast::BitLiteral {
                val: val.0,
                n_bits,
                dbg: dbg.map(|dbg| dbg.dbg),
            }),
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
    pub(crate) stmt: ast::Stmt<ast::classical::Expr>,
}

impl fmt::Display for ClassicalStmt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.stmt)
    }
}

#[pymethods]
impl ClassicalStmt {
    #[classmethod]
    fn new_expr(_cls: &Bound<'_, PyType>, expr: ClassicalExpr, dbg: Option<DebugLoc>) -> Self {
        Self {
            stmt: ast::Stmt::Expr(ast::StmtExpr {
                expr: expr.expr,
                dbg: dbg.map(|dbg| dbg.dbg),
            }),
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
            stmt: ast::Stmt::Assign(ast::Assign {
                lhs,
                rhs: rhs.expr,
                dbg: dbg.map(|dbg| dbg.dbg),
            }),
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
            stmt: ast::Stmt::UnpackAssign(ast::UnpackAssign {
                lhs,
                rhs: rhs.expr,
                dbg: dbg.map(|dbg| dbg.dbg),
            }),
        }
    }

    #[classmethod]
    fn new_return(_cls: &Bound<'_, PyType>, val: ClassicalExpr, dbg: Option<DebugLoc>) -> Self {
        Self {
            stmt: ast::Stmt::Return(ast::Return {
                val: val.expr,
                dbg: dbg.map(|dbg| dbg.dbg),
            }),
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
    pub function_def: ast::FunctionDef<ast::classical::Expr>,
}

#[pymethods]
impl ClassicalFunctionDef {
    #[new]
    fn new(
        name: String,
        args: Vec<(Type, String)>,
        ret_type: Type,
        body: Vec<ClassicalStmt>,
        is_rev: bool,
        dbg: Option<DebugLoc>,
    ) -> Self {
        Self {
            function_def: ast::FunctionDef {
                name,
                args: args
                    .iter()
                    .map(|(arg_ty, arg_name)| (arg_ty.ty.clone(), arg_name.to_string()))
                    .collect(),
                ret_type: ret_type.ty.clone(),
                body: body.iter().map(|stmt| stmt.stmt.clone()).collect(),
                is_rev,
                dbg: dbg.map(|dbg| dbg.dbg),
            },
        }
    }

    fn get_name(&self) -> String {
        self.function_def.name.to_string()
    }
}
