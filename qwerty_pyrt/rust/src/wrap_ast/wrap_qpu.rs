use crate::wrap_ast::{
    py_glue::{ProgErrKind, UBigWrap, get_err},
    wrap_type::{DebugLoc, Type, TypeEnv},
};
use pyo3::{prelude::*, types::PyType};
use qwerty_ast::{ast, meta};
use std::fmt;

/// A "plain" Qwerty expression AST node, that is, an expression without any
/// metaQwerty features (e.g., `__SYM_STD0__()` instead of `'0'`).
#[pyclass(str, eq)]
#[derive(Clone, PartialEq)]
pub struct PlainQpuExpr {
    pub expr: ast::qpu::Expr,
}

impl fmt::Display for PlainQpuExpr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.expr)
    }
}

/// A "plain" Qwerty statement AST node, that is, a statement without any
/// metaQwerty features (e.g., no basis alias definitions such as
/// `std = {'0','1'}`).
#[pyclass(str, eq)]
#[derive(Clone, PartialEq)]
pub struct PlainQpuStmt {
    pub stmt: ast::Stmt<ast::qpu::Expr>,
}

#[pymethods]
impl PlainQpuStmt {
    /// Perform type checking on this statement, but do not allow return
    /// statements. Used in the REPL.
    pub fn type_check_no_ret(&self, py: Python<'_>, env: &mut TypeEnv) -> PyResult<()> {
        self.stmt
            .typecheck(&mut env.env, /*expected_ret_type=*/ None)
            // Discard compute kind for now, since Python does not need it
            .map(|_compute_kind| ())
            .map_err(|err| get_err(py, ProgErrKind::Type, err.kind.to_string(), err.dbg))
    }
}

impl fmt::Display for PlainQpuStmt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.stmt)
    }
}

#[pyclass]
#[derive(Clone)]
pub struct Vector {
    vec: meta::qpu::MetaVector,
}

#[pymethods]
impl Vector {
    #[classmethod]
    fn new_vector_symbol(_cls: &Bound<'_, PyType>, sym: char, dbg: Option<DebugLoc>) -> Self {
        Self {
            vec: meta::qpu::MetaVector::VectorSymbol {
                sym,
                dbg: dbg.map(|dbg| dbg.dbg),
            },
        }
    }

    #[classmethod]
    fn new_zero_vector(_cls: &Bound<'_, PyType>, dbg: Option<DebugLoc>) -> Self {
        Self {
            vec: meta::qpu::MetaVector::ZeroVector {
                dbg: dbg.map(|dbg| dbg.dbg),
            },
        }
    }

    #[classmethod]
    fn new_one_vector(_cls: &Bound<'_, PyType>, dbg: Option<DebugLoc>) -> Self {
        Self {
            vec: meta::qpu::MetaVector::OneVector {
                dbg: dbg.map(|dbg| dbg.dbg),
            },
        }
    }

    #[classmethod]
    fn new_pad_vector(_cls: &Bound<'_, PyType>, dbg: Option<DebugLoc>) -> Self {
        Self {
            vec: meta::qpu::MetaVector::PadVector {
                dbg: dbg.map(|dbg| dbg.dbg),
            },
        }
    }

    #[classmethod]
    fn new_target_vector(_cls: &Bound<'_, PyType>, dbg: Option<DebugLoc>) -> Self {
        Self {
            vec: meta::qpu::MetaVector::TargetVector {
                dbg: dbg.map(|dbg| dbg.dbg),
            },
        }
    }

    #[classmethod]
    fn new_uniform_vector_superpos(
        _cls: &Bound<'_, PyType>,
        q1: Vector,
        q2: Vector,
        dbg: Option<DebugLoc>,
    ) -> Self {
        Self {
            vec: meta::qpu::MetaVector::UniformVectorSuperpos {
                q1: Box::new(q1.vec.clone()),
                q2: Box::new(q2.vec.clone()),
                dbg: dbg.map(|dbg| dbg.dbg),
            },
        }
    }

    #[classmethod]
    fn new_vector_bi_tensor(
        _cls: &Bound<'_, PyType>,
        left: Vector,
        right: Vector,
        dbg: Option<DebugLoc>,
    ) -> Self {
        Self {
            vec: meta::qpu::MetaVector::VectorBiTensor {
                left: Box::new(left.vec),
                right: Box::new(right.vec),
                dbg: dbg.map(|dbg| dbg.dbg),
            },
        }
    }

    #[classmethod]
    fn new_vector_tilt(
        _cls: &Bound<'_, PyType>,
        q: Vector,
        angle_deg: f64,
        dbg: Option<DebugLoc>,
    ) -> Self {
        let dbg = dbg.map(|dbg| dbg.dbg);
        Self {
            vec: meta::qpu::MetaVector::VectorTilt {
                q: Box::new(q.vec.clone()),
                angle_deg: meta::qpu::FloatExpr::FloatConst {
                    val: angle_deg,
                    dbg: dbg.clone(),
                },
                dbg,
            },
        }
    }

    #[classmethod]
    fn new_vector_unit(_cls: &Bound<'_, PyType>, dbg: Option<DebugLoc>) -> Self {
        Self {
            vec: meta::qpu::MetaVector::VectorUnit {
                dbg: dbg.map(|dbg| dbg.dbg),
            },
        }
    }
}

#[pyclass]
#[derive(Clone)]
pub struct BasisGenerator {
    generator: meta::qpu::MetaBasisGenerator,
}

#[pymethods]
impl BasisGenerator {
    #[classmethod]
    fn new_revolve(
        _cls: &Bound<'_, PyType>,
        v1: Vector,
        v2: Vector,
        dbg: Option<DebugLoc>,
    ) -> Self {
        Self {
            generator: meta::qpu::MetaBasisGenerator::Revolve {
                v1: v1.vec,
                v2: v2.vec,
                dbg: dbg.map(|dbg| dbg.dbg),
            },
        }
    }
}

#[pyclass]
#[derive(Clone)]
pub struct Basis {
    basis: meta::qpu::MetaBasis,
}

#[pymethods]
impl Basis {
    #[classmethod]
    fn new_empty_basis_literal(_cls: &Bound<'_, PyType>, dbg: Option<DebugLoc>) -> Self {
        Self {
            basis: meta::qpu::MetaBasis::EmptyBasisLiteral {
                dbg: dbg.map(|dbg| dbg.dbg),
            },
        }
    }

    #[classmethod]
    fn new_basis_literal(
        _cls: &Bound<'_, PyType>,
        vecs: Vec<Vector>,
        dbg: Option<DebugLoc>,
    ) -> Self {
        Self {
            basis: meta::qpu::MetaBasis::BasisLiteral {
                vecs: vecs.iter().map(|vec| vec.vec.clone()).collect(),
                dbg: dbg.map(|dbg| dbg.dbg),
            },
        }
    }

    #[classmethod]
    fn new_basis_bi_tensor(
        _cls: &Bound<'_, PyType>,
        left: Basis,
        right: Basis,
        dbg: Option<DebugLoc>,
    ) -> Self {
        Self {
            basis: meta::qpu::MetaBasis::BasisBiTensor {
                left: Box::new(left.basis),
                right: Box::new(right.basis),
                dbg: dbg.map(|dbg| dbg.dbg),
            },
        }
    }

    #[classmethod]
    fn new_apply_basis_generator(
        _cls: &Bound<'_, PyType>,
        basis: Basis,
        generator: BasisGenerator,
        dbg: Option<DebugLoc>,
    ) -> Self {
        Self {
            basis: meta::qpu::MetaBasis::ApplyBasisGenerator {
                basis: Box::new(basis.basis),
                generator: generator.generator,
                dbg: dbg.map(|dbg| dbg.dbg),
            },
        }
    }
}

#[pyclass(eq, hash, frozen)]
#[derive(Clone, PartialEq, Hash)]
pub enum EmbedKind {
    Sign,
    Xor,
    InPlace,
}

// Intentionally not annotated with #[pymethods]: this is only for use in Rust
impl EmbedKind {
    fn to_ast_kind(&self) -> ast::qpu::EmbedKind {
        match self {
            EmbedKind::Sign => ast::qpu::EmbedKind::Sign,
            EmbedKind::Xor => ast::qpu::EmbedKind::Xor,
            EmbedKind::InPlace => ast::qpu::EmbedKind::InPlace,
        }
    }
}

#[pyclass(str, eq)]
#[derive(Clone, PartialEq)]
pub struct QpuExpr {
    pub expr: meta::qpu::MetaExpr,
}

impl fmt::Display for QpuExpr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.expr)
    }
}

#[pymethods]
impl QpuExpr {
    #[classmethod]
    fn new_variable(_cls: &Bound<'_, PyType>, name: String, dbg: Option<DebugLoc>) -> Self {
        Self {
            expr: meta::qpu::MetaExpr::Variable {
                name,
                dbg: dbg.map(|dbg| dbg.dbg),
            },
        }
    }

    #[classmethod]
    fn new_unit_literal(_cls: &Bound<'_, PyType>, dbg: Option<DebugLoc>) -> Self {
        Self {
            expr: meta::qpu::MetaExpr::UnitLiteral {
                dbg: dbg.map(|dbg| dbg.dbg),
            },
        }
    }

    #[classmethod]
    fn new_embed_classical(
        _cls: &Bound<'_, PyType>,
        func_name: String,
        embed_kind: EmbedKind,
        dbg: Option<DebugLoc>,
    ) -> Self {
        let dbg = dbg.map(|dbg| dbg.dbg);
        Self {
            expr: meta::qpu::MetaExpr::EmbedClassical {
                func: Box::new(meta::qpu::MetaExpr::Variable {
                    name: func_name,
                    dbg: dbg.clone(),
                }),
                embed_kind: embed_kind.to_ast_kind(),
                dbg,
            },
        }
    }

    #[classmethod]
    fn new_measure(_cls: &Bound<'_, PyType>, basis: Basis, dbg: Option<DebugLoc>) -> Self {
        Self {
            expr: meta::qpu::MetaExpr::Measure {
                basis: basis.basis,
                dbg: dbg.map(|dbg| dbg.dbg),
            },
        }
    }

    #[classmethod]
    fn new_pipe(
        _cls: &Bound<'_, PyType>,
        lhs: QpuExpr,
        rhs: QpuExpr,
        dbg: Option<DebugLoc>,
    ) -> Self {
        Self {
            expr: meta::qpu::MetaExpr::Pipe {
                lhs: Box::new(lhs.expr),
                rhs: Box::new(rhs.expr),
                dbg: dbg.map(|dbg| dbg.dbg),
            },
        }
    }

    #[classmethod]
    fn new_discard(_cls: &Bound<'_, PyType>, dbg: Option<DebugLoc>) -> Self {
        Self {
            expr: meta::qpu::MetaExpr::Discard {
                dbg: dbg.map(|dbg| dbg.dbg),
            },
        }
    }

    #[classmethod]
    fn new_bi_tensor(
        _cls: &Bound<'_, PyType>,
        left: QpuExpr,
        right: QpuExpr,
        dbg: Option<DebugLoc>,
    ) -> Self {
        Self {
            expr: meta::qpu::MetaExpr::BiTensor {
                left: Box::new(left.expr),
                right: Box::new(right.expr),
                dbg: dbg.map(|dbg| dbg.dbg),
            },
        }
    }

    #[classmethod]
    fn new_basis_translation(
        _cls: &Bound<'_, PyType>,
        bin: Basis,
        bout: Basis,
        dbg: Option<DebugLoc>,
    ) -> Self {
        Self {
            expr: meta::qpu::MetaExpr::BasisTranslation {
                bin: bin.basis,
                bout: bout.basis,
                dbg: dbg.map(|dbg| dbg.dbg),
            },
        }
    }

    #[classmethod]
    fn new_predicated(
        _cls: &Bound<'_, PyType>,
        then_func: QpuExpr,
        else_func: QpuExpr,
        pred: Basis,
        dbg: Option<DebugLoc>,
    ) -> Self {
        Self {
            expr: meta::qpu::MetaExpr::Predicated {
                then_func: Box::new(then_func.expr),
                else_func: Box::new(else_func.expr),
                pred: pred.basis,
                dbg: dbg.map(|dbg| dbg.dbg),
            },
        }
    }

    #[classmethod]
    fn new_conditional(
        _cls: &Bound<'_, PyType>,
        then_expr: QpuExpr,
        else_expr: QpuExpr,
        cond: QpuExpr,
        dbg: Option<DebugLoc>,
    ) -> Self {
        Self {
            expr: meta::qpu::MetaExpr::Conditional {
                then_expr: Box::new(then_expr.expr),
                else_expr: Box::new(else_expr.expr),
                cond: Box::new(cond.expr),
                dbg: dbg.map(|dbg| dbg.dbg),
            },
        }
    }

    #[classmethod]
    fn new_qlit(_cls: &Bound<'_, PyType>, qlit: Vector) -> Self {
        Self {
            expr: meta::qpu::MetaExpr::QLit { vec: qlit.vec },
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
            expr: meta::qpu::MetaExpr::BitLiteral {
                val: val.0,
                n_bits: meta::DimExpr::DimConst {
                    val: n_bits.into(),
                    dbg: dbg.clone(),
                },
                dbg,
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
pub struct QpuStmt {
    pub stmt: meta::qpu::MetaStmt,
}

impl fmt::Display for QpuStmt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.stmt)
    }
}

#[pymethods]
impl QpuStmt {
    #[classmethod]
    fn new_expr(_cls: &Bound<'_, PyType>, expr: QpuExpr) -> Self {
        Self {
            stmt: meta::qpu::MetaStmt::Expr { expr: expr.expr },
        }
    }

    #[classmethod]
    fn new_assign(
        _cls: &Bound<'_, PyType>,
        lhs: String,
        rhs: QpuExpr,
        dbg: Option<DebugLoc>,
    ) -> Self {
        Self {
            stmt: meta::qpu::MetaStmt::Assign {
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
        rhs: QpuExpr,
        dbg: Option<DebugLoc>,
    ) -> Self {
        Self {
            stmt: meta::qpu::MetaStmt::UnpackAssign {
                lhs,
                rhs: rhs.expr,
                dbg: dbg.map(|dbg| dbg.dbg),
            },
        }
    }

    #[classmethod]
    fn new_return(_cls: &Bound<'_, PyType>, val: QpuExpr, dbg: Option<DebugLoc>) -> Self {
        Self {
            stmt: meta::qpu::MetaStmt::Return {
                val: val.expr,
                dbg: dbg.map(|dbg| dbg.dbg),
            },
        }
    }

    pub fn extract<'py>(&self, py: Python<'py>) -> PyResult<PlainQpuStmt> {
        self.stmt
            .extract()
            .map(|ast_stmt| PlainQpuStmt { stmt: ast_stmt })
            .map_err(|err| get_err(py, ProgErrKind::Expand, err.kind.to_string(), err.dbg))
    }

    /// Return the Debug form of this Stmt from __repr__(). By contrast,
    /// __str__() returns the Display form.
    pub fn __repr__(&self) -> String {
        format!("{:?}", self.stmt)
    }
}

#[pyclass]
#[derive(Clone)]
pub struct QpuFunctionDef {
    pub function_def: meta::MetaFunctionDef<meta::qpu::MetaStmt>,
}

#[pymethods]
impl QpuFunctionDef {
    #[new]
    fn new(
        name: String,
        args: Vec<(Type, String)>,
        ret_type: Type,
        body: Vec<QpuStmt>,
        is_rev: bool,
        dbg: Option<DebugLoc>,
    ) -> Self {
        Self {
            function_def: meta::MetaFunctionDef {
                name,
                args: args
                    .iter()
                    .map(|(arg_ty, arg_name)| (arg_ty.ty.clone(), arg_name.to_string()))
                    .collect(),
                ret_type: ret_type.ty.clone(),
                body: body.iter().map(|stmt| stmt.stmt.clone()).collect(),
                is_rev,
                dim_vars: vec![],
                dbg: dbg.map(|dbg| dbg.dbg),
            },
        }
    }

    fn get_name(&self) -> String {
        self.function_def.name.to_string()
    }
}
