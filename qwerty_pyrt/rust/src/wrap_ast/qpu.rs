use crate::wrap_ast::{
    py_glue::{get_err, ProgErrKind, UBigWrap},
    ty::{DebugLoc, Type, TypeEnv},
};
use pyo3::{prelude::*, types::PyType};
use qwerty_ast::ast;
use std::fmt;

#[pyclass]
#[derive(Clone)]
pub struct QLit {
    qlit: ast::qpu::QLit,
}

#[pymethods]
impl QLit {
    #[classmethod]
    fn new_zero_qubit(_cls: &Bound<'_, PyType>, dbg: Option<DebugLoc>) -> Self {
        Self {
            qlit: ast::qpu::QLit::ZeroQubit {
                dbg: dbg.map(|dbg| dbg.dbg),
            },
        }
    }

    #[classmethod]
    fn new_one_qubit(_cls: &Bound<'_, PyType>, dbg: Option<DebugLoc>) -> Self {
        Self {
            qlit: ast::qpu::QLit::OneQubit {
                dbg: dbg.map(|dbg| dbg.dbg),
            },
        }
    }

    #[classmethod]
    fn new_uniform_superpos(
        _cls: &Bound<'_, PyType>,
        q1: QLit,
        q2: QLit,
        dbg: Option<DebugLoc>,
    ) -> Self {
        Self {
            qlit: ast::qpu::QLit::UniformSuperpos {
                q1: Box::new(q1.qlit),
                q2: Box::new(q2.qlit),
                dbg: dbg.map(|dbg| dbg.dbg),
            },
        }
    }

    #[classmethod]
    fn new_qubit_tensor(_cls: &Bound<'_, PyType>, qs: Vec<QLit>, dbg: Option<DebugLoc>) -> Self {
        Self {
            qlit: ast::qpu::QLit::QubitTensor {
                qs: qs.iter().map(|ql| ql.qlit.clone()).collect(),
                dbg: dbg.map(|dbg| dbg.dbg),
            },
        }
    }

    #[classmethod]
    fn new_qubit_tilt(
        _cls: &Bound<'_, PyType>,
        q: QLit,
        angle_deg: f64,
        dbg: Option<DebugLoc>,
    ) -> Self {
        Self {
            qlit: ast::qpu::QLit::QubitTilt {
                q: Box::new(q.qlit.clone()),
                angle_deg,
                dbg: dbg.map(|dbg| dbg.dbg),
            },
        }
    }

    #[classmethod]
    fn new_qubit_unit(_cls: &Bound<'_, PyType>, dbg: Option<DebugLoc>) -> Self {
        Self {
            qlit: ast::qpu::QLit::QubitUnit {
                dbg: dbg.map(|dbg| dbg.dbg),
            },
        }
    }
}

#[pyclass]
#[derive(Clone)]
pub struct Vector {
    vec: ast::qpu::Vector,
}

#[pymethods]
impl Vector {
    #[classmethod]
    fn new_zero_vector(_cls: &Bound<'_, PyType>, dbg: Option<DebugLoc>) -> Self {
        Self {
            vec: ast::qpu::Vector::ZeroVector {
                dbg: dbg.map(|dbg| dbg.dbg),
            },
        }
    }

    #[classmethod]
    fn new_one_vector(_cls: &Bound<'_, PyType>, dbg: Option<DebugLoc>) -> Self {
        Self {
            vec: ast::qpu::Vector::OneVector {
                dbg: dbg.map(|dbg| dbg.dbg),
            },
        }
    }

    #[classmethod]
    fn new_pad_vector(_cls: &Bound<'_, PyType>, dbg: Option<DebugLoc>) -> Self {
        Self {
            vec: ast::qpu::Vector::PadVector {
                dbg: dbg.map(|dbg| dbg.dbg),
            },
        }
    }

    #[classmethod]
    fn new_target_vector(_cls: &Bound<'_, PyType>, dbg: Option<DebugLoc>) -> Self {
        Self {
            vec: ast::qpu::Vector::TargetVector {
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
            vec: ast::qpu::Vector::UniformVectorSuperpos {
                q1: Box::new(q1.vec.clone()),
                q2: Box::new(q2.vec.clone()),
                dbg: dbg.map(|dbg| dbg.dbg),
            },
        }
    }

    #[classmethod]
    fn new_vector_tensor(_cls: &Bound<'_, PyType>, qs: Vec<Vector>, dbg: Option<DebugLoc>) -> Self {
        Self {
            vec: ast::qpu::Vector::VectorTensor {
                qs: qs.iter().map(|vec| vec.vec.clone()).collect(),
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
        Self {
            vec: ast::qpu::Vector::VectorTilt {
                q: Box::new(q.vec.clone()),
                angle_deg,
                dbg: dbg.map(|dbg| dbg.dbg),
            },
        }
    }

    #[classmethod]
    fn new_vector_unit(_cls: &Bound<'_, PyType>, dbg: Option<DebugLoc>) -> Self {
        Self {
            vec: ast::qpu::Vector::VectorUnit {
                dbg: dbg.map(|dbg| dbg.dbg),
            },
        }
    }

    fn convert_to_qubit_literal(&self) -> Option<QLit> {
        self.vec
            .convert_to_qubit_literal()
            .map(|qlit| QLit { qlit })
    }
}

#[pyclass]
#[derive(Clone)]
pub struct BasisGenerator {
    gen: ast::qpu::BasisGenerator,
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
            gen: ast::qpu::BasisGenerator::Revolve {
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
    basis: ast::qpu::Basis,
}

#[pymethods]
impl Basis {
    #[classmethod]
    fn new_empty_basis_literal(_cls: &Bound<'_, PyType>, dbg: Option<DebugLoc>) -> Self {
        Self {
            basis: ast::qpu::Basis::EmptyBasisLiteral {
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
            basis: ast::qpu::Basis::BasisLiteral {
                vecs: vecs.iter().map(|vec| vec.vec.clone()).collect(),
                dbg: dbg.map(|dbg| dbg.dbg),
            },
        }
    }

    #[classmethod]
    fn new_basis_tensor(
        _cls: &Bound<'_, PyType>,
        bases: Vec<Basis>,
        dbg: Option<DebugLoc>,
    ) -> Self {
        Self {
            basis: ast::qpu::Basis::BasisTensor {
                bases: bases.iter().map(|b| b.basis.clone()).collect(),
                dbg: dbg.map(|dbg| dbg.dbg),
            },
        }
    }

    #[classmethod]
    fn new_apply_basis_generator(
        _cls: &Bound<'_, PyType>,
        basis: Basis,
        gen: BasisGenerator,
        dbg: Option<DebugLoc>,
    ) -> Self {
        Self {
            basis: ast::qpu::Basis::ApplyBasisGenerator {
                basis: Box::new(basis.basis),
                gen: gen.gen,
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
    pub(crate) expr: ast::qpu::Expr,
}

impl QpuExpr {
    pub fn new(expr: ast::qpu::Expr) -> Self {
        Self { expr }
    }
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
            expr: ast::qpu::Expr::Variable(ast::Variable {
                name,
                dbg: dbg.map(|dbg| dbg.dbg),
            }),
        }
    }

    #[classmethod]
    fn new_unit_literal(_cls: &Bound<'_, PyType>, dbg: Option<DebugLoc>) -> Self {
        Self {
            expr: ast::qpu::Expr::UnitLiteral(ast::qpu::UnitLiteral {
                dbg: dbg.map(|dbg| dbg.dbg),
            }),
        }
    }

    #[classmethod]
    fn new_embed_classical(
        _cls: &Bound<'_, PyType>,
        func_name: String,
        embed_kind: EmbedKind,
        dbg: Option<DebugLoc>,
    ) -> Self {
        Self {
            expr: ast::qpu::Expr::EmbedClassical(ast::qpu::EmbedClassical {
                func_name,
                embed_kind: embed_kind.to_ast_kind(),
                dbg: dbg.map(|dbg| dbg.dbg),
            }),
        }
    }

    #[classmethod]
    fn new_measure(_cls: &Bound<'_, PyType>, basis: Basis, dbg: Option<DebugLoc>) -> Self {
        Self {
            expr: ast::qpu::Expr::Measure(ast::qpu::Measure {
                basis: basis.basis,
                dbg: dbg.map(|dbg| dbg.dbg),
            }),
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
            expr: ast::qpu::Expr::Pipe(ast::qpu::Pipe {
                lhs: Box::new(lhs.expr),
                rhs: Box::new(rhs.expr),
                dbg: dbg.map(|dbg| dbg.dbg),
            }),
        }
    }

    #[classmethod]
    fn new_discard(_cls: &Bound<'_, PyType>, dbg: Option<DebugLoc>) -> Self {
        Self {
            expr: ast::qpu::Expr::Discard(ast::qpu::Discard {
                dbg: dbg.map(|dbg| dbg.dbg),
            }),
        }
    }

    #[classmethod]
    fn new_tensor(_cls: &Bound<'_, PyType>, vals: Vec<QpuExpr>, dbg: Option<DebugLoc>) -> Self {
        Self {
            expr: ast::qpu::Expr::Tensor(ast::qpu::Tensor {
                vals: vals.iter().map(|v| v.expr.clone()).collect(),
                dbg: dbg.map(|dbg| dbg.dbg),
            }),
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
            expr: ast::qpu::Expr::BasisTranslation(ast::qpu::BasisTranslation {
                bin: bin.basis.clone(),
                bout: bout.basis.clone(),
                dbg: dbg.map(|dbg| dbg.dbg),
            }),
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
            expr: ast::qpu::Expr::Predicated(ast::qpu::Predicated {
                then_func: Box::new(then_func.expr),
                else_func: Box::new(else_func.expr),
                pred: pred.basis,
                dbg: dbg.map(|dbg| dbg.dbg),
            }),
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
            expr: ast::qpu::Expr::Conditional(ast::qpu::Conditional {
                then_expr: Box::new(then_expr.expr),
                else_expr: Box::new(else_expr.expr),
                cond: Box::new(cond.expr),
                dbg: dbg.map(|dbg| dbg.dbg),
            }),
        }
    }

    #[classmethod]
    fn new_qlit(_cls: &Bound<'_, PyType>, qlit: QLit) -> Self {
        Self {
            expr: ast::qpu::Expr::QLit(qlit.qlit),
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
            expr: ast::qpu::Expr::BitLiteral(ast::BitLiteral {
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
pub struct QpuStmt {
    pub stmt: ast::Stmt<ast::qpu::Expr>,
}

impl fmt::Display for QpuStmt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.stmt)
    }
}

#[pymethods]
impl QpuStmt {
    #[classmethod]
    fn new_expr(_cls: &Bound<'_, PyType>, expr: QpuExpr, dbg: Option<DebugLoc>) -> Self {
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
        rhs: QpuExpr,
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
        rhs: QpuExpr,
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
    fn new_return(_cls: &Bound<'_, PyType>, val: QpuExpr, dbg: Option<DebugLoc>) -> Self {
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

    pub fn type_check(
        &self,
        py: Python<'_>,
        env: &mut TypeEnv,
        expected_ret_type: Option<Type>,
    ) -> PyResult<()> {
        self.stmt
            .typecheck(&mut env.env, expected_ret_type.map(|ty| ty.ty))
            // Discard compute kind for now, since Python does not need it
            .map(|_compute_kind| ())
            .map_err(|err| get_err(py, ProgErrKind::Type, err.kind.to_string(), err.dbg))
    }
}

#[pyclass]
#[derive(Clone)]
pub struct QpuFunctionDef {
    pub function_def: ast::FunctionDef<ast::qpu::Expr>,
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
