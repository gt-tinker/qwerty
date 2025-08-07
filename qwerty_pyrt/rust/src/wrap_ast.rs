use crate::mlir::run_ast;
use dashu::integer::UBig;
use pyo3::{
    conversion::{FromPyObject, IntoPyObject},
    prelude::*,
    sync::GILOnceCell,
    types::{PyBytes, PyInt, PyType},
};
use qwerty_ast::{ast, dbg, typecheck};
use std::fmt;

static BIT_TYPE: GILOnceCell<Py<PyType>> = GILOnceCell::new();
static QWERTY_PROGRAMMER_ERROR_TYPE: GILOnceCell<Py<PyType>> = GILOnceCell::new();

fn get_bit_reg<'py>(
    py: Python<'py>,
    as_int: Bound<'py, PyInt>,
    n_bits: usize,
) -> PyResult<Bound<'py, PyAny>> {
    let bit_type = BIT_TYPE.import(py, "qwerty.runtime", "bit")?;
    bit_type.call1((as_int, n_bits))
}

/// A "newtype" around UBig that allows implementing the IntoPyObject and
/// FromPyObject traits without violating the orphan rule (since UBig is from
/// the dashu crate)
#[derive(Clone, Debug)]
struct UBigWrap(UBig);

impl<'py> IntoPyObject<'py> for UBigWrap {
    type Target = PyInt;
    type Output = Bound<'py, Self::Target>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> PyResult<Bound<'py, PyInt>> {
        let big_endian_bytes = self.0.to_be_bytes();
        let py_bytes = PyBytes::new(py, &*big_endian_bytes);
        Ok(py
            .get_type::<PyInt>()
            .call_method1("from_bytes", (py_bytes, "big"))?
            .downcast_into()?)
    }
}

impl<'py> FromPyObject<'py> for UBigWrap {
    fn extract_bound(obj: &Bound<'py, PyAny>) -> PyResult<Self> {
        let as_int = obj.downcast::<PyInt>()?;
        let num_bits = as_int.call_method0("bit_length")?.extract::<usize>()?;
        let num_bytes = (num_bits + 7) / 8;
        let bytes = as_int
            .call_method1("to_bytes", (num_bytes, "big"))?
            .extract::<Vec<u8>>()?;
        Ok(Self(UBig::from_be_bytes(&bytes)))
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ProgErrKind {
    Type,
}

fn get_err_ty<'py>(py: Python<'py>, kind: ProgErrKind) -> PyResult<Bound<'py, PyType>> {
    match kind {
        ProgErrKind::Type => {
            QWERTY_PROGRAMMER_ERROR_TYPE.import(py, "qwerty.err", "QwertyTypeError")
        }
    }
    .cloned()
}

fn get_err<'py>(
    py: Python<'py>,
    kind: ProgErrKind,
    msg: String,
    dbg: Option<dbg::DebugLoc>,
) -> PyErr {
    let err_ty_res = get_err_ty(py, kind);
    match err_ty_res {
        Err(err) => err,
        Ok(err_ty) => {
            let dbg_wrapped = dbg.map(|ast_dbg| DebugLoc {
                dbg: ast_dbg.clone(),
            });
            PyErr::from_type(err_ty.clone(), (msg, dbg_wrapped))
        }
    }
}

#[pyclass(eq, hash, frozen)]
#[derive(Clone, PartialEq, Hash)]
pub struct DebugLoc {
    dbg: dbg::DebugLoc,
}

#[pymethods]
impl DebugLoc {
    #[new]
    fn new(file: String, line: usize, col: usize) -> Self {
        Self {
            dbg: dbg::DebugLoc { file, line, col },
        }
    }

    fn get_col(&self) -> usize {
        self.dbg.col
    }

    fn get_line(&self) -> usize {
        self.dbg.line
    }
}

#[pyclass(eq, hash, frozen)]
#[derive(Clone, PartialEq, Hash)]
pub enum RegKind {
    Bit,
    Qubit,
    Basis,
}

// Intentionally not annotated with #[pymethods]: this is only for use in Rust
impl RegKind {
    fn to_ast_kind(&self) -> ast::RegKind {
        match self {
            RegKind::Bit => ast::RegKind::Bit,
            RegKind::Qubit => ast::RegKind::Qubit,
            RegKind::Basis => ast::RegKind::Basis,
        }
    }
}

#[pyclass]
#[derive(Clone)]
pub struct Type {
    ty: ast::Type,
}

#[pymethods]
impl Type {
    #[classmethod]
    fn new_func(_cls: &Bound<'_, PyType>, in_ty: Type, out_ty: Type) -> Self {
        Self {
            ty: ast::Type::FuncType {
                in_ty: Box::new(in_ty.ty),
                out_ty: Box::new(out_ty.ty),
            },
        }
    }

    #[classmethod]
    fn new_rev_func(_cls: &Bound<'_, PyType>, in_out_ty: Type) -> Self {
        Self {
            ty: ast::Type::RevFuncType {
                in_out_ty: Box::new(in_out_ty.ty),
            },
        }
    }

    #[classmethod]
    fn new_reg(_cls: &Bound<'_, PyType>, elem_ty: RegKind, dim: usize) -> Self {
        Self {
            ty: ast::Type::RegType {
                elem_ty: elem_ty.to_ast_kind(),
                dim: dim,
            },
        }
    }

    #[classmethod]
    fn new_unit(_cls: &Bound<'_, PyType>) -> Self {
        Self {
            ty: ast::Type::UnitType,
        }
    }
}

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
pub struct QpuStmt {
    pub(crate) stmt: ast::Stmt<ast::qpu::Expr>,
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
pub struct QpuFunctionDef {
    function_def: ast::FunctionDef<ast::qpu::Expr>,
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

#[pyclass]
#[derive(Clone)]
pub struct ClassicalFunctionDef {
    function_def: ast::FunctionDef<ast::classical::Expr>,
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

#[pyclass]
pub struct Program {
    program: ast::Program,
    type_checked: bool,
}

impl Program {
    fn add_function_def(&mut self, func: ast::Func) {
        self.program.funcs.push(func);
        self.type_checked = false;
    }
}

#[pymethods]
impl Program {
    #[new]
    fn new(dbg: Option<DebugLoc>) -> Self {
        Self {
            program: ast::Program {
                funcs: vec![],
                dbg: dbg.map(|dbg| dbg.dbg),
            },
            type_checked: false,
        }
    }

    fn add_qpu_function_def(&mut self, func: QpuFunctionDef) {
        self.add_function_def(ast::Func::Qpu(func.function_def));
    }

    fn add_classical_function_def(&mut self, func: ClassicalFunctionDef) {
        self.add_function_def(ast::Func::Classical(func.function_def));
    }

    fn type_check(&mut self, py: Python<'_>) -> PyResult<()> {
        if !self.type_checked {
            if let Err(type_err) = self.program.typecheck() {
                return Err(get_err(
                    py,
                    ProgErrKind::Type,
                    type_err.kind.to_string(),
                    type_err.dbg,
                ));
            }
            self.type_checked = true;
        }
        Ok(())
    }

    fn call<'py>(
        &mut self,
        py: Python<'py>,
        func_name: String,
        num_shots: usize,
        debug: bool,
    ) -> PyResult<Vec<(Bound<'py, PyAny>, usize)>> {
        self.type_check(py)?;

        run_ast(&self.program, &func_name, num_shots, debug)
            .into_iter()
            .map(|shot_result| {
                let as_int = UBigWrap(shot_result.bits).into_pyobject(py)?;
                get_bit_reg(py, as_int, shot_result.num_bits)
                    .map(|bit_reg| (bit_reg, shot_result.count))
            })
            .collect()
    }
}

#[pyclass]
pub struct TypeEnv {
    env: typecheck::TypeEnv,
}

#[pymethods]
impl TypeEnv {
    #[new]
    pub fn new() -> Self {
        TypeEnv {
            env: typecheck::TypeEnv::new(),
        }
    }
}
