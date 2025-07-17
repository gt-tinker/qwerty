use crate::mlir::run_ast;
use dashu::integer::UBig;
use pyo3::{
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

fn ubig_to_pyobject<'py>(py: Python<'py>, ubig: &UBig) -> PyResult<Bound<'py, PyInt>> {
    let big_endian_bytes = ubig.to_be_bytes();
    let py_bytes = PyBytes::new(py, &*big_endian_bytes);
    Ok(py
        .get_type::<PyInt>()
        .call_method1("from_bytes", (py_bytes, "big"))?
        .downcast_into()?)
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

#[pyclass]
#[derive(Clone)]
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
    fn new_reg(_cls: &Bound<'_, PyType>, elem_ty: RegKind, dim: u64) -> Self {
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
    qlit: ast::QLit,
}

#[pymethods]
impl QLit {
    #[classmethod]
    fn new_zero_qubit(_cls: &Bound<'_, PyType>, dbg: Option<DebugLoc>) -> Self {
        Self {
            qlit: ast::QLit::ZeroQubit {
                dbg: dbg.map(|dbg| dbg.dbg),
            },
        }
    }

    #[classmethod]
    fn new_one_qubit(_cls: &Bound<'_, PyType>, dbg: Option<DebugLoc>) -> Self {
        Self {
            qlit: ast::QLit::OneQubit {
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
            qlit: ast::QLit::UniformSuperpos {
                q1: Box::new(q1.qlit),
                q2: Box::new(q2.qlit),
                dbg: dbg.map(|dbg| dbg.dbg),
            },
        }
    }

    #[classmethod]
    fn new_qubit_tensor(_cls: &Bound<'_, PyType>, qs: Vec<QLit>, dbg: Option<DebugLoc>) -> Self {
        Self {
            qlit: ast::QLit::QubitTensor {
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
            qlit: ast::QLit::QubitTilt {
                q: Box::new(q.qlit.clone()),
                angle_deg,
                dbg: dbg.map(|dbg| dbg.dbg),
            },
        }
    }

    #[classmethod]
    fn new_qubit_unit(_cls: &Bound<'_, PyType>, dbg: Option<DebugLoc>) -> Self {
        Self {
            qlit: ast::QLit::QubitUnit {
                dbg: dbg.map(|dbg| dbg.dbg),
            },
        }
    }
}

#[pyclass]
#[derive(Clone)]
pub struct Vector {
    vec: ast::Vector,
}

#[pymethods]
impl Vector {
    #[classmethod]
    fn new_zero_vector(_cls: &Bound<'_, PyType>, dbg: Option<DebugLoc>) -> Self {
        Self {
            vec: ast::Vector::ZeroVector {
                dbg: dbg.map(|dbg| dbg.dbg),
            },
        }
    }

    #[classmethod]
    fn new_one_vector(_cls: &Bound<'_, PyType>, dbg: Option<DebugLoc>) -> Self {
        Self {
            vec: ast::Vector::OneVector {
                dbg: dbg.map(|dbg| dbg.dbg),
            },
        }
    }

    #[classmethod]
    fn new_pad_vector(_cls: &Bound<'_, PyType>, dbg: Option<DebugLoc>) -> Self {
        Self {
            vec: ast::Vector::PadVector {
                dbg: dbg.map(|dbg| dbg.dbg),
            },
        }
    }

    #[classmethod]
    fn new_target_vector(_cls: &Bound<'_, PyType>, dbg: Option<DebugLoc>) -> Self {
        Self {
            vec: ast::Vector::TargetVector {
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
            vec: ast::Vector::UniformVectorSuperpos {
                q1: Box::new(q1.vec.clone()),
                q2: Box::new(q2.vec.clone()),
                dbg: dbg.map(|dbg| dbg.dbg),
            },
        }
    }

    #[classmethod]
    fn new_vector_tensor(_cls: &Bound<'_, PyType>, qs: Vec<Vector>, dbg: Option<DebugLoc>) -> Self {
        Self {
            vec: ast::Vector::VectorTensor {
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
            vec: ast::Vector::VectorTilt {
                q: Box::new(q.vec.clone()),
                angle_deg,
                dbg: dbg.map(|dbg| dbg.dbg),
            },
        }
    }

    #[classmethod]
    fn new_vector_unit(_cls: &Bound<'_, PyType>, dbg: Option<DebugLoc>) -> Self {
        Self {
            vec: ast::Vector::VectorUnit {
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
pub struct Basis {
    basis: ast::Basis,
}

#[pymethods]
impl Basis {
    #[classmethod]
    fn new_empty_basis_literal(_cls: &Bound<'_, PyType>, dbg: Option<DebugLoc>) -> Self {
        Self {
            basis: ast::Basis::EmptyBasisLiteral {
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
            basis: ast::Basis::BasisLiteral {
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
            basis: ast::Basis::BasisTensor {
                bases: bases.iter().map(|b| b.basis.clone()).collect(),
                dbg: dbg.map(|dbg| dbg.dbg),
            },
        }
    }
}

#[pyclass(str)]
#[derive(Clone)]
pub struct Expr {
    pub(crate) expr: ast::Expr,
}

impl Expr {
    pub fn new(expr: ast::Expr) -> Self {
        Self { expr }
    }
}

impl fmt::Display for Expr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.expr)
    }
}

#[pymethods]
impl Expr {
    #[classmethod]
    fn new_qlit(_cls: &Bound<'_, PyType>, qlit: QLit, dbg: Option<DebugLoc>) -> Self {
        Self {
            expr: ast::Expr::QLit {
                qlit: qlit.qlit,
                dbg: dbg.map(|dbg| dbg.dbg),
            },
        }
    }

    #[classmethod]
    fn new_measure(_cls: &Bound<'_, PyType>, basis: Basis, dbg: Option<DebugLoc>) -> Self {
        Self {
            expr: ast::Expr::Measure {
                basis: basis.basis,
                dbg: dbg.map(|dbg| dbg.dbg),
            },
        }
    }

    #[classmethod]
    fn new_pipe(_cls: &Bound<'_, PyType>, lhs: Expr, rhs: Expr, dbg: Option<DebugLoc>) -> Self {
        Self {
            expr: ast::Expr::Pipe {
                lhs: Box::new(lhs.expr),
                rhs: Box::new(rhs.expr),
                dbg: dbg.map(|dbg| dbg.dbg),
            },
        }
    }

    #[classmethod]
    fn new_tensor(_cls: &Bound<'_, PyType>, vals: Vec<Expr>, dbg: Option<DebugLoc>) -> Self {
        Self {
            expr: ast::Expr::Tensor {
                vals: vals.iter().map(|v| v.expr.clone()).collect(),
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
            expr: ast::Expr::BasisTranslation {
                bin: bin.basis.clone(),
                bout: bout.basis.clone(),
                dbg: dbg.map(|dbg| dbg.dbg),
            },
        }
    }
}

#[pyclass]
#[derive(Clone)]
pub struct Stmt {
    stmt: ast::Stmt,
}

#[pymethods]
impl Stmt {
    #[classmethod]
    fn new_assign(_cls: &Bound<'_, PyType>, lhs: String, rhs: Expr, dbg: Option<DebugLoc>) -> Self {
        Self {
            stmt: ast::Stmt::Assign {
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
        rhs: Expr,
        dbg: Option<DebugLoc>,
    ) -> Self {
        Self {
            stmt: ast::Stmt::UnpackAssign {
                lhs,
                rhs: rhs.expr,
                dbg: dbg.map(|dbg| dbg.dbg),
            },
        }
    }

    #[classmethod]
    fn new_return(_cls: &Bound<'_, PyType>, val: Expr, dbg: Option<DebugLoc>) -> Self {
        Self {
            stmt: ast::Stmt::Return {
                val: val.expr,
                dbg: dbg.map(|dbg| dbg.dbg),
            },
        }
    }
}

#[pyclass]
#[derive(Clone)]
pub struct FunctionDef {
    function_def: ast::FunctionDef,
}

#[pymethods]
impl FunctionDef {
    #[new]
    fn new(
        name: String,
        args: Vec<(Type, String)>,
        ret_type: Type,
        body: Vec<Stmt>,
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

    fn add_function_def(&mut self, func: FunctionDef) {
        self.program.funcs.push(func.function_def);
        self.type_checked = false;
    }

    fn type_check(&mut self, py: Python<'_>) -> PyResult<()> {
        if !self.type_checked {
            if let Err(type_err) = typecheck::typecheck_program(&self.program) {
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
    ) -> PyResult<Vec<(Bound<'py, PyAny>, usize)>> {
        self.type_check(py)?;

        run_ast(&self.program, &func_name, num_shots)
            .iter()
            .map(|shot_result| {
                ubig_to_pyobject(py, &shot_result.bits)
                    .and_then(|as_int| get_bit_reg(py, as_int, shot_result.num_bits))
                    .map(|bit_reg| (bit_reg, shot_result.count))
            })
            .collect()
    }
}
