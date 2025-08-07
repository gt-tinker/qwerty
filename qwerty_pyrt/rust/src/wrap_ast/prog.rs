use crate::{
    mlir::run_ast,
    wrap_ast::{
        classical::ClassicalFunctionDef,
        py_glue::{get_bit_reg, get_err, ProgErrKind, UBigWrap},
        qpu::QpuFunctionDef,
        ty::DebugLoc,
    },
};
use pyo3::{conversion::IntoPyObject, prelude::*};
use qwerty_ast::ast;

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
