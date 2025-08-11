use crate::{
    mlir::run_ast,
    wrap_ast::{
        py_glue::{ProgErrKind, UBigWrap, get_bit_reg, get_err},
        wrap_classical::ClassicalFunctionDef,
        wrap_qpu::QpuFunctionDef,
        wrap_type::DebugLoc,
    },
};
use pyo3::{conversion::IntoPyObject, prelude::*};
use qwerty_ast::meta;

#[pyclass]
pub struct Program {
    program: meta::MetaProgram,
}

impl Program {
    fn add_function_def(&mut self, func: meta::MetaFunc) {
        self.program.funcs.push(func);
    }
}

#[pymethods]
impl Program {
    #[new]
    fn new(dbg: Option<DebugLoc>) -> Self {
        Self {
            program: meta::MetaProgram {
                funcs: vec![],
                dbg: dbg.map(|dbg| dbg.dbg),
            },
        }
    }

    fn add_qpu_function_def(&mut self, func: QpuFunctionDef) {
        self.add_function_def(meta::MetaFunc::Qpu(func.function_def));
    }

    fn add_classical_function_def(&mut self, func: ClassicalFunctionDef) {
        self.add_function_def(meta::MetaFunc::Classical(func.function_def));
    }

    fn call<'py>(
        &mut self,
        py: Python<'py>,
        func_name: String,
        num_shots: usize,
        debug: bool,
    ) -> PyResult<Vec<(Bound<'py, PyAny>, usize)>> {
        let plain_ast = self
            .program
            .lower()
            .map_err(|err| get_err(py, ProgErrKind::Expand, err.kind.to_string(), err.dbg))?;

        plain_ast
            .typecheck()
            .map_err(|err| get_err(py, ProgErrKind::Type, err.kind.to_string(), err.dbg))?;

        run_ast(&plain_ast, &func_name, num_shots, debug)
            .into_iter()
            .map(|shot_result| {
                let as_int = UBigWrap(shot_result.bits).into_pyobject(py)?;
                get_bit_reg(py, as_int, shot_result.num_bits)
                    .map(|bit_reg| (bit_reg, shot_result.count))
            })
            .collect()
    }
}
