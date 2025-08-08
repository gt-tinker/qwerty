use crate::wrap_ast::wrap_type::DebugLoc;
use dashu::integer::UBig;
use pyo3::{
    conversion::{FromPyObject, IntoPyObject},
    prelude::*,
    sync::GILOnceCell,
    types::{PyBytes, PyInt, PyType},
};
use qwerty_ast::dbg;

static BIT_TYPE: GILOnceCell<Py<PyType>> = GILOnceCell::new();
static QWERTY_PROGRAMMER_ERROR_TYPE: GILOnceCell<Py<PyType>> = GILOnceCell::new();

pub fn get_bit_reg<'py>(
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
pub struct UBigWrap(pub UBig);

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
pub enum ProgErrKind {
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

pub fn get_err<'py>(
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
