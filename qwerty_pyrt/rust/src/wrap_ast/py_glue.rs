use crate::wrap_ast::wrap_type::DebugLoc;
use dashu::{
    base::Sign,
    integer::{IBig, UBig},
};
use pyo3::{
    conversion::{FromPyObject, IntoPyObject},
    exceptions::PyValueError,
    prelude::*,
    sync::GILOnceCell,
    types::{PyBytes, PyInt, PyType},
};
use qwerty_ast::dbg;

static BIT_TYPE: GILOnceCell<Py<PyType>> = GILOnceCell::new();
static QWERTY_TYPE_ERROR_TYPE: GILOnceCell<Py<PyType>> = GILOnceCell::new();
static QWERTY_EXPAND_ERROR_TYPE: GILOnceCell<Py<PyType>> = GILOnceCell::new();

pub fn get_bit_reg<'py>(
    py: Python<'py>,
    as_int: Bound<'py, PyInt>,
    n_bits: usize,
) -> PyResult<Bound<'py, PyAny>> {
    let bit_type = BIT_TYPE.import(py, "qwerty.runtime", "bit")?;
    bit_type.call1((as_int, n_bits))
}

/// Converts a Python `int` into a [`UBig`]. Integer must be nonnegative or
/// this will throw a Python exception.
fn py_uint_to_ubig<'py>(int: &Bound<'py, PyInt>) -> PyResult<UBig> {
    let num_bits = int.call_method0("bit_length")?.extract::<usize>()?;
    let num_bytes = (num_bits + 7) / 8;
    let bytes = int
        .call_method1("to_bytes", (num_bytes, "big"))?
        .extract::<Vec<u8>>()?;
    Ok(UBig::from_be_bytes(&bytes))
}

/// Converts a [`UBig`] into a Python `int`.
fn ubig_to_py_int<'py>(py: Python<'py>, ubig: UBig) -> PyResult<Bound<'py, PyInt>> {
    let big_endian_bytes = ubig.to_be_bytes();
    let py_bytes = PyBytes::new(py, &*big_endian_bytes);
    py.get_type::<PyInt>()
        .call_method1("from_bytes", (py_bytes, "big"))?
        .downcast_into()
        .map_err(|err| err.into())
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
        ubig_to_py_int(py, self.0)
    }
}

impl<'py> FromPyObject<'py> for UBigWrap {
    fn extract_bound(obj: &Bound<'py, PyAny>) -> PyResult<UBigWrap> {
        let int = obj.downcast::<PyInt>()?;
        if int.lt(0)? {
            Err(PyValueError::new_err("value cannot be negative"))
        } else {
            py_uint_to_ubig(int).map(|ubig| UBigWrap(ubig))
        }
    }
}

/// Similar to [`UBigWrap`] except for [`IBig`] instead.
#[derive(Clone, Debug)]
pub struct IBigWrap(pub IBig);

impl<'py> IntoPyObject<'py> for IBigWrap {
    type Target = PyInt;
    type Output = Bound<'py, Self::Target>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> PyResult<Bound<'py, PyInt>> {
        let (sgn, mag) = self.0.into_parts();
        let py_int = ubig_to_py_int(py, mag)?;
        match sgn {
            Sign::Positive => Ok(py_int),
            Sign::Negative => py_int
                .neg()
                .and_then(|neg_any| neg_any.downcast_into().map_err(|err| err.into())),
        }
    }
}

impl<'py> FromPyObject<'py> for IBigWrap {
    fn extract_bound(obj: &Bound<'py, PyAny>) -> PyResult<IBigWrap> {
        let int = obj.downcast::<PyInt>()?;
        let sgn = if int.lt(0)? {
            Sign::Negative
        } else {
            Sign::Positive
        };
        let mag = py_uint_to_ubig(int)?;
        Ok(IBigWrap(IBig::from_parts(sgn, mag)))
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum ProgErrKind {
    Type,
    Expand,
}

fn get_err_ty<'py>(py: Python<'py>, kind: ProgErrKind) -> PyResult<Bound<'py, PyType>> {
    match kind {
        ProgErrKind::Type => QWERTY_TYPE_ERROR_TYPE.import(py, "qwerty.err", "QwertyTypeError"),
        ProgErrKind::Expand => {
            QWERTY_EXPAND_ERROR_TYPE.import(py, "qwerty.err", "QwertyExpandError")
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
