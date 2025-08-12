use crate::wrap_ast::{py_glue::IBigWrap, wrap_type::DebugLoc};
use pyo3::{prelude::*, types::PyType};
use qwerty_ast::meta;

#[pyclass]
#[derive(Clone)]
pub struct DimExpr {
    pub dim_expr: meta::DimExpr,
}

#[pymethods]
impl DimExpr {
    #[classmethod]
    fn new_var(_cls: &Bound<'_, PyType>, name: String, dbg: Option<DebugLoc>) -> Self {
        Self {
            dim_expr: meta::DimExpr::DimVar {
                name,
                dbg: dbg.map(|dbg| dbg.dbg),
            },
        }
    }

    #[classmethod]
    fn new_const(_cls: &Bound<'_, PyType>, val: IBigWrap, dbg: Option<DebugLoc>) -> Self {
        Self {
            dim_expr: meta::DimExpr::DimConst {
                val: val.0,
                dbg: dbg.map(|dbg| dbg.dbg),
            },
        }
    }

    #[classmethod]
    fn new_sum(
        _cls: &Bound<'_, PyType>,
        left: DimExpr,
        right: DimExpr,
        dbg: Option<DebugLoc>,
    ) -> Self {
        Self {
            dim_expr: meta::DimExpr::DimSum {
                left: Box::new(left.dim_expr),
                right: Box::new(right.dim_expr),
                dbg: dbg.map(|dbg| dbg.dbg),
            },
        }
    }

    #[classmethod]
    fn new_prod(
        _cls: &Bound<'_, PyType>,
        left: DimExpr,
        right: DimExpr,
        dbg: Option<DebugLoc>,
    ) -> Self {
        Self {
            dim_expr: meta::DimExpr::DimProd {
                left: Box::new(left.dim_expr),
                right: Box::new(right.dim_expr),
                dbg: dbg.map(|dbg| dbg.dbg),
            },
        }
    }

    #[classmethod]
    fn new_neg(_cls: &Bound<'_, PyType>, val: DimExpr, dbg: Option<DebugLoc>) -> Self {
        Self {
            dim_expr: meta::DimExpr::DimNeg {
                val: Box::new(val.dim_expr),
                dbg: dbg.map(|dbg| dbg.dbg),
            },
        }
    }
}
