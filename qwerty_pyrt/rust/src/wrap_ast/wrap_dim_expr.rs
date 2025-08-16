use crate::wrap_ast::{py_glue::IBigWrap, wrap_type::DebugLoc};
use pyo3::{prelude::*, types::PyType};
use qwerty_ast::meta;

#[pyclass]
#[derive(Clone)]
pub struct DimVar {
    pub var: meta::DimVar,
}

#[pymethods]
impl DimVar {
    #[classmethod]
    fn new_macro_param(_cls: &Bound<'_, PyType>, var_name: String) -> Self {
        Self {
            var: meta::DimVar::MacroParam { var_name },
        }
    }

    #[classmethod]
    fn new_func_var(_cls: &Bound<'_, PyType>, var_name: String, func_name: String) -> Self {
        Self {
            var: meta::DimVar::FuncVar {
                var_name,
                func_name,
            },
        }
    }
}

#[pyclass]
#[derive(Clone)]
pub struct DimExpr {
    pub dim_expr: meta::DimExpr,
}

#[pymethods]
impl DimExpr {
    #[classmethod]
    fn new_var(_cls: &Bound<'_, PyType>, var: DimVar, dbg: Option<DebugLoc>) -> Self {
        Self {
            dim_expr: meta::DimExpr::DimVar {
                var: var.var,
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
    fn new_pow(
        _cls: &Bound<'_, PyType>,
        base: DimExpr,
        pow: DimExpr,
        dbg: Option<DebugLoc>,
    ) -> Self {
        Self {
            dim_expr: meta::DimExpr::DimPow {
                base: Box::new(base.dim_expr),
                pow: Box::new(pow.dim_expr),
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
