use pyo3::{prelude::*, types::PyType};
use qwerty_ast::{ast, dbg};

pyo3::import_exception!(qwerty.err, QwertyProgrammerError);

#[pyclass]
#[derive(Clone)]
struct DebugLoc {
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
}

#[pyclass]
#[derive(Clone)]
enum RegKind {
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
struct Type {
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
struct QLit {
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
}

#[pyclass]
#[derive(Clone)]
struct Vector {
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
}

#[pyclass]
#[derive(Clone)]
struct Basis {
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
}

#[pyclass]
#[derive(Clone)]
struct Expr {
    expr: ast::Expr,
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
}

#[pyclass]
#[derive(Clone)]
struct Stmt {
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
struct FunctionDef {
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
                dbg: dbg.map(|dbg| dbg.dbg),
            },
        }
    }

    fn get_name(&self) -> String {
        self.function_def.name.to_string()
    }
}

#[pyclass]
struct Program {
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

    fn type_check(&mut self) -> PyResult<()> {
        if !self.type_checked {
            // TODO: need to run type checking
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
        self.type_check()?;

        let zero_bit = PyModule::import(py, "qwerty.runtime")?
            .getattr("bit")?
            .call1((0, 1))?;

        let counts = vec![(zero_bit, num_shots)];

        println!("imagine we are calling {func_name}()...");

        Ok(counts)
    }
}

/// A Python module implemented in Rust.
#[pymodule]
#[pyo3(name = "_qwerty_pyrt")]
fn qwerty_pyrt(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<DebugLoc>()?;
    module.add_class::<RegKind>()?;
    module.add_class::<Type>()?;
    module.add_class::<QLit>()?;
    module.add_class::<Vector>()?;
    module.add_class::<Basis>()?;
    module.add_class::<Expr>()?;
    module.add_class::<Stmt>()?;
    module.add_class::<FunctionDef>()?;
    module.add_class::<Program>()?;

    Ok(())
}
