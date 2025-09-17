use crate::wrap_ast::{
    py_glue::{IBigWrap, ProgErrKind, UBigWrap, get_err},
    wrap_dim_expr::DimExpr,
    wrap_type::{DebugLoc, MacroEnv, Type, TypeEnv},
};
use pyo3::{prelude::*, types::PyType};
use qwerty_ast::{ast, meta};
use std::fmt;

/// A "plain" Qwerty expression AST node, that is, an expression without any
/// metaQwerty features (e.g., `__SYM_STD0__()` instead of `'0'`).
#[pyclass(str, eq)]
#[derive(Clone, PartialEq)]
pub struct PlainQpuExpr {
    pub expr: ast::qpu::Expr,
}

impl fmt::Display for PlainQpuExpr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.expr)
    }
}

/// A "plain" Qwerty statement AST node, that is, a statement without any
/// metaQwerty features (e.g., no basis alias definitions such as
/// `std = {'0','1'}`).
#[pyclass(str, eq)]
#[derive(Clone, PartialEq)]
pub struct PlainQpuStmt {
    pub stmt: ast::Stmt<ast::qpu::Expr>,
}

#[pymethods]
impl PlainQpuStmt {
    /// Perform type checking on this statement, but do not allow return
    /// statements. Used in the REPL.
    pub fn type_check_no_ret(&self, py: Python<'_>, env: &mut TypeEnv) -> PyResult<()> {
        self.stmt
            .typecheck(&mut env.env, /*expected_ret_type=*/ None)
            // Discard compute kind for now, since Python does not need it
            .map(|_compute_kind| ())
            .map_err(|err| get_err(py, ProgErrKind::Type, err.kind.to_string(), err.dbg))
    }
}

impl fmt::Display for PlainQpuStmt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.stmt)
    }
}

#[pyclass]
#[derive(Clone)]
pub struct FloatExpr {
    pub expr: meta::qpu::FloatExpr,
}

#[pymethods]
impl FloatExpr {
    #[classmethod]
    fn new_dim_expr(_cls: &Bound<'_, PyType>, expr: DimExpr, dbg: Option<DebugLoc>) -> Self {
        Self {
            expr: meta::qpu::FloatExpr::FloatDimExpr {
                expr: expr.dim_expr,
                dbg: dbg.map(|dbg| dbg.dbg),
            },
        }
    }

    #[classmethod]
    fn new_const(_cls: &Bound<'_, PyType>, val: f64, dbg: Option<DebugLoc>) -> Self {
        Self {
            expr: meta::qpu::FloatExpr::FloatConst {
                val,
                dbg: dbg.map(|dbg| dbg.dbg),
            },
        }
    }

    #[classmethod]
    fn new_sum(
        _cls: &Bound<'_, PyType>,
        left: FloatExpr,
        right: FloatExpr,
        dbg: Option<DebugLoc>,
    ) -> Self {
        Self {
            expr: meta::qpu::FloatExpr::FloatSum {
                left: Box::new(left.expr),
                right: Box::new(right.expr),
                dbg: dbg.map(|dbg| dbg.dbg),
            },
        }
    }

    #[classmethod]
    fn new_prod(
        _cls: &Bound<'_, PyType>,
        left: FloatExpr,
        right: FloatExpr,
        dbg: Option<DebugLoc>,
    ) -> Self {
        Self {
            expr: meta::qpu::FloatExpr::FloatProd {
                left: Box::new(left.expr),
                right: Box::new(right.expr),
                dbg: dbg.map(|dbg| dbg.dbg),
            },
        }
    }

    #[classmethod]
    fn new_div(
        _cls: &Bound<'_, PyType>,
        left: FloatExpr,
        right: FloatExpr,
        dbg: Option<DebugLoc>,
    ) -> Self {
        Self {
            expr: meta::qpu::FloatExpr::FloatDiv {
                left: Box::new(left.expr),
                right: Box::new(right.expr),
                dbg: dbg.map(|dbg| dbg.dbg),
            },
        }
    }

    #[classmethod]
    fn new_neg(_cls: &Bound<'_, PyType>, val: FloatExpr, dbg: Option<DebugLoc>) -> Self {
        Self {
            expr: meta::qpu::FloatExpr::FloatNeg {
                val: Box::new(val.expr),
                dbg: dbg.map(|dbg| dbg.dbg),
            },
        }
    }
}

#[pyclass]
#[derive(Clone)]
pub struct Vector {
    vec: meta::qpu::MetaVector,
}

#[pymethods]
impl Vector {
    #[classmethod]
    fn new_vector_alias(_cls: &Bound<'_, PyType>, name: String, dbg: Option<DebugLoc>) -> Self {
        Self {
            vec: meta::qpu::MetaVector::VectorAlias {
                name,
                dbg: dbg.map(|dbg| dbg.dbg),
            },
        }
    }

    #[classmethod]
    fn new_vector_symbol(_cls: &Bound<'_, PyType>, sym: char, dbg: Option<DebugLoc>) -> Self {
        Self {
            vec: meta::qpu::MetaVector::VectorSymbol {
                sym,
                dbg: dbg.map(|dbg| dbg.dbg),
            },
        }
    }

    #[classmethod]
    fn new_vector_broadcast_tensor(
        _cls: &Bound<'_, PyType>,
        val: Vector,
        factor: DimExpr,
        dbg: Option<DebugLoc>,
    ) -> Self {
        Self {
            vec: meta::qpu::MetaVector::VectorBroadcastTensor {
                val: Box::new(val.vec),
                factor: factor.dim_expr,
                dbg: dbg.map(|dbg| dbg.dbg),
            },
        }
    }

    #[classmethod]
    fn new_zero_vector(_cls: &Bound<'_, PyType>, dbg: Option<DebugLoc>) -> Self {
        Self {
            vec: meta::qpu::MetaVector::ZeroVector {
                dbg: dbg.map(|dbg| dbg.dbg),
            },
        }
    }

    #[classmethod]
    fn new_one_vector(_cls: &Bound<'_, PyType>, dbg: Option<DebugLoc>) -> Self {
        Self {
            vec: meta::qpu::MetaVector::OneVector {
                dbg: dbg.map(|dbg| dbg.dbg),
            },
        }
    }

    #[classmethod]
    fn new_pad_vector(_cls: &Bound<'_, PyType>, dbg: Option<DebugLoc>) -> Self {
        Self {
            vec: meta::qpu::MetaVector::PadVector {
                dbg: dbg.map(|dbg| dbg.dbg),
            },
        }
    }

    #[classmethod]
    fn new_target_vector(_cls: &Bound<'_, PyType>, dbg: Option<DebugLoc>) -> Self {
        Self {
            vec: meta::qpu::MetaVector::TargetVector {
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
            vec: meta::qpu::MetaVector::UniformVectorSuperpos {
                q1: Box::new(q1.vec.clone()),
                q2: Box::new(q2.vec.clone()),
                dbg: dbg.map(|dbg| dbg.dbg),
            },
        }
    }

    #[classmethod]
    fn new_vector_bi_tensor(
        _cls: &Bound<'_, PyType>,
        left: Vector,
        right: Vector,
        dbg: Option<DebugLoc>,
    ) -> Self {
        Self {
            vec: meta::qpu::MetaVector::VectorBiTensor {
                left: Box::new(left.vec),
                right: Box::new(right.vec),
                dbg: dbg.map(|dbg| dbg.dbg),
            },
        }
    }

    #[classmethod]
    fn new_vector_tilt(
        _cls: &Bound<'_, PyType>,
        q: Vector,
        angle_deg: FloatExpr,
        dbg: Option<DebugLoc>,
    ) -> Self {
        let dbg = dbg.map(|dbg| dbg.dbg);
        Self {
            vec: meta::qpu::MetaVector::VectorTilt {
                q: Box::new(q.vec.clone()),
                angle_deg: angle_deg.expr,
                dbg,
            },
        }
    }

    #[classmethod]
    fn new_vector_unit(_cls: &Bound<'_, PyType>, dbg: Option<DebugLoc>) -> Self {
        Self {
            vec: meta::qpu::MetaVector::VectorUnit {
                dbg: dbg.map(|dbg| dbg.dbg),
            },
        }
    }
}

#[pyclass]
#[derive(Clone)]
pub struct BasisGenerator {
    generator: meta::qpu::MetaBasisGenerator,
}

#[pymethods]
impl BasisGenerator {
    #[classmethod]
    fn new_basis_generator_macro(
        _cls: &Bound<'_, PyType>,
        name: String,
        arg: Basis,
        dbg: Option<DebugLoc>,
    ) -> Self {
        Self {
            generator: meta::qpu::MetaBasisGenerator::BasisGeneratorMacro {
                name,
                arg: Box::new(arg.basis),
                dbg: dbg.map(|dbg| dbg.dbg),
            },
        }
    }

    #[classmethod]
    fn new_revolve(
        _cls: &Bound<'_, PyType>,
        v1: Vector,
        v2: Vector,
        dbg: Option<DebugLoc>,
    ) -> Self {
        Self {
            generator: meta::qpu::MetaBasisGenerator::Revolve {
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
    basis: meta::qpu::MetaBasis,
}

#[pymethods]
impl Basis {
    #[classmethod]
    fn new_basis_alias(_cls: &Bound<'_, PyType>, name: String, dbg: Option<DebugLoc>) -> Self {
        Self {
            basis: meta::qpu::MetaBasis::BasisAlias {
                name,
                dbg: dbg.map(|dbg| dbg.dbg),
            },
        }
    }

    #[classmethod]
    fn new_basis_alias_rec(
        _cls: &Bound<'_, PyType>,
        name: String,
        param: DimExpr,
        dbg: Option<DebugLoc>,
    ) -> Self {
        Self {
            basis: meta::qpu::MetaBasis::BasisAliasRec {
                name,
                param: param.dim_expr,
                dbg: dbg.map(|dbg| dbg.dbg),
            },
        }
    }

    #[classmethod]
    fn new_basis_broadcast_tensor(
        _cls: &Bound<'_, PyType>,
        val: Basis,
        factor: DimExpr,
        dbg: Option<DebugLoc>,
    ) -> Self {
        Self {
            basis: meta::qpu::MetaBasis::BasisBroadcastTensor {
                val: Box::new(val.basis),
                factor: factor.dim_expr,
                dbg: dbg.map(|dbg| dbg.dbg),
            },
        }
    }

    #[classmethod]
    fn new_empty_basis_literal(_cls: &Bound<'_, PyType>, dbg: Option<DebugLoc>) -> Self {
        Self {
            basis: meta::qpu::MetaBasis::EmptyBasisLiteral {
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
            basis: meta::qpu::MetaBasis::BasisLiteral {
                vecs: vecs.iter().map(|vec| vec.vec.clone()).collect(),
                dbg: dbg.map(|dbg| dbg.dbg),
            },
        }
    }

    #[classmethod]
    fn new_basis_bi_tensor(
        _cls: &Bound<'_, PyType>,
        left: Basis,
        right: Basis,
        dbg: Option<DebugLoc>,
    ) -> Self {
        Self {
            basis: meta::qpu::MetaBasis::BasisBiTensor {
                left: Box::new(left.basis),
                right: Box::new(right.basis),
                dbg: dbg.map(|dbg| dbg.dbg),
            },
        }
    }

    #[classmethod]
    fn new_apply_basis_generator(
        _cls: &Bound<'_, PyType>,
        basis: Basis,
        generator: BasisGenerator,
        dbg: Option<DebugLoc>,
    ) -> Self {
        Self {
            basis: meta::qpu::MetaBasis::ApplyBasisGenerator {
                basis: Box::new(basis.basis),
                generator: generator.generator,
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
    pub expr: meta::qpu::MetaExpr,
}

impl fmt::Display for QpuExpr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.expr)
    }
}

#[pymethods]
impl QpuExpr {
    #[classmethod]
    fn new_expr_macro(
        _cls: &Bound<'_, PyType>,
        name: String,
        arg: QpuExpr,
        dbg: Option<DebugLoc>,
    ) -> Self {
        Self {
            expr: meta::qpu::MetaExpr::ExprMacro {
                name,
                arg: Box::new(arg.expr),
                dbg: dbg.map(|dbg| dbg.dbg),
            },
        }
    }

    #[classmethod]
    fn new_basis_macro(
        _cls: &Bound<'_, PyType>,
        name: String,
        arg: Basis,
        dbg: Option<DebugLoc>,
    ) -> Self {
        Self {
            expr: meta::qpu::MetaExpr::BasisMacro {
                name,
                arg: Box::new(arg.basis),
                dbg: dbg.map(|dbg| dbg.dbg),
            },
        }
    }

    #[classmethod]
    fn new_broadcast_tensor(
        _cls: &Bound<'_, PyType>,
        val: QpuExpr,
        factor: DimExpr,
        dbg: Option<DebugLoc>,
    ) -> Self {
        Self {
            expr: meta::qpu::MetaExpr::BroadcastTensor {
                val: Box::new(val.expr),
                factor: factor.dim_expr,
                dbg: dbg.map(|dbg| dbg.dbg),
            },
        }
    }

    #[classmethod]
    fn new_instantiate(
        _cls: &Bound<'_, PyType>,
        name: String,
        param: DimExpr,
        dbg: Option<DebugLoc>,
    ) -> Self {
        Self {
            expr: meta::qpu::MetaExpr::Instantiate {
                name,
                param: param.dim_expr,
                dbg: dbg.map(|dbg| dbg.dbg),
            },
        }
    }

    #[classmethod]
    fn new_repeat(
        _cls: &Bound<'_, PyType>,
        for_each: QpuExpr,
        iter_var: String,
        upper_bound: DimExpr,
        dbg: Option<DebugLoc>,
    ) -> Self {
        Self {
            expr: meta::qpu::MetaExpr::Repeat {
                for_each: Box::new(for_each.expr),
                iter_var,
                upper_bound: upper_bound.dim_expr,
                dbg: dbg.map(|dbg| dbg.dbg),
            },
        }
    }

    #[classmethod]
    fn new_variable(_cls: &Bound<'_, PyType>, name: String, dbg: Option<DebugLoc>) -> Self {
        Self {
            expr: meta::qpu::MetaExpr::Variable {
                name,
                dbg: dbg.map(|dbg| dbg.dbg),
            },
        }
    }

    #[classmethod]
    fn new_unit_literal(_cls: &Bound<'_, PyType>, dbg: Option<DebugLoc>) -> Self {
        Self {
            expr: meta::qpu::MetaExpr::UnitLiteral {
                dbg: dbg.map(|dbg| dbg.dbg),
            },
        }
    }

    #[classmethod]
    fn new_embed_classical(
        _cls: &Bound<'_, PyType>,
        func: QpuExpr,
        embed_kind: EmbedKind,
        dbg: Option<DebugLoc>,
    ) -> Self {
        Self {
            expr: meta::qpu::MetaExpr::EmbedClassical {
                func: Box::new(func.expr),
                embed_kind: embed_kind.to_ast_kind(),
                dbg: dbg.map(|dbg| dbg.dbg),
            },
        }
    }

    #[classmethod]
    fn new_measure(_cls: &Bound<'_, PyType>, basis: Basis, dbg: Option<DebugLoc>) -> Self {
        Self {
            expr: meta::qpu::MetaExpr::Measure {
                basis: basis.basis,
                dbg: dbg.map(|dbg| dbg.dbg),
            },
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
            expr: meta::qpu::MetaExpr::Pipe {
                lhs: Box::new(lhs.expr),
                rhs: Box::new(rhs.expr),
                dbg: dbg.map(|dbg| dbg.dbg),
            },
        }
    }

    #[classmethod]
    fn new_discard(_cls: &Bound<'_, PyType>, dbg: Option<DebugLoc>) -> Self {
        Self {
            expr: meta::qpu::MetaExpr::Discard {
                dbg: dbg.map(|dbg| dbg.dbg),
            },
        }
    }

    #[classmethod]
    fn new_bi_tensor(
        _cls: &Bound<'_, PyType>,
        left: QpuExpr,
        right: QpuExpr,
        dbg: Option<DebugLoc>,
    ) -> Self {
        Self {
            expr: meta::qpu::MetaExpr::BiTensor {
                left: Box::new(left.expr),
                right: Box::new(right.expr),
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
            expr: meta::qpu::MetaExpr::BasisTranslation {
                bin: bin.basis,
                bout: bout.basis,
                dbg: dbg.map(|dbg| dbg.dbg),
            },
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
            expr: meta::qpu::MetaExpr::Predicated {
                then_func: Box::new(then_func.expr),
                else_func: Box::new(else_func.expr),
                pred: pred.basis,
                dbg: dbg.map(|dbg| dbg.dbg),
            },
        }
    }

    #[classmethod]
    fn new_non_uniform_superpos(
        _cls: &Bound<'_, PyType>,
        pairs: Vec<(f64, Vector)>,
        dbg: Option<DebugLoc>,
    ) -> Self {
        let dbg = dbg.map(|dbg| dbg.dbg);
        Self {
            expr: meta::qpu::MetaExpr::NonUniformSuperpos {
                pairs: pairs
                    .into_iter()
                    .map(|(prob, vector)| {
                        let prob_expr = meta::qpu::FloatExpr::FloatConst {
                            val: prob,
                            dbg: dbg.clone(),
                        };
                        (prob_expr, vector.vec)
                    })
                    .collect(),
                dbg: dbg,
            },
        }
    }

    #[classmethod]
    fn new_ensemble(
        _cls: &Bound<'_, PyType>,
        pairs: Vec<(f64, Vector)>,
        dbg: Option<DebugLoc>,
    ) -> Self {
        let dbg = dbg.map(|dbg| dbg.dbg);
        Self {
            expr: meta::qpu::MetaExpr::Ensemble {
                pairs: pairs
                    .into_iter()
                    .map(|(prob, vector)| {
                        let prob_expr = meta::qpu::FloatExpr::FloatConst {
                            val: prob,
                            dbg: dbg.clone(),
                        };
                        (prob_expr, vector.vec)
                    })
                    .collect(),
                dbg: dbg,
            },
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
            expr: meta::qpu::MetaExpr::Conditional {
                then_expr: Box::new(then_expr.expr),
                else_expr: Box::new(else_expr.expr),
                cond: Box::new(cond.expr),
                dbg: dbg.map(|dbg| dbg.dbg),
            },
        }
    }

    #[classmethod]
    fn new_qlit(_cls: &Bound<'_, PyType>, qlit: Vector) -> Self {
        Self {
            expr: meta::qpu::MetaExpr::QLit { vec: qlit.vec },
        }
    }

    #[classmethod]
    fn new_bit_literal(
        _cls: &Bound<'_, PyType>,
        val: UBigWrap,
        n_bits: usize,
        dbg: Option<DebugLoc>,
    ) -> Self {
        let dbg = dbg.map(|dbg| dbg.dbg);
        Self {
            expr: meta::qpu::MetaExpr::BitLiteral {
                val: val.0,
                n_bits: meta::DimExpr::DimConst {
                    val: n_bits.into(),
                    dbg: dbg.clone(),
                },
                dbg,
            },
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
pub struct ExprMacroPattern {
    pub pat: meta::qpu::ExprMacroPattern,
}

impl fmt::Display for ExprMacroPattern {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.pat)
    }
}

#[pymethods]
impl ExprMacroPattern {
    #[classmethod]
    fn new_any_expr(_cls: &Bound<'_, PyType>, name: String, dbg: Option<DebugLoc>) -> Self {
        Self {
            pat: meta::qpu::ExprMacroPattern::AnyExpr {
                name,
                dbg: dbg.map(|dbg| dbg.dbg),
            },
        }
    }

    /// Return the Debug form of this node from __repr__(). By contrast,
    /// __str__() returns the Display form.
    pub fn __repr__(&self) -> String {
        format!("{:?}", self.pat)
    }
}

#[pyclass(str, eq)]
#[derive(Clone, PartialEq)]
pub struct BasisMacroPattern {
    pub pat: meta::qpu::BasisMacroPattern,
}

impl fmt::Display for BasisMacroPattern {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.pat)
    }
}

#[pymethods]
impl BasisMacroPattern {
    #[classmethod]
    fn new_any_basis(_cls: &Bound<'_, PyType>, name: String, dbg: Option<DebugLoc>) -> Self {
        Self {
            pat: meta::qpu::BasisMacroPattern::AnyBasis {
                name,
                dbg: dbg.map(|dbg| dbg.dbg),
            },
        }
    }

    #[classmethod]
    fn new_basis_literal(
        _cls: &Bound<'_, PyType>,
        vec_names: Vec<String>,
        dbg: Option<DebugLoc>,
    ) -> Self {
        Self {
            pat: meta::qpu::BasisMacroPattern::BasisLiteral {
                vec_names,
                dbg: dbg.map(|dbg| dbg.dbg),
            },
        }
    }

    /// Return the Debug form of this node from __repr__(). By contrast,
    /// __str__() returns the Display form.
    pub fn __repr__(&self) -> String {
        format!("{:?}", self.pat)
    }
}

#[pyclass(str, eq)]
#[derive(Clone, PartialEq)]
pub struct RecDefParam {
    pub param: meta::qpu::RecDefParam,
}

impl fmt::Display for RecDefParam {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.param)
    }
}

#[pymethods]
impl RecDefParam {
    #[classmethod]
    fn new_base(_cls: &Bound<'_, PyType>, param: IBigWrap) -> Self {
        Self {
            param: meta::qpu::RecDefParam::Base(param.0),
        }
    }

    #[classmethod]
    fn new_rec(_cls: &Bound<'_, PyType>, param: String) -> Self {
        Self {
            param: meta::qpu::RecDefParam::Rec(param),
        }
    }

    /// Return the Debug form of this node from __repr__(). By contrast,
    /// __str__() returns the Display form.
    pub fn __repr__(&self) -> String {
        format!("{:?}", self.param)
    }
}

#[pyclass(str, eq)]
#[derive(Clone, PartialEq)]
pub struct QpuStmt {
    pub stmt: meta::qpu::MetaStmt,
}

impl fmt::Display for QpuStmt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.stmt)
    }
}

#[pymethods]
impl QpuStmt {
    #[classmethod]
    fn new_expr_macro_def(
        _cls: &Bound<'_, PyType>,
        lhs_pat: ExprMacroPattern,
        lhs_name: String,
        rhs: QpuExpr,
        dbg: Option<DebugLoc>,
    ) -> Self {
        Self {
            stmt: meta::qpu::MetaStmt::ExprMacroDef {
                lhs_pat: lhs_pat.pat,
                lhs_name,
                rhs: rhs.expr,
                dbg: dbg.map(|dbg| dbg.dbg),
            },
        }
    }

    #[classmethod]
    fn new_basis_macro_def(
        _cls: &Bound<'_, PyType>,
        lhs_pat: BasisMacroPattern,
        lhs_name: String,
        rhs: QpuExpr,
        dbg: Option<DebugLoc>,
    ) -> Self {
        Self {
            stmt: meta::qpu::MetaStmt::BasisMacroDef {
                lhs_pat: lhs_pat.pat,
                lhs_name,
                rhs: rhs.expr,
                dbg: dbg.map(|dbg| dbg.dbg),
            },
        }
    }

    #[classmethod]
    fn new_basis_generator_macro_def(
        _cls: &Bound<'_, PyType>,
        lhs_pat: BasisMacroPattern,
        lhs_name: String,
        rhs: BasisGenerator,
        dbg: Option<DebugLoc>,
    ) -> Self {
        Self {
            stmt: meta::qpu::MetaStmt::BasisGeneratorMacroDef {
                lhs_pat: lhs_pat.pat,
                lhs_name,
                rhs: rhs.generator,
                dbg: dbg.map(|dbg| dbg.dbg),
            },
        }
    }

    #[classmethod]
    fn new_vector_symbol_def(
        _cls: &Bound<'_, PyType>,
        lhs: char,
        rhs: Vector,
        dbg: Option<DebugLoc>,
    ) -> Self {
        Self {
            stmt: meta::qpu::MetaStmt::VectorSymbolDef {
                lhs,
                rhs: rhs.vec,
                dbg: dbg.map(|dbg| dbg.dbg),
            },
        }
    }

    #[classmethod]
    fn new_basis_alias_def(
        _cls: &Bound<'_, PyType>,
        lhs: String,
        rhs: Basis,
        dbg: Option<DebugLoc>,
    ) -> Self {
        Self {
            stmt: meta::qpu::MetaStmt::BasisAliasDef {
                lhs,
                rhs: rhs.basis,
                dbg: dbg.map(|dbg| dbg.dbg),
            },
        }
    }

    #[classmethod]
    fn new_basis_alias_rec_def(
        _cls: &Bound<'_, PyType>,
        lhs: String,
        param: RecDefParam,
        rhs: Basis,
        dbg: Option<DebugLoc>,
    ) -> Self {
        Self {
            stmt: meta::qpu::MetaStmt::BasisAliasRecDef {
                lhs,
                param: param.param,
                rhs: rhs.basis,
                dbg: dbg.map(|dbg| dbg.dbg),
            },
        }
    }

    #[classmethod]
    fn new_expr(_cls: &Bound<'_, PyType>, expr: QpuExpr) -> Self {
        Self {
            stmt: meta::qpu::MetaStmt::Expr { expr: expr.expr },
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
            stmt: meta::qpu::MetaStmt::Assign {
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
        rhs: QpuExpr,
        dbg: Option<DebugLoc>,
    ) -> Self {
        Self {
            stmt: meta::qpu::MetaStmt::UnpackAssign {
                lhs,
                rhs: rhs.expr,
                dbg: dbg.map(|dbg| dbg.dbg),
            },
        }
    }

    #[classmethod]
    fn new_return(_cls: &Bound<'_, PyType>, val: QpuExpr, dbg: Option<DebugLoc>) -> Self {
        Self {
            stmt: meta::qpu::MetaStmt::Return {
                val: val.expr,
                dbg: dbg.map(|dbg| dbg.dbg),
            },
        }
    }

    pub fn lower<'py>(
        &self,
        py: Python<'py>,
        env: &mut MacroEnv,
        plain_ty_env: &TypeEnv,
    ) -> PyResult<PlainQpuStmt> {
        self.stmt
            .lower(&mut env.env, &plain_ty_env.env)
            .map(|ast_stmt| PlainQpuStmt { stmt: ast_stmt })
            .map_err(|err| get_err(py, ProgErrKind::Expand, err.kind.to_string(), err.dbg))
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
    pub function_def: meta::MetaFunctionDef<meta::qpu::MetaStmt>,
}

#[pymethods]
impl QpuFunctionDef {
    #[new]
    fn new(
        name: String,
        args: Vec<(Option<Type>, String)>,
        ret_type: Option<Type>,
        body: Vec<QpuStmt>,
        is_rev: bool,
        dim_vars: Vec<String>,
        dbg: Option<DebugLoc>,
    ) -> Self {
        Self {
            function_def: meta::MetaFunctionDef {
                name,
                args: args
                    .into_iter()
                    .map(|(arg_ty, arg_name)| (arg_ty.map(|arg_ty| arg_ty.ty), arg_name))
                    .collect(),
                ret_type: ret_type.map(|ret_type| ret_type.ty.clone()),
                body: body.into_iter().map(|stmt| stmt.stmt).collect(),
                is_rev,
                dim_vars,
                dbg: dbg.map(|dbg| dbg.dbg),
            },
        }
    }

    fn get_name(&self) -> String {
        self.function_def.name.to_string()
    }

    fn add_prelude(&mut self, prelude: QpuPrelude) {
        self.function_def.add_prelude(&prelude.prelude);
    }
}

#[pyclass]
#[derive(Clone)]
pub struct QpuPrelude {
    pub prelude: meta::Prelude<meta::qpu::MetaStmt>,
}

#[pymethods]
impl QpuPrelude {
    #[new]
    fn new(body: Vec<QpuStmt>, dbg: Option<DebugLoc>) -> Self {
        Self {
            prelude: meta::Prelude {
                body: body.into_iter().map(|stmt| stmt.stmt).collect(),
                dbg: dbg.map(|dbg| dbg.dbg),
            },
        }
    }

    fn get_stmts(&self) -> Vec<QpuStmt> {
        let meta::Prelude { body, .. } = &self.prelude;
        body.iter().cloned().map(|stmt| QpuStmt { stmt }).collect()
    }
}
