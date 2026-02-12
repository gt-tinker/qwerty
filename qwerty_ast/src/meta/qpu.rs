use crate::{
    ast::{self, RegKind, Trivializable, qpu::EmbedKind},
    dbg::DebugLoc,
    error::{LowerError, LowerErrorKind},
    meta::{DimExpr, DimVar, Progress, classical, expand::MacroEnv},
};
use dashu::integer::{IBig, UBig};
use itertools::Itertools;
use qwerty_ast_macros::{gen_rebuild, rebuild, rewrite_match, rewrite_ty, visitor_write};
use std::fmt;

#[gen_rebuild {
    substitute_dim_var(
        more_copied_args(dim_var: &DimVar, new_dim_expr: &DimExpr),
        recurse_attrs,
    ),
    expand(
        rewrite(expand_rewriter),
        progress(Progress),
        more_copied_args(env: &MacroEnv),
        result_err(LowerError),
        recurse_attrs,
    ),
}]
#[derive(Debug, Clone, PartialEq)]
pub enum FloatExpr {
    /// A dimension variable expression used in a float expression. Example
    /// syntax:
    /// ```text
    /// '1' @ 3.141593 / N
    /// ```
    FloatDimExpr {
        expr: DimExpr,
        dbg: Option<DebugLoc>,
    },

    /// A float contant. Example syntax:
    /// ```text
    /// 3.141593
    /// ```
    FloatConst { val: f64, dbg: Option<DebugLoc> },

    /// A sum of float values. Example syntax:
    /// ```text
    /// 3.141593 + 2.0
    /// ```
    FloatSum {
        left: Box<FloatExpr>,
        right: Box<FloatExpr>,
        dbg: Option<DebugLoc>,
    },

    /// A product of float values. Example syntax:
    /// ```text
    /// 3.141593 * 2.0
    /// ```
    FloatProd {
        left: Box<FloatExpr>,
        right: Box<FloatExpr>,
        dbg: Option<DebugLoc>,
    },

    /// A quotient of float values. Example syntax:
    /// ```text
    /// 3.141593 / 2.0
    /// ```
    FloatDiv {
        left: Box<FloatExpr>,
        right: Box<FloatExpr>,
        dbg: Option<DebugLoc>,
    },

    /// A negated float values. Example syntax:
    /// ```text
    /// -3.141593
    /// ```
    FloatNeg {
        val: Box<FloatExpr>,
        dbg: Option<DebugLoc>,
    },
}

impl FloatExpr {
    /// Extract a constant double-precision float from this float expression or
    /// return an error if it is not fully folded yet.
    pub fn extract(self) -> Result<f64, LowerError> {
        match self {
            FloatExpr::FloatConst { val, .. } => Ok(val),
            FloatExpr::FloatDimExpr { dbg, .. }
            | FloatExpr::FloatSum { dbg, .. }
            | FloatExpr::FloatProd { dbg, .. }
            | FloatExpr::FloatDiv { dbg, .. }
            | FloatExpr::FloatNeg { dbg, .. } => Err(LowerError {
                kind: LowerErrorKind::NotFullyFolded,
                dbg,
            }),
        }
    }
}

impl fmt::Display for FloatExpr {
    /// Returns a representation of a dimension variable expression that
    /// matches the syntax in the Python DSL.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        visitor_write! {FloatExpr, self,
            FloatExpr::FloatDimExpr { expr, .. } => write!(f, "{}", expr),
            FloatExpr::FloatConst { val, .. } => write!(f, "{}", val),
            FloatExpr::FloatSum { left, right, .. } => write!(f, "({!})+({!})", *left, *right),
            FloatExpr::FloatProd { left, right, .. } => write!(f, "({!})*({!})", *left, *right),
            FloatExpr::FloatDiv { left, right, .. } => write!(f, "({!})/({!})", *left, *right),
            FloatExpr::FloatNeg { val, .. } => write!(f, "-({!})", *val),
        }
    }
}

#[gen_rebuild {
    substitute_dim_var(
        more_copied_args(dim_var: &DimVar, new_dim_expr: &DimExpr),
        recurse_attrs,
    ),
    substitute_vector_alias(
        rewrite(substitute_vector_alias_rewriter),
        more_copied_args(vector_alias: &str, new_vector: &MetaVector),
    ),
    expand(
        rewrite(expand_rewriter),
        progress(Progress),
        more_copied_args(env: &MacroEnv),
        result_err(LowerError),
        recurse_attrs,
    ),
    extract(
        rewrite(extract_rewriter),
        rewrite_to(
            MetaVector => ast::qpu::Vector,
            FloatExpr => f64,
        ),
        result_err(LowerError),
        recurse_attrs,
    ),
}]
#[derive(Debug, Clone, PartialEq)]
pub enum MetaVector {
    /// A name for a vector. Currently used only in macro definitions.
    /// Example syntax:
    /// ```text
    /// {bv1, bv2}.flip = {bv1, bv2} >> {bv2, bv1}
    ///                    ^^^  ^^^      ^^^  ^^^
    /// ```
    VectorAlias { name: String, dbg: Option<DebugLoc> },

    /// A vector symbol that should eventually expanded to a MetaVector.
    /// Example syntax:
    /// ```text
    /// 'p'
    /// ```
    VectorSymbol { sym: char, dbg: Option<DebugLoc> },

    /// An n-fold tensor product of a vector. Example syntax:
    /// ```text
    /// 'p'**N
    /// ```
    VectorBroadcastTensor {
        #[gen_rebuild::skip_recurse(extract)]
        val: Box<MetaVector>,
        #[gen_rebuild::skip_recurse(extract)]
        factor: DimExpr,
        dbg: Option<DebugLoc>,
    },

    /// The first standard basis vector, |0⟩. Example syntax:
    /// ```text
    /// __SYM_STD0__()
    /// ```
    ZeroVector { dbg: Option<DebugLoc> },

    /// The second standard basis vector, |1⟩. Example syntax:
    /// ```text
    /// __SYM_STD1__()
    /// ```
    OneVector { dbg: Option<DebugLoc> },

    /// The pad atom. Example syntax:
    /// ```text
    /// __SYM_PAD__()
    /// ```
    PadVector { dbg: Option<DebugLoc> },

    /// The target atom. Example syntax:
    /// ```text
    /// __SYM_TARGET__()
    /// ```
    TargetVector { dbg: Option<DebugLoc> },

    /// Tilts a vector. Example syntax:
    /// ```text
    /// '1' @ 180
    /// ```
    VectorTilt {
        q: Box<MetaVector>,
        angle_deg: FloatExpr,
        dbg: Option<DebugLoc>,
    },

    /// A uniform vector superposition. Example syntax:
    /// ```text
    /// '0' + '1'
    /// ```
    UniformVectorSuperpos {
        q1: Box<MetaVector>,
        q2: Box<MetaVector>,
        dbg: Option<DebugLoc>,
    },

    /// A tensor product. Example syntax:
    /// ```text
    /// '0' * '1'
    /// ```
    VectorBiTensor {
        left: Box<MetaVector>,
        right: Box<MetaVector>,
        dbg: Option<DebugLoc>,
    },

    /// An empty vector. Example syntax:
    /// ```text
    /// ''
    /// ```
    VectorUnit { dbg: Option<DebugLoc> },
}

impl MetaVector {
    /// Returns the debug location for this vector.
    pub fn get_dbg(&self) -> Option<DebugLoc> {
        match self {
            MetaVector::VectorAlias { dbg, .. }
            | MetaVector::VectorSymbol { dbg, .. }
            | MetaVector::VectorBroadcastTensor { dbg, .. }
            | MetaVector::ZeroVector { dbg }
            | MetaVector::OneVector { dbg }
            | MetaVector::PadVector { dbg }
            | MetaVector::TargetVector { dbg }
            | MetaVector::VectorTilt { dbg, .. }
            | MetaVector::UniformVectorSuperpos { dbg, .. }
            | MetaVector::VectorBiTensor { dbg, .. }
            | MetaVector::VectorUnit { dbg } => dbg.clone(),
        }
    }

    /// Extracts a plain-AST `@qpu` Vector from this metaQwerty vector. Returns
    /// an error instead if metaQwerty constructs are still present.
    pub fn extract(self) -> Result<ast::qpu::Vector, LowerError> {
        rebuild!(MetaVector, self, extract)
    }

    pub(crate) fn extract_rewriter(
        rewritten: rewrite_ty!(MetaVector, extract),
    ) -> Result<ast::qpu::Vector, LowerError> {
        rewrite_match! {MetaVector, extract, rewritten,
            ZeroVector { dbg } => Ok(ast::qpu::Vector::ZeroVector { dbg }),

            OneVector { dbg } => Ok(ast::qpu::Vector::OneVector { dbg }),

            PadVector { dbg } => Ok(ast::qpu::Vector::PadVector { dbg }),

            TargetVector { dbg } => Ok(ast::qpu::Vector::TargetVector { dbg }),

            VectorTilt { q, angle_deg, dbg } => {
                Ok(ast::qpu::Vector::VectorTilt {
                    q: Box::new(q),
                    angle_deg,
                    dbg,
                })
            }

            UniformVectorSuperpos { q1, q2, dbg } => {
                Ok(ast::qpu::Vector::UniformVectorSuperpos {
                    q1: Box::new(q1),
                    q2: Box::new(q2),
                    dbg,
                })
            }

            VectorBiTensor { left, right, dbg } => {
                Ok(ast::qpu::Vector::VectorTensor {
                    qs: vec![left, right],
                    dbg,
                })
            }

            VectorUnit { dbg } => Ok(ast::qpu::Vector::VectorUnit { dbg }),

            VectorAlias { dbg, .. }
            | VectorSymbol { dbg, .. }
            | VectorBroadcastTensor { dbg, .. } => Err(LowerError {
                kind: LowerErrorKind::NotFullyFolded,
                dbg,
            }),
        }
    }
}

// TODO: don't duplicate with qpu.rs
impl fmt::Display for MetaVector {
    /// Represents a vector in a human-readable form for error messages sent
    /// back to the programmer.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        visitor_write! {MetaVector, self,
            MetaVector::VectorAlias { name, .. } => write!(f, "{}", name),
            MetaVector::VectorSymbol { sym, .. } => write!(f, "'{}'", sym),
            MetaVector::VectorBroadcastTensor { val, factor, .. } => write!(f, "({!})*({})", *val, factor),
            MetaVector::ZeroVector { .. } => write!(f, "'0'"),
            MetaVector::OneVector { .. } => write!(f, "'1'"),
            MetaVector::PadVector { .. } => write!(f, "'?'"),
            MetaVector::TargetVector { .. } => write!(f, "'_'"),
            MetaVector::VectorTilt { q, angle_deg, .. } => write!(f, "({!})@({})", *q, angle_deg),
            MetaVector::UniformVectorSuperpos { q1, q2, .. } => {
                write!(f, "({!}) + ({!})", *q1, *q2)
            }
            MetaVector::VectorBiTensor { left, right, .. } => {
                write!(f, "({!}) * ({!})", *left, *right)
            }
            MetaVector::VectorUnit { .. } => write!(f, "''"),
        }
    }
}

#[gen_rebuild {
    substitute_dim_var(
        more_copied_args(dim_var: &DimVar, new_dim_expr: &DimExpr),
        recurse_attrs,
    ),
    substitute_vector_alias(
        more_copied_args(vector_alias: &str, new_vector: &MetaVector),
        recurse_attrs,
    ),
    substitute_basis_alias(
        more_copied_args(basis_alias: &str, new_basis: &MetaBasis),
        recurse_attrs,
    ),
    expand(
        rewrite(expand_rewriter),
        progress(Progress),
        more_copied_args(env: &MacroEnv),
        result_err(LowerError),
        recurse_attrs,
    ),
    extract(
        rewrite(extract_rewriter),
        rewrite_to(
            MetaBasisGenerator => ast::qpu::BasisGenerator,
            MetaBasis => ast::qpu::Basis,
            MetaVector => ast::qpu::Vector,
        ),
        result_err(LowerError),
        recurse_attrs,
    ),
}]
#[derive(Debug, Clone, PartialEq)]
pub enum MetaBasisGenerator {
    /// Invokes a macro. Example syntax:
    /// ```text
    /// {'0','1'}.revolve
    /// ```
    BasisGeneratorMacro {
        name: String,
        #[gen_rebuild::skip_recurse(extract)]
        arg: Box<MetaBasis>,
        dbg: Option<DebugLoc>,
    },

    /// A revolve generator, used to define the Fourier basis. Example
    /// syntax:
    /// ```text
    /// __REVOLVE__('p', 'm')
    /// ```
    Revolve {
        // There are no bases to substitute in these two fields
        #[gen_rebuild::skip_recurse(substitute_basis_alias)]
        v1: MetaVector,
        #[gen_rebuild::skip_recurse(substitute_basis_alias)]
        v2: MetaVector,
        dbg: Option<DebugLoc>,
    },
}

impl MetaBasisGenerator {
    /// Extracts a plain-AST `@qpu` BasisGenerator from this metaQwerty basis
    /// generator. Returns an error instead if metaQwerty constructs are still
    /// present.
    pub fn extract(self) -> Result<ast::qpu::BasisGenerator, LowerError> {
        rebuild!(MetaBasisGenerator, self, extract)
    }

    pub(crate) fn extract_rewriter(
        rewritten: rewrite_ty!(MetaBasisGenerator, extract),
    ) -> Result<ast::qpu::BasisGenerator, LowerError> {
        rewrite_match! {MetaBasisGenerator, extract, rewritten,
            Revolve { v1, v2, dbg } => {
                Ok(ast::qpu::BasisGenerator::Revolve { v1, v2, dbg })
            }

            BasisGeneratorMacro { dbg, .. } => Err(LowerError {
                kind: LowerErrorKind::NotFullyFolded,
                dbg,
            }),
        }
    }
}

// TODO: don't duplicate with qpu.rs
impl fmt::Display for MetaBasisGenerator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MetaBasisGenerator::BasisGeneratorMacro { name, arg, .. } => {
                write!(f, "({}).{}", arg, name)
            }
            MetaBasisGenerator::Revolve { v1, v2, .. } => {
                write!(f, "{{{},{}}}.revolve", v1, v2)
            }
        }
    }
}

#[gen_rebuild {
    substitute_dim_var(
        more_copied_args(dim_var: &DimVar, new_dim_expr: &DimExpr),
        recurse_attrs,
    ),
    substitute_basis_alias(
        rewrite(substitute_basis_alias_rewriter),
        more_copied_args(basis_alias: &str, new_basis: &MetaBasis),
    ),
    substitute_vector_alias(
        more_copied_args(vector_alias: &str, new_vector: &MetaVector),
        recurse_attrs,
    ),
    expand(
        rewrite(expand_rewriter),
        progress(Progress),
        more_copied_args(env: &MacroEnv),
        result_err(LowerError),
        recurse_attrs,
    ),
    extract(
        rewrite(extract_rewriter),
        rewrite_to(
            MetaBasis => ast::qpu::Basis,
            MetaBasisGenerator => ast::qpu::BasisGenerator,
            MetaVector => ast::qpu::Vector,
        ),
        result_err(LowerError),
        recurse_attrs,
    ),
}]
#[derive(Debug, Clone, PartialEq)]
pub enum MetaBasis {
    /// A basis alias name. Example syntax:
    /// ```text
    /// pm
    /// ```
    BasisAlias { name: String, dbg: Option<DebugLoc> },

    /// A recursive basis alias with a parameter. Example syntax:
    /// ```text
    /// fourier[N]
    /// ```
    BasisAliasRec {
        name: String,
        #[gen_rebuild::skip_recurse(substitute_basis_alias, substitute_vector_alias, extract)]
        param: DimExpr,
        dbg: Option<DebugLoc>,
    },

    /// An n-fold tensor product of a basis. Example syntax:
    /// ```text
    /// pm**N
    /// ```
    BasisBroadcastTensor {
        val: Box<MetaBasis>,
        #[gen_rebuild::skip_recurse(substitute_basis_alias, substitute_vector_alias, extract)]
        factor: DimExpr,
        dbg: Option<DebugLoc>,
    },

    /// A basis literal. Example syntax:
    /// ```text
    /// {'0', '1'}
    /// ```
    BasisLiteral {
        vecs: Vec<MetaVector>,
        dbg: Option<DebugLoc>,
    },

    /// An empty basis literal. Example syntax:
    /// ```text
    /// {}
    /// ```
    EmptyBasisLiteral { dbg: Option<DebugLoc> },

    /// Tensor product of bases. Example syntax:
    /// ```text
    /// {'0', '1'} * {'0', '1'}
    /// ```
    BasisBiTensor {
        left: Box<MetaBasis>,
        right: Box<MetaBasis>,
        dbg: Option<DebugLoc>,
    },

    /// Apply a basis generator. Example syntax:
    /// ```text
    /// {'0'+'1', '0'-'1'} // __REVOLVE__('0', '1')
    /// ```
    ApplyBasisGenerator {
        basis: Box<MetaBasis>,
        generator: MetaBasisGenerator,
        dbg: Option<DebugLoc>,
    },
}

impl MetaBasis {
    /// Extracts a plain-AST `@qpu` Basis from this metaQwerty basis.
    pub fn extract(self) -> Result<ast::qpu::Basis, LowerError> {
        rebuild!(MetaBasis, self, extract)
    }

    pub(crate) fn extract_rewriter(
        rewritten: rewrite_ty!(MetaBasis, extract),
    ) -> Result<ast::qpu::Basis, LowerError> {
        rewrite_match! {MetaBasis, extract, rewritten,
            BasisLiteral { vecs, dbg } => Ok(ast::qpu::Basis::BasisLiteral { vecs, dbg }),

            EmptyBasisLiteral { dbg } => Ok(ast::qpu::Basis::EmptyBasisLiteral { dbg }),

            BasisBiTensor { left, right, dbg } => {
                Ok(ast::qpu::Basis::BasisTensor {
                    bases: vec![left, right],
                    dbg,
                })
            }

            ApplyBasisGenerator { basis, generator, dbg } => {
                Ok(ast::qpu::Basis::ApplyBasisGenerator {
                    basis: Box::new(basis),
                    generator,
                    dbg,
                })
            }

            BasisAlias { dbg, .. }
            | BasisAliasRec { dbg, .. }
            | BasisBroadcastTensor { dbg, .. } => Err(LowerError {
                kind: LowerErrorKind::NotFullyFolded,
                dbg,
            }),
        }
    }
}

impl fmt::Display for MetaBasis {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        visitor_write! {MetaBasis, self,
            MetaBasis::BasisAlias { name, .. } => write!(f, "{}", name),
            MetaBasis::BasisAliasRec { name, param, .. } => write!(f, "{}[{}]", name, param),
            MetaBasis::BasisBroadcastTensor { val, factor, .. } => {
                write!(f, "({!})**({})", *val, factor)
            }
            MetaBasis::BasisLiteral { vecs, .. } => write!(f, "{{{}}}", vecs.iter().format(", ")),
            MetaBasis::EmptyBasisLiteral { .. } => write!(f, "{{}}"),
            MetaBasis::BasisBiTensor { left, right, .. } => write!(f, "({!})*({!})", *left, *right),
            MetaBasis::ApplyBasisGenerator { basis, generator, .. } => {
                write!(f, "({!}) // ({})", *basis, generator)
            }
        }
    }
}

#[gen_rebuild {
    expand_instantiations(
        rewrite(expand_instantiations_rewriter),
        result_err(LowerError),
        // fn rebuild<F>(root: MetaExpr, f: F)
        //     where F: FnMut(String, DimExpr, Option<DebugLoc>) -> Result<String, LowerError>
        // { ... }
        more_generic_params(F),
        more_moved_args(f: F),
        more_where(F: FnMut(String, DimExpr, Option<DebugLoc>) -> Result<String, LowerError>),
    ),
    substitute_dim_var(
        more_copied_args(dim_var: &DimVar, new_dim_expr: &DimExpr),
        recurse_attrs,
    ),
    substitute_vector_alias(
        more_copied_args(vector_alias: &str, new_vector: &MetaVector),
        recurse_attrs,
    ),
    substitute_basis_alias(
        more_copied_args(basis_alias: &str, new_basis: &MetaBasis),
        recurse_attrs,
    ),
    substitute_variable(
        rewrite(substitute_variable_rewriter),
        more_copied_args(var_name: &str, new_expr: &MetaExpr),
    ),
    expand(
        rewrite(expand_rewriter),
        progress(Progress),
        more_copied_args(env: &MacroEnv),
        result_err(LowerError),
        recurse_attrs,
    ),
    extract(
        rewrite(extract_rewriter),
        rewrite_to(
            MetaExpr => ast::qpu::Expr,
            MetaBasis => ast::qpu::Basis,
            MetaVector => ast::qpu::Vector,
            DimExpr => usize,
            FloatExpr => f64,
        ),
        result_err(LowerError),
        recurse_attrs,
    ),
}]
#[derive(Debug, Clone, PartialEq)]
pub enum MetaExpr {
    /// Another example:
    /// ```text
    /// my_classical_func.inplace
    /// ```
    ExprMacro {
        name: String,
        arg: Box<MetaExpr>,
        dbg: Option<DebugLoc>,
    },

    /// Invokes a macro with a basis argument. Example syntax:
    /// ```text
    /// {'0','1'}.measure
    /// ```
    BasisMacro {
        name: String,
        arg: MetaBasis,
        dbg: Option<DebugLoc>,
    },

    /// An n-fold tensor product. Example syntax:
    /// ```text
    /// id**N
    /// ```
    BroadcastTensor {
        val: Box<MetaExpr>,
        #[gen_rebuild::skip_recurse(substitute_basis_alias, substitute_vector_alias)]
        factor: DimExpr,
        dbg: Option<DebugLoc>,
    },

    /// Instantiate a function with a given dimension variable expression.
    /// Example syntax:
    /// ```text
    /// func[[N+1]]
    /// ```
    Instantiate {
        name: String,
        #[gen_rebuild::skip_recurse(substitute_basis_alias, substitute_vector_alias)]
        param: DimExpr,
        dbg: Option<DebugLoc>,
    },

    /// A macro that expands into a pipeline. Example syntax:
    /// ```text
    /// (op[[i] for i in range(N))
    /// ```
    Repeat {
        // We can't expand this yet because the iter_var would not be defined
        // in the env, but that is not an error
        #[gen_rebuild::skip_recurse(expand)]
        for_each: Box<MetaExpr>,
        iter_var: String,
        #[gen_rebuild::skip_recurse(substitute_basis_alias, substitute_vector_alias)]
        upper_bound: DimExpr,
        dbg: Option<DebugLoc>,
    },

    /// A variable name used in an expression. Example syntax:
    /// ```text
    /// my_var
    /// ```
    Variable { name: String, dbg: Option<DebugLoc> },

    /// A unit literal. Represents an empty register or void. Example syntax:
    /// ```text
    /// []
    /// ```
    UnitLiteral { dbg: Option<DebugLoc> },

    /// Embeds a classical function into a quantum context. Example syntax:
    /// ```text
    /// __EMBED_SIGN__(my_classical_func)
    /// ```
    EmbedClassical {
        func: Box<MetaExpr>,
        #[gen_rebuild::skip_recurse]
        embed_kind: EmbedKind,
        dbg: Option<DebugLoc>,
    },

    /// Takes the adjoint of a function value. Example syntax:
    /// ```text
    /// ~f
    /// ```
    Adjoint {
        func: Box<MetaExpr>,
        dbg: Option<DebugLoc>,
    },

    /// Calls a function value. Example syntax for `f(x)`:
    /// ```text
    /// x | f
    /// ```
    Pipe {
        lhs: Box<MetaExpr>,
        rhs: Box<MetaExpr>,
        dbg: Option<DebugLoc>,
    },

    /// A function value that measures its input when called. Example syntax:
    /// ```text
    /// __MEASURE__({'0','1'})
    /// ```
    Measure {
        basis: MetaBasis,
        dbg: Option<DebugLoc>,
    },

    /// A function value that discards its input when called. Example syntax:
    /// ```text
    /// __DISCARD__()
    /// ```
    Discard { dbg: Option<DebugLoc> },

    /// A tensor product of function values or register values. Example
    /// syntax:
    /// ```text
    /// '0' * '1'
    /// ```
    BiTensor {
        left: Box<MetaExpr>,
        right: Box<MetaExpr>,
        dbg: Option<DebugLoc>,
    },

    /// Tilt a qubit state or function value. Example syntax:
    /// ```text
    /// -q
    /// ```
    Tilt {
        val: Box<MetaExpr>,
        #[gen_rebuild::skip_recurse(substitute_basis_alias)]
        angle_deg: FloatExpr,
        dbg: Option<DebugLoc>,
    },

    /// The mighty basis translation. Example syntax:
    /// ```text
    /// pm >> std
    /// ```
    BasisTranslation {
        bin: MetaBasis,
        bout: MetaBasis,
        dbg: Option<DebugLoc>,
    },

    /// A function value that, when called, runs a function value (`then_func`)
    /// in a proper subspace and another function (`else_func`) in the orthogonal
    /// complement of that subspace. Example syntax:
    /// ```text
    /// flip if {'m_'} else id
    /// ```
    Predicated {
        then_func: Box<MetaExpr>,
        else_func: Box<MetaExpr>,
        pred: MetaBasis,
        dbg: Option<DebugLoc>,
    },

    /// A superposition of qubit literals that may not have uniform
    /// probabilities. Example syntax:
    /// ```text
    /// (1/4)*'p' + (3/4)*'m'
    /// ```
    NonUniformSuperpos {
        #[gen_rebuild::skip_recurse(substitute_basis_alias)]
        pairs: Vec<(FloatExpr, MetaVector)>,
        dbg: Option<DebugLoc>,
    },

    /// An ensemble of qubit literals that may not have uniform
    /// probabilities. Example syntax:
    /// ```text
    /// (1/4)*'p' ^ (3/4)*'m'
    /// ```
    /// Another example:
    /// ```text
    /// 'p' ^ 'm'
    /// ```
    Ensemble {
        #[gen_rebuild::skip_recurse(substitute_basis_alias)]
        pairs: Vec<(FloatExpr, MetaVector)>,
        dbg: Option<DebugLoc>,
    },

    /// A classical conditional (ternary) expression. Example syntax:
    /// ```text
    /// flip if meas_result else id
    /// ```
    Conditional {
        then_expr: Box<MetaExpr>,
        else_expr: Box<MetaExpr>,
        cond: Box<MetaExpr>,
        dbg: Option<DebugLoc>,
    },

    /// A qubit literal. Example syntax:
    /// ```text
    /// 'p' + 'm'
    /// ```
    QLit {
        #[gen_rebuild::skip_recurse(substitute_basis_alias)]
        vec: MetaVector,
    },

    /// A classical bit literal. Example syntax:
    /// ```text
    /// bit[4](0b1101)
    /// ```
    BitLiteral {
        val: UBig,
        #[gen_rebuild::skip_recurse(substitute_basis_alias, substitute_vector_alias)]
        n_bits: DimExpr,
        dbg: Option<DebugLoc>,
    },
}

impl MetaExpr {
    /// Returns the debug location for this expression.
    pub fn get_dbg(&self) -> Option<DebugLoc> {
        match self {
            MetaExpr::ExprMacro { dbg, .. }
            | MetaExpr::BasisMacro { dbg, .. }
            | MetaExpr::BroadcastTensor { dbg, .. }
            | MetaExpr::Instantiate { dbg, .. }
            | MetaExpr::Repeat { dbg, .. }
            | MetaExpr::Variable { dbg, .. }
            | MetaExpr::UnitLiteral { dbg }
            | MetaExpr::EmbedClassical { dbg, .. }
            | MetaExpr::Adjoint { dbg, .. }
            | MetaExpr::Pipe { dbg, .. }
            | MetaExpr::Measure { dbg, .. }
            | MetaExpr::Discard { dbg }
            | MetaExpr::BiTensor { dbg, .. }
            | MetaExpr::Tilt { dbg, .. }
            | MetaExpr::BasisTranslation { dbg, .. }
            | MetaExpr::Predicated { dbg, .. }
            | MetaExpr::NonUniformSuperpos { dbg, .. }
            | MetaExpr::Ensemble { dbg, .. }
            | MetaExpr::Conditional { dbg, .. }
            | MetaExpr::BitLiteral { dbg, .. } => dbg.clone(),
            MetaExpr::QLit { vec } => vec.get_dbg(),
        }
    }

    /// Extracts a plain-AST `@qpu` expression from this metaQwerty expression.
    pub fn extract(self) -> Result<ast::qpu::Expr, LowerError> {
        rebuild!(MetaExpr, self, extract)
    }

    pub(crate) fn extract_rewriter(
        rewritten: rewrite_ty!(MetaExpr, extract),
    ) -> Result<ast::qpu::Expr, LowerError> {
        rewrite_match! {MetaExpr, extract, rewritten,
            Variable { name, dbg } => Ok(ast::qpu::Expr::Variable(ast::qpu::Variable { name, dbg })),

            UnitLiteral { dbg } => Ok(ast::qpu::Expr::UnitLiteral(ast::qpu::UnitLiteral { dbg })),

            EmbedClassical { func, embed_kind, dbg } => {
                if let ast::qpu::Expr::Variable(ast::qpu::Variable { name, dbg: _ }) = func {
                    Ok(ast::qpu::Expr::EmbedClassical(ast::qpu::EmbedClassical {
                        func_name: name,
                        embed_kind,
                        dbg,
                    }))
                } else {
                    Err(LowerError {
                        kind: LowerErrorKind::InvalidEmbedOperand,
                        dbg,
                    })
                }
            }

            Adjoint { func, dbg } => {
                Ok(ast::qpu::Expr::Adjoint(ast::qpu::Adjoint {
                    func: Box::new(func),
                    dbg,
                }))
            }

            Pipe { lhs, rhs, dbg } => {
                Ok(ast::qpu::Expr::Pipe(ast::qpu::Pipe {
                    lhs: Box::new(lhs),
                    rhs: Box::new(rhs),
                    dbg,
                }))
            }

            Measure { basis, dbg } => {
                Ok(ast::qpu::Expr::Measure(ast::qpu::Measure {
                    basis,
                    dbg,
                }))
            }

            Discard { dbg } => Ok(ast::qpu::Expr::Discard(ast::qpu::Discard { dbg })),

            BiTensor { left, right, dbg } => {
                Ok(ast::qpu::Expr::Tensor(ast::qpu::Tensor {
                    vals: vec![left, right],
                    dbg,
                }))
            }

            Tilt { val, angle_deg, dbg } => {
                Ok(ast::qpu::Expr::Tilt(ast::qpu::Tilt {
                    val: Box::new(val),
                    angle_deg,
                    dbg,
                }))
            }

            BasisTranslation { bin, bout, dbg } => {
                Ok(ast::qpu::Expr::BasisTranslation(ast::qpu::BasisTranslation {
                    bin,
                    bout,
                    dbg,
                }))
            }

            Predicated { then_func, else_func, pred, dbg } => {
                Ok(ast::qpu::Expr::Predicated(ast::qpu::Predicated {
                    then_func: Box::new(then_func),
                    else_func: Box::new(else_func),
                    pred,
                    dbg,
                }))
            }

            NonUniformSuperpos { pairs, dbg } => {
                let pairs = pairs
                    .into_iter()
                    .map(|(prob, vec)| {
                        let dbg = vec.get_dbg();
                        let qlit = vec.try_into_qubit_literal()
                            .ok_or_else(|| LowerError {
                                kind: LowerErrorKind::IllegalQubitSymbolInQubitLiteral,
                                dbg,
                            })?;
                        Ok((prob, qlit))
                    })
                    .collect::<Result<Vec<_>, LowerError>>()?;
                Ok(ast::qpu::Expr::NonUniformSuperpos(ast::qpu::NonUniformSuperpos {
                    pairs,
                    dbg,
                }))
            }

            Ensemble { pairs, dbg } => {
                let pairs = pairs
                    .into_iter()
                    .map(|(prob, vec)| {
                        let dbg = vec.get_dbg();
                        let qlit = vec.try_into_qubit_literal()
                            .ok_or_else(|| LowerError {
                                kind: LowerErrorKind::IllegalQubitSymbolInQubitLiteral,
                                dbg,
                            })?;
                        Ok((prob, qlit))
                    })
                    .collect::<Result<Vec<_>, LowerError>>()?;
                Ok(ast::qpu::Expr::Ensemble(ast::qpu::Ensemble {
                    pairs,
                    dbg,
                }))
            }

            Conditional { then_expr, else_expr, cond, dbg } => {
                Ok(ast::qpu::Expr::Conditional(ast::qpu::Conditional {
                    then_expr: Box::new(then_expr),
                    else_expr: Box::new(else_expr),
                    cond: Box::new(cond),
                    dbg,
                }))
            }

            QLit { vec } => {
                let dbg = vec.get_dbg();
                let qlit = vec.try_into_qubit_literal()
                    .ok_or_else(|| LowerError {
                        kind: LowerErrorKind::IllegalQubitSymbolInQubitLiteral,
                        dbg: dbg.clone(),
                    })?;
                Ok(ast::qpu::Expr::QLitExpr(ast::qpu::QLitExpr { qlit, dbg }))
            }

            BitLiteral { val, n_bits, dbg } => {
                Ok(ast::qpu::Expr::BitLiteral(ast::qpu::BitLiteral { val, n_bits, dbg }))
            }

            ExprMacro { dbg, .. }
            | BasisMacro { dbg, .. }
            | BroadcastTensor { dbg, .. }
            | Instantiate { dbg, .. }
            | Repeat { dbg, .. } => Err(LowerError {
                kind: LowerErrorKind::NotFullyFolded,
                dbg,
            }),
        }
    }
}

// TODO: don't duplicate with qpu.rs
impl fmt::Display for MetaExpr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        visitor_write! {MetaExpr, self,
            MetaExpr::ExprMacro { name, arg, .. } => write!(f, "({!}).{}", *arg, name),
            MetaExpr::BasisMacro { name, arg, .. } => write!(f, "({}).{}", arg, name),
            MetaExpr::BroadcastTensor { val, factor, .. } => write!(f, "({!})**({})", *val, factor),
            MetaExpr::Instantiate { name, param, .. } => write!(f, "{}[[{}]]", name, param),
            MetaExpr::Repeat { for_each, iter_var, upper_bound, .. } => {
                write!(f, "({!} for {} in range({}))", *for_each, iter_var, upper_bound)
            }
            MetaExpr::Variable { name, .. } => write!(f, "{}", name),
            MetaExpr::UnitLiteral { .. } => write!(f, "[]"),
            MetaExpr::EmbedClassical { func, embed_kind, .. } => {
                write!(f, "({!}).{}", *func, match embed_kind {
                    EmbedKind::Sign => "sign",
                    EmbedKind::Xor => "xor",
                    EmbedKind::InPlace => "inplace",
                })
            }
            MetaExpr::Adjoint { func, .. } => write!(f, "~({!})", *func),
            MetaExpr::Pipe { lhs, rhs, .. } => write!(f, "({!}) | ({!})", *lhs, *rhs),
            MetaExpr::Measure { basis, .. } => write!(f, "({}).measure", basis),
            MetaExpr::Discard { .. } => write!(f, "discard"),
            MetaExpr::BiTensor { left, right, .. } => write!(f, "({!})*({!})", *left, *right),
            MetaExpr::Tilt { val, angle_deg, .. } => write!(f, "({!})*({})", *val, angle_deg),
            MetaExpr::BasisTranslation { bin, bout, .. } => write!(f, "({}) >> ({})", bin, bout),
            MetaExpr::Predicated { then_func, else_func, pred, .. } => {
                write!(f, "({!}) if ({}) else ({!})", *then_func, pred, *else_func)
            }
            MetaExpr::NonUniformSuperpos { pairs, .. } => {
                write!(
                    f,
                    "{}",
                    pairs
                        .iter()
                        .format_with(
                            " + ",
                            |(prob, qlit), f| {
                                f(&format_args!("({})*({})", prob, qlit))
                            }
                        )
                )
            }
            MetaExpr::Ensemble { pairs, .. } => {
                write!(
                    f,
                    "{}",
                    pairs
                        .iter()
                        .format_with(
                            " ^ ",
                            |(prob, qlit), f| {
                                f(&format_args!("({})*({})", prob, qlit))
                            }
                        )
                )
            }
            MetaExpr::Conditional { then_expr, else_expr, cond, .. } => {
                write!(f, "({!}) if ({!}) else ({!})", *then_expr, *cond, *else_expr)
            }
            MetaExpr::QLit { vec } => write!(f, "{}", vec),
            MetaExpr::BitLiteral { val, n_bits, .. } => write!(f, "bit[{}]({})", val, n_bits),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum ExprMacroPattern {
    /// Match an arbitrary expression and bind it to a name. Example syntax:
    /// ```text
    /// f.expr.xor = __EMBED_XOR__(f)
    /// ^
    /// ```
    AnyExpr { name: String, dbg: Option<DebugLoc> },
}

impl fmt::Display for ExprMacroPattern {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ExprMacroPattern::AnyExpr { name, .. } => write!(f, "{}", name),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum BasisMacroPattern {
    /// Match an arbitrary basis and bind it to a name. Example syntax:
    /// ```text
    /// b.measure = __MEASURE__(b)
    /// ^
    /// ```
    AnyBasis { name: String, dbg: Option<DebugLoc> },

    /// Match a basis literal and bind it to a name. Example syntax:
    /// ```text
    /// {bv1, bv2}.flip = {bv1, bv2} >> {bv2, bv1}
    /// ^^^^^^^^^^
    /// ```
    BasisLiteral {
        vec_names: Vec<String>,
        dbg: Option<DebugLoc>,
    },
}

impl fmt::Display for BasisMacroPattern {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BasisMacroPattern::AnyBasis { name, .. } => write!(f, "{}", name),
            BasisMacroPattern::BasisLiteral { vec_names, .. } => {
                write!(f, "{{")?;
                for (i, vec_name) in vec_names.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", vec_name)?;
                }
                write!(f, "}}")
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum RecDefParam {
    Base(IBig),
    Rec(String),
}

impl fmt::Display for RecDefParam {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RecDefParam::Base(constant) => write!(f, "{}", constant),
            RecDefParam::Rec(dim_var_name) => write!(f, "{}", dim_var_name),
        }
    }
}

/// A `@qpu` metaQwerty statement can produce either a `@qpu` plain-Qwerty
/// statement:
/// ```text
/// return '0'   ==>  return __SYM_STD0__()
/// ```
/// ...or an entire `@classical` function in the case of inline
/// `@classical` function declarations:
/// ```text
/// f: cfunc = lambda x: ~x    ==>   @classical
///                                  def f(x: bit) -> bit:
///                                      return ~x
/// ```
#[derive(Debug, Clone)]
pub enum LoweredStmt {
    QpuStmt(ast::Stmt<ast::qpu::Expr>),
    ClassicalFunc(ast::FunctionDef<ast::classical::Expr>),
}

#[gen_rebuild {
    substitute_dim_var(
        more_copied_args(dim_var: &DimVar, new_dim_expr: &DimExpr),
        recurse_attrs,
    ),
    expand(
        rewrite(expand_rewriter),
        progress(Progress),
        more_copied_args(env: &mut MacroEnv),
        result_err(LowerError),
        recurse_attrs,
    ),
    extract(
        rewrite(extract_rewriter),
        rewrite_to(
            MetaStmt => LoweredStmt,
            MetaExpr => ast::qpu::Expr,
            classical::MetaExpr => ast::classical::Expr,
            DimExpr => usize,
        ),
        result_err(LowerError),
        recurse_attrs,
    ),
}]
#[derive(Debug, Clone, PartialEq)]
pub enum MetaStmt {
    /// A macro definition that takes an expression argument and expands to an
    /// expression. Example syntax:
    /// ```text
    /// f.expr.xor = __EMBED_XOR__(f)
    /// ```
    ExprMacroDef {
        #[gen_rebuild::skip_recurse]
        lhs_pat: ExprMacroPattern,
        lhs_name: String,
        #[gen_rebuild::skip_recurse(expand, extract)]
        rhs: MetaExpr,
        dbg: Option<DebugLoc>,
    },

    /// A macro definition that takes a basis argument and expands to an
    /// expression. Example syntax:
    /// ```text
    /// {bv1, bv2}.flip = {bv1, bv2} >> {bv2, bv1}
    /// ```
    /// Another example:
    /// ```text
    /// b.measure = __MEASURE__(b)
    /// ```
    BasisMacroDef {
        #[gen_rebuild::skip_recurse]
        lhs_pat: BasisMacroPattern,
        lhs_name: String,
        #[gen_rebuild::skip_recurse(expand, extract)]
        rhs: MetaExpr,
        dbg: Option<DebugLoc>,
    },

    /// A macro definition with a basis argument that expands to a basis
    /// generator. Example syntax:
    /// ```text
    /// {bv1, bv2}.revolve = __REVOLVE__(bv1, bv2)
    /// ```
    BasisGeneratorMacroDef {
        #[gen_rebuild::skip_recurse]
        lhs_pat: BasisMacroPattern,
        lhs_name: String,
        #[gen_rebuild::skip_recurse(expand, extract)]
        rhs: MetaBasisGenerator,
        dbg: Option<DebugLoc>,
    },

    /// A vector symbol defintion. Example syntax:
    /// ```text
    /// 'p'.sym = '0'+'1'
    /// ```
    VectorSymbolDef {
        lhs: char,
        #[gen_rebuild::skip_recurse(expand, extract)]
        rhs: MetaVector,
        dbg: Option<DebugLoc>,
    },

    /// A basis alias definition. Example syntax:
    /// ```text
    /// std = {'0','1'}
    /// ```
    BasisAliasDef {
        lhs: String,
        #[gen_rebuild::skip_recurse(expand, extract)]
        rhs: MetaBasis,
        dbg: Option<DebugLoc>,
    },

    /// A recursive basis alias definition. Example syntax:
    /// ```text
    /// fourier[N] = fourier[N-1] // std.revolve
    /// ```
    BasisAliasRecDef {
        lhs: String,
        #[gen_rebuild::skip_recurse]
        param: RecDefParam,
        #[gen_rebuild::skip_recurse(expand, extract)]
        rhs: MetaBasis,
        dbg: Option<DebugLoc>,
    },

    /// An inline classical function definition. Example syntax:
    /// ```text
    /// f: cfunc[N,1] = lambda x: (x & secret_string).xor_reduce()
    /// ```
    ClassicalLambdaDef {
        name: String,
        arg_name: String,
        in_dim: DimExpr,
        out_dim: DimExpr,
        body: classical::MetaExpr,
        dbg: Option<DebugLoc>,
    },

    /// An expression statement. Example syntax:
    /// ```text
    /// f(x)
    /// ```
    Expr { expr: MetaExpr },

    /// An assignment statement. Example syntax:
    /// ```text
    /// q = '0'
    /// ```
    Assign {
        lhs: String,
        rhs: MetaExpr,
        dbg: Option<DebugLoc>,
    },

    /// A register-unpacking assignment statement. Example syntax:
    /// ```text
    /// q1, q2 = '01'
    /// ```
    UnpackAssign {
        lhs: Vec<String>,
        rhs: MetaExpr,
        dbg: Option<DebugLoc>,
    },

    /// A return statement. Example syntax:
    /// ```text
    /// return q
    /// ```
    Return {
        val: MetaExpr,
        dbg: Option<DebugLoc>,
    },
}

impl MetaStmt {
    /// Extracts a plain-AST `@qpu` statement from this metaQwerty statement.
    pub fn extract(self) -> Result<LoweredStmt, LowerError> {
        rebuild!(MetaStmt, self, extract)
    }

    pub(crate) fn extract_rewriter(
        rewritten: rewrite_ty!(MetaStmt, extract),
    ) -> Result<LoweredStmt, LowerError> {
        Ok(rewrite_match! {MetaStmt, extract, rewritten,
            // TODO: don't duplicate this part with classical.rs
            Expr { expr } => {
                let dbg = expr.get_dbg();
                LoweredStmt::QpuStmt(ast::Stmt::Expr(ast::StmtExpr { expr, dbg }))
            }
            Assign { lhs, rhs, dbg } => {
                LoweredStmt::QpuStmt(ast::Stmt::Assign(ast::Assign { lhs, rhs, dbg }))
            }
            UnpackAssign { lhs, rhs, dbg } => {
                LoweredStmt::QpuStmt(ast::Stmt::UnpackAssign(ast::UnpackAssign { lhs, rhs, dbg }))
            }
            Return { val, dbg } => {
                LoweredStmt::QpuStmt(ast::Stmt::Return(ast::Return { val, dbg }))
            }

            ClassicalLambdaDef { name, arg_name, in_dim, out_dim, body, dbg } => {
                let arg_type = ast::Type::RegType { elem_ty: RegKind::Bit, dim: in_dim };
                let ret_type = ast::Type::RegType { elem_ty: RegKind::Bit, dim: out_dim };

                LoweredStmt::ClassicalFunc(ast::FunctionDef {
                    name,
                    args: vec![(arg_type, arg_name)],
                    ret_type,
                    body: vec![ast::Stmt::Return(ast::Return { val: body, dbg: dbg.clone(), })],
                    is_rev: false,
                    dbg,
                })
            }

            // Definition of macros does not present any problem in conversion;
            // it is using them that is an issue.
            ExprMacroDef { dbg, .. }
            | BasisMacroDef { dbg, .. }
            | BasisGeneratorMacroDef { dbg, .. }
            | VectorSymbolDef { dbg, .. }
            | BasisAliasDef { dbg, .. }
            | BasisAliasRecDef { dbg, .. } => LoweredStmt::QpuStmt(ast::Stmt::trivial(dbg)),
        })
    }
}

impl MetaStmt {
    /// Returns a trivial statement. Replacing a statement with this trivial
    /// statement effectively deletes it.
    pub fn trivial(dbg: Option<DebugLoc>) -> MetaStmt {
        MetaStmt::Expr {
            expr: MetaExpr::UnitLiteral { dbg },
        }
    }
}

// TODO: don't duplicate with ast.rs
impl fmt::Display for MetaStmt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MetaStmt::ExprMacroDef {
                lhs_pat,
                lhs_name,
                rhs,
                ..
            } => write!(f, "{}.expr.{} = {}", lhs_pat, lhs_name, rhs),
            MetaStmt::BasisMacroDef {
                lhs_pat,
                lhs_name,
                rhs,
                ..
            } => write!(f, "{}.{} = {}", lhs_pat, lhs_name, rhs),
            MetaStmt::BasisGeneratorMacroDef {
                lhs_pat,
                lhs_name,
                rhs,
                ..
            } => write!(f, "{}.{} = {}", lhs_pat, lhs_name, rhs),
            MetaStmt::VectorSymbolDef { lhs, rhs, .. } => write!(f, "'{}'.sym = {}", lhs, rhs),
            MetaStmt::BasisAliasDef { lhs, rhs, .. } => write!(f, "{} = {}", lhs, rhs),
            MetaStmt::BasisAliasRecDef {
                lhs, param, rhs, ..
            } => write!(f, "{}[{}] = {}", lhs, param, rhs),
            MetaStmt::ClassicalLambdaDef {
                name,
                arg_name,
                in_dim,
                out_dim,
                body,
                ..
            } => {
                write!(
                    f,
                    "{}: cfunc[{},{}] = lambda {}: {}",
                    name, in_dim, out_dim, arg_name, body
                )
            }
            MetaStmt::Expr { expr, .. } => write!(f, "{}", expr),
            MetaStmt::Assign { lhs, rhs, .. } => write!(f, "{} = {}", lhs, rhs),
            MetaStmt::UnpackAssign { lhs, rhs, .. } => {
                for (i, name) in lhs.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", name)?;
                }
                write!(f, " = {}", rhs)
            }
            MetaStmt::Return { val, .. } => write!(f, "return {}", val),
        }
    }
}

#[cfg(test)]
mod test_display {
    use super::{FloatExpr, MetaVector};

    #[test]
    fn test_meta_vec_to_string_zero() {
        let vec = MetaVector::ZeroVector { dbg: None };
        assert_eq!(vec.to_string(), "'0'");
    }

    #[test]
    fn test_meta_vec_to_string_one() {
        let vec = MetaVector::OneVector { dbg: None };
        assert_eq!(vec.to_string(), "'1'");
    }

    #[test]
    fn test_meta_vec_to_string_pad() {
        let vec = MetaVector::PadVector { dbg: None };
        assert_eq!(vec.to_string(), "'?'");
    }

    #[test]
    fn test_meta_vec_to_string_tgt() {
        let vec = MetaVector::TargetVector { dbg: None };
        assert_eq!(vec.to_string(), "'_'");
    }

    #[test]
    fn test_meta_vec_to_string_tilt_180() {
        let vec = MetaVector::VectorTilt {
            q: Box::new(MetaVector::OneVector { dbg: None }),
            angle_deg: FloatExpr::FloatConst {
                val: 180.0,
                dbg: None,
            },
            dbg: None,
        };
        assert_eq!(vec.to_string(), "('1')@(180)");
    }

    #[test]
    fn test_meta_vec_to_string_tilt_non_180() {
        let vec = MetaVector::VectorTilt {
            q: Box::new(MetaVector::OneVector { dbg: None }),
            angle_deg: FloatExpr::FloatConst {
                val: 1.23456,
                dbg: None,
            },
            dbg: None,
        };
        assert_eq!(vec.to_string(), "('1')@(1.23456)");
    }

    #[test]
    fn test_meta_vec_to_string_superpos() {
        let vec = MetaVector::UniformVectorSuperpos {
            q1: Box::new(MetaVector::ZeroVector { dbg: None }),
            q2: Box::new(MetaVector::OneVector { dbg: None }),
            dbg: None,
        };
        assert_eq!(vec.to_string(), "('0') + ('1')");
    }

    #[test]
    fn test_meta_vec_to_string_tensor_01() {
        let vec = MetaVector::VectorBiTensor {
            left: Box::new(MetaVector::ZeroVector { dbg: None }),
            right: Box::new(MetaVector::OneVector { dbg: None }),
            dbg: None,
        };
        assert_eq!(vec.to_string(), "('0') * ('1')");
    }

    #[test]
    fn test_meta_vec_to_string_tensor_01pad() {
        let vec = MetaVector::VectorBiTensor {
            left: Box::new(MetaVector::ZeroVector { dbg: None }),
            right: Box::new(MetaVector::VectorBiTensor {
                left: Box::new(MetaVector::OneVector { dbg: None }),
                right: Box::new(MetaVector::PadVector { dbg: None }),
                dbg: None,
            }),
            dbg: None,
        };
        assert_eq!(vec.to_string(), "('0') * (('1') * ('?'))");
    }

    #[test]
    fn test_meta_vec_to_string_vector_unit() {
        let vec = MetaVector::VectorUnit { dbg: None };
        assert_eq!(vec.to_string(), "''");
    }
}
