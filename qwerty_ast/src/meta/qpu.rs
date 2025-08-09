use crate::{
    ast::{self, qpu::EmbedKind},
    dbg::DebugLoc,
    error::{ExtractError, ExtractErrorKind},
    meta::DimExpr,
};
use dashu::integer::UBig;
use std::fmt;

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
    pub fn extract(&self) -> Result<f64, ExtractError> {
        match self {
            FloatExpr::FloatConst { val, .. } => Ok(*val),
            FloatExpr::FloatDimExpr { dbg, .. }
            | FloatExpr::FloatSum { dbg, .. }
            | FloatExpr::FloatProd { dbg, .. }
            | FloatExpr::FloatDiv { dbg, .. }
            | FloatExpr::FloatNeg { dbg, .. } => Err(ExtractError {
                kind: ExtractErrorKind::NotFullyFolded,
                dbg: dbg.clone(),
            }),
        }
    }
}

impl fmt::Display for FloatExpr {
    /// Returns a representation of a dimension variable expression that
    /// matches the syntax in the Python DSL.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FloatExpr::FloatDimExpr { expr, .. } => write!(f, "{}", expr),
            FloatExpr::FloatConst { val, .. } => write!(f, "{}", val),
            FloatExpr::FloatSum { left, right, .. } => write!(f, "({})+({})", left, right),
            FloatExpr::FloatProd { left, right, .. } => write!(f, "({})*({})", left, right),
            FloatExpr::FloatDiv { left, right, .. } => write!(f, "({})/({})", left, right),
            FloatExpr::FloatNeg { val, .. } => write!(f, "-({})", val),
        }
    }
}

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
        val: Box<MetaVector>,
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
    pub fn extract(&self) -> Result<ast::qpu::Vector, ExtractError> {
        match self {
            MetaVector::ZeroVector { dbg } => Ok(ast::qpu::Vector::ZeroVector { dbg: dbg.clone() }),
            MetaVector::OneVector { dbg } => Ok(ast::qpu::Vector::OneVector { dbg: dbg.clone() }),
            MetaVector::PadVector { dbg } => Ok(ast::qpu::Vector::PadVector { dbg: dbg.clone() }),
            MetaVector::TargetVector { dbg } => {
                Ok(ast::qpu::Vector::TargetVector { dbg: dbg.clone() })
            }
            MetaVector::VectorTilt { q, angle_deg, dbg } => q.extract().and_then(|ast_q| {
                angle_deg
                    .extract()
                    .map(|angle_deg_double| ast::qpu::Vector::VectorTilt {
                        q: Box::new(ast_q),
                        angle_deg: angle_deg_double,
                        dbg: dbg.clone(),
                    })
            }),
            MetaVector::UniformVectorSuperpos { q1, q2, dbg } => q1.extract().and_then(|ast_q1| {
                q2.extract()
                    .map(|ast_q2| ast::qpu::Vector::UniformVectorSuperpos {
                        q1: Box::new(ast_q1),
                        q2: Box::new(ast_q2),
                        dbg: dbg.clone(),
                    })
            }),
            MetaVector::VectorBiTensor { left, right, dbg } => {
                left.extract().and_then(|ast_left| {
                    right
                        .extract()
                        .map(|ast_right| ast::qpu::Vector::VectorTensor {
                            qs: vec![ast_left, ast_right],
                            dbg: dbg.clone(),
                        })
                })
            }
            MetaVector::VectorUnit { dbg } => Ok(ast::qpu::Vector::VectorUnit { dbg: dbg.clone() }),

            MetaVector::VectorAlias { dbg, .. }
            | MetaVector::VectorSymbol { dbg, .. }
            | MetaVector::VectorBroadcastTensor { dbg, .. } => Err(ExtractError {
                kind: ExtractErrorKind::NotFullyFolded,
                dbg: dbg.clone(),
            }),
        }
    }
}

// TODO: don't duplicate with qpu.rs
impl fmt::Display for MetaVector {
    /// Represents a vector in a human-readable form for error messages sent
    /// back to the programmer.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MetaVector::VectorAlias { name, .. } => write!(f, "{}", name),
            MetaVector::VectorSymbol { sym, .. } => write!(f, "'{}'", sym),
            MetaVector::VectorBroadcastTensor { val, factor, .. } => {
                write!(f, "({})*({})", *val, factor)
            }
            MetaVector::ZeroVector { .. } => write!(f, "'0'"),
            MetaVector::OneVector { .. } => write!(f, "'1'"),
            MetaVector::PadVector { .. } => write!(f, "'?'"),
            MetaVector::TargetVector { .. } => write!(f, "'_'"),
            MetaVector::VectorTilt { q, angle_deg, .. } => {
                write!(f, "({})@({})", **q, *angle_deg)
            }
            MetaVector::UniformVectorSuperpos { q1, q2, .. } => {
                write!(f, "({}) + ({})", **q1, **q2)
            }
            MetaVector::VectorBiTensor { left, right, .. } => {
                write!(f, "({}) * ({})", **left, **right)
            }
            MetaVector::VectorUnit { .. } => write!(f, "''"),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum MetaBasisGenerator {
    /// Invokes a macro. Example syntax:
    /// ```text
    /// {'0','1'}.revolve
    /// ```
    BasisGeneratorMacro {
        name: String,
        arg: Box<MetaBasis>,
        dbg: Option<DebugLoc>,
    },

    /// A revolve generator, used to define the Fourier basis. Example
    /// syntax:
    /// ```text
    /// __REVOLVE__('p', 'm')
    /// ```
    Revolve {
        v1: MetaVector,
        v2: MetaVector,
        dbg: Option<DebugLoc>,
    },
}

impl MetaBasisGenerator {
    /// Extracts a plain-AST `@qpu` BasisGenerator from this metaQwerty basis
    /// generator. Returns an error instead if metaQwerty constructs are still
    /// present.
    pub fn extract(&self) -> Result<ast::qpu::BasisGenerator, ExtractError> {
        match self {
            MetaBasisGenerator::Revolve { v1, v2, dbg } => v1.extract().and_then(|ast_v1| {
                v2.extract()
                    .map(|ast_v2| ast::qpu::BasisGenerator::Revolve {
                        v1: ast_v1,
                        v2: ast_v2,
                        dbg: dbg.clone(),
                    })
            }),

            MetaBasisGenerator::BasisGeneratorMacro { dbg, .. } => Err(ExtractError {
                kind: ExtractErrorKind::NotFullyFolded,
                dbg: dbg.clone(),
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

#[derive(Debug, Clone, PartialEq)]
pub enum MetaBasis {
    /// A basis alias name. Example syntax:
    /// ```text
    /// pm
    /// ```
    BasisAlias { name: String, dbg: Option<DebugLoc> },

    /// An n-fold tensor product of a basis. Example syntax:
    /// ```text
    /// pm**N
    /// ```
    BasisBroadcastTensor {
        val: Box<MetaBasis>,
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
        gen: MetaBasisGenerator,
        dbg: Option<DebugLoc>,
    },
}

impl MetaBasis {
    /// Extracts a plain-AST `@qpu` Basis from this metaQwerty basis.
    pub fn extract(&self) -> Result<ast::qpu::Basis, ExtractError> {
        match self {
            MetaBasis::BasisLiteral { vecs, dbg } => vecs
                .iter()
                .map(MetaVector::extract)
                .collect::<Result<Vec<ast::qpu::Vector>, ExtractError>>()
                .map(|ast_vecs| ast::qpu::Basis::BasisLiteral {
                    vecs: ast_vecs,
                    dbg: dbg.clone(),
                }),
            MetaBasis::EmptyBasisLiteral { dbg } => {
                Ok(ast::qpu::Basis::EmptyBasisLiteral { dbg: dbg.clone() })
            }
            MetaBasis::BasisBiTensor { left, right, dbg } => left.extract().and_then(|ast_left| {
                right
                    .extract()
                    .map(|ast_right| ast::qpu::Basis::BasisTensor {
                        bases: vec![ast_left, ast_right],
                        dbg: dbg.clone(),
                    })
            }),
            MetaBasis::ApplyBasisGenerator { basis, gen, dbg } => {
                basis.extract().and_then(|ast_basis| {
                    gen.extract()
                        .map(|ast_gen| ast::qpu::Basis::ApplyBasisGenerator {
                            basis: Box::new(ast_basis),
                            gen: ast_gen,
                            dbg: dbg.clone(),
                        })
                })
            }

            MetaBasis::BasisAlias { dbg, .. } | MetaBasis::BasisBroadcastTensor { dbg, .. } => {
                Err(ExtractError {
                    kind: ExtractErrorKind::NotFullyFolded,
                    dbg: dbg.clone(),
                })
            }
        }
    }
}

impl fmt::Display for MetaBasis {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MetaBasis::BasisAlias { name, .. } => write!(f, "{}", name),
            MetaBasis::BasisBroadcastTensor { val, factor, .. } => {
                write!(f, "({})**({})", val, factor)
            }
            MetaBasis::BasisLiteral { vecs, .. } => {
                write!(f, "{{")?;
                for (i, vec) in vecs.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", vec)?;
                }
                Ok(())
            }
            MetaBasis::EmptyBasisLiteral { .. } => write!(f, "{{}}"),
            MetaBasis::BasisBiTensor { left, right, .. } => {
                write!(f, "({})*({})", *left, *right)
            }
            MetaBasis::ApplyBasisGenerator { basis, gen, .. } => {
                write!(f, "({}) // ({})", *basis, gen)
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum MetaExpr {
    /// Invokes a macro. Example syntax:
    /// ```text
    /// {'0','1'}.measure
    /// ```
    /// Another example:
    /// ```text
    /// my_classical_func.inplace
    /// ```
    Macro {
        name: String,
        arg: Box<MetaBasis>,
        dbg: Option<DebugLoc>,
    },

    /// An n-fold tensor product. Example syntax:
    /// ```text
    /// id**N
    /// ```
    BroadcastTensor {
        val: Box<MetaExpr>,
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
        param: DimExpr,
        dbg: Option<DebugLoc>,
    },

    /// A macro that expands into a pipeline. Example syntax:
    /// ```text
    /// (op[[i] for i in range(N))
    /// ```
    Repeat {
        for_each: Box<MetaExpr>,
        iter_var: String,
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
    QLit { vec: MetaVector },

    /// A classical bit literal. Example syntax:
    /// ```text
    /// bit[4](0b1101)
    /// ```
    BitLiteral {
        val: UBig,
        n_bits: DimExpr,
        dbg: Option<DebugLoc>,
    },
}

impl MetaExpr {
    /// Returns the debug location for this expression.
    pub fn get_dbg(&self) -> Option<DebugLoc> {
        match self {
            MetaExpr::Macro { dbg, .. }
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
            | MetaExpr::BasisTranslation { dbg, .. }
            | MetaExpr::Predicated { dbg, .. }
            | MetaExpr::NonUniformSuperpos { dbg, .. }
            | MetaExpr::Conditional { dbg, .. }
            | MetaExpr::BitLiteral { dbg, .. } => dbg.clone(),
            MetaExpr::QLit { vec } => vec.get_dbg(),
        }
    }

    /// Extracts a plain-AST `@qpu` expression from this metaQwerty expression.
    pub fn extract(&self) -> Result<ast::qpu::Expr, ExtractError> {
        match self {
            MetaExpr::Variable { name, dbg } => Ok(ast::qpu::Expr::Variable(ast::Variable {
                name: name.to_string(),
                dbg: dbg.clone(),
            })),
            MetaExpr::UnitLiteral { dbg } => {
                Ok(ast::qpu::Expr::UnitLiteral(ast::qpu::UnitLiteral {
                    dbg: dbg.clone(),
                }))
            }
            MetaExpr::EmbedClassical {
                func,
                embed_kind,
                dbg,
            } => {
                if let MetaExpr::Variable { name, dbg: _ } = &**func {
                    Ok(ast::qpu::Expr::EmbedClassical(ast::qpu::EmbedClassical {
                        func_name: name.to_string(),
                        embed_kind: *embed_kind,
                        dbg: dbg.clone(),
                    }))
                } else {
                    Err(ExtractError {
                        kind: ExtractErrorKind::Malformed,
                        dbg: dbg.clone(),
                    })
                }
            }
            MetaExpr::Adjoint { func, dbg } => func.extract().map(|ast_func| {
                ast::qpu::Expr::Adjoint(ast::qpu::Adjoint {
                    func: Box::new(ast_func),
                    dbg: dbg.clone(),
                })
            }),
            MetaExpr::Pipe { lhs, rhs, dbg } => lhs.extract().and_then(|ast_lhs| {
                rhs.extract().map(|ast_rhs| {
                    ast::qpu::Expr::Pipe(ast::qpu::Pipe {
                        lhs: Box::new(ast_lhs),
                        rhs: Box::new(ast_rhs),
                        dbg: dbg.clone(),
                    })
                })
            }),
            MetaExpr::Measure { basis, dbg } => basis.extract().map(|ast_basis| {
                ast::qpu::Expr::Measure(ast::qpu::Measure {
                    basis: ast_basis,
                    dbg: dbg.clone(),
                })
            }),
            MetaExpr::Discard { dbg } => Ok(ast::qpu::Expr::Discard(ast::qpu::Discard {
                dbg: dbg.clone(),
            })),
            MetaExpr::BiTensor { left, right, dbg } => left.extract().and_then(|ast_left| {
                right.extract().map(|ast_right| {
                    ast::qpu::Expr::Tensor(ast::qpu::Tensor {
                        vals: vec![ast_left, ast_right],
                        dbg: dbg.clone(),
                    })
                })
            }),
            MetaExpr::BasisTranslation { bin, bout, dbg } => bin.extract().and_then(|ast_bin| {
                bout.extract().map(|ast_bout| {
                    ast::qpu::Expr::BasisTranslation(ast::qpu::BasisTranslation {
                        bin: ast_bin,
                        bout: ast_bout,
                        dbg: dbg.clone(),
                    })
                })
            }),
            MetaExpr::Predicated {
                then_func,
                else_func,
                pred,
                dbg,
            } => then_func.extract().and_then(|ast_then| {
                else_func.extract().and_then(|ast_else| {
                    pred.extract().map(|ast_pred| {
                        ast::qpu::Expr::Predicated(ast::qpu::Predicated {
                            then_func: Box::new(ast_then),
                            else_func: Box::new(ast_else),
                            pred: ast_pred,
                            dbg: dbg.clone(),
                        })
                    })
                })
            }),
            MetaExpr::NonUniformSuperpos { pairs, dbg } => pairs
                .iter()
                .map(|(prob, vec)| {
                    prob.extract().and_then(|prob_double| {
                        vec.extract().and_then(|ast_vec| {
                            ast_vec
                                .convert_to_qubit_literal()
                                .ok_or_else(|| ExtractError {
                                    kind: ExtractErrorKind::Malformed,
                                    dbg: ast_vec.get_dbg(),
                                })
                                .map(|ast_qlit| (prob_double, ast_qlit))
                        })
                    })
                })
                .collect::<Result<Vec<(f64, ast::qpu::QLit)>, ExtractError>>()
                .map(|ast_pairs| {
                    ast::qpu::Expr::NonUniformSuperpos(ast::qpu::NonUniformSuperpos {
                        pairs: ast_pairs,
                        dbg: dbg.clone(),
                    })
                }),
            MetaExpr::Conditional {
                then_expr,
                else_expr,
                cond,
                dbg,
            } => then_expr.extract().and_then(|ast_then| {
                else_expr.extract().and_then(|ast_else| {
                    cond.extract().map(|ast_cond| {
                        ast::qpu::Expr::Conditional(ast::qpu::Conditional {
                            then_expr: Box::new(ast_then),
                            else_expr: Box::new(ast_else),
                            cond: Box::new(ast_cond),
                            dbg: dbg.clone(),
                        })
                    })
                })
            }),
            MetaExpr::QLit { vec } => vec.extract().and_then(|ast_vec| {
                ast_vec
                    .convert_to_qubit_literal()
                    .ok_or_else(|| ExtractError {
                        kind: ExtractErrorKind::Malformed,
                        dbg: ast_vec.get_dbg(),
                    })
                    .map(|ast_qlit| ast::qpu::Expr::QLit(ast_qlit))
            }),
            MetaExpr::BitLiteral { val, n_bits, dbg } => n_bits.extract().map(|n_bits_int| {
                ast::qpu::Expr::BitLiteral(ast::BitLiteral {
                    val: val.clone(),
                    n_bits: n_bits_int,
                    dbg: dbg.clone(),
                })
            }),

            MetaExpr::Macro { dbg, .. }
            | MetaExpr::BroadcastTensor { dbg, .. }
            | MetaExpr::Instantiate { dbg, .. }
            | MetaExpr::Repeat { dbg, .. } => Err(ExtractError {
                kind: ExtractErrorKind::NotFullyFolded,
                dbg: dbg.clone(),
            }),
        }
    }
}

// TODO: don't duplicate with qpu.rs
impl fmt::Display for MetaExpr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MetaExpr::Macro { name, arg, .. } => write!(f, "({}).{}", *arg, name),
            MetaExpr::BroadcastTensor { val, factor, .. } => {
                write!(f, "({})**({})", *val, factor)
            }
            MetaExpr::Instantiate { name, param, .. } => write!(f, "{}[[{}]]", name, param),
            MetaExpr::Repeat {
                for_each,
                iter_var,
                upper_bound,
                ..
            } => write!(
                f,
                "({} for {} in range({}))",
                *for_each, iter_var, upper_bound
            ),
            MetaExpr::Variable { name, .. } => write!(f, "{}", name),
            MetaExpr::UnitLiteral { .. } => write!(f, "[]"),
            MetaExpr::EmbedClassical {
                func, embed_kind, ..
            } => {
                let embed_kind_str = match embed_kind {
                    EmbedKind::Sign => "sign",
                    EmbedKind::Xor => "xor",
                    EmbedKind::InPlace => "inplace",
                };
                write!(f, "({}).{}", func, embed_kind_str)
            }
            MetaExpr::Adjoint { func, .. } => write!(f, "~({})", *func),
            MetaExpr::Pipe { lhs, rhs, .. } => write!(f, "({}) | ({})", *lhs, *rhs),
            MetaExpr::Measure { basis, .. } => write!(f, "({}).measure", basis),
            MetaExpr::Discard { .. } => write!(f, "discard"),
            MetaExpr::BiTensor { left, right, .. } => write!(f, "({})*({})", *left, *right),
            MetaExpr::BasisTranslation { bin, bout, .. } => {
                write!(f, "({}) >> ({})", bin, bout)
            }
            MetaExpr::Predicated {
                then_func,
                else_func,
                pred,
                ..
            } => write!(f, "({}) if ({}) else ({})", then_func, pred, else_func),
            MetaExpr::NonUniformSuperpos { pairs, .. } => {
                for (i, (prob, qlit)) in pairs.iter().enumerate() {
                    if i > 0 {
                        write!(f, " + ")?;
                    }
                    write!(f, "({})*({})", prob, qlit)?;
                }
                Ok(())
            }
            MetaExpr::Conditional {
                then_expr,
                else_expr,
                cond,
                ..
            } => write!(f, "({}) if ({}) else ({})", then_expr, cond, else_expr),
            MetaExpr::QLit { vec } => write!(f, "{}", vec),
            MetaExpr::BitLiteral { val, n_bits, .. } => write!(f, "bit[{}]({})", val, n_bits),
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
pub enum MetaStmt {
    /// A macro definition that expands to an expression. Example syntax:
    /// ```text
    /// {bv1, bv2}.flip = {bv1, bv2} >> {bv2, bv1}
    /// ```
    /// Another example:
    /// ```text
    /// f.sign = __EMBED_SIGN__(f)
    /// ```
    MacroDef {
        lhs_pat: BasisMacroPattern,
        lhs_name: String,
        rhs: MetaExpr,
        dbg: Option<DebugLoc>,
    },

    /// A macro definition that expands to a basis generator. Example syntax:
    /// ```text
    /// {bv1, bv2}.revolve = __REVOLVE__(bv1, bv2)
    /// ```
    BasisGeneratorMacroDef {
        lhs_pat: BasisMacroPattern,
        lhs_name: String,
        rhs: MetaBasisGenerator,
        dbg: Option<DebugLoc>,
    },

    /// A vector symbol defintion. Example syntax:
    /// ```text
    /// 'p'.sym = '0'+'1'
    /// ```
    VectorSymbolDef {
        lhs: char,
        rhs: MetaVector,
        dbg: Option<DebugLoc>,
    },

    /// A basis alias definition. Example syntax:
    /// ```text
    /// std = {'0','1'}
    /// ```
    BasisAliasDef {
        lhs: String,
        rhs: MetaBasis,
        dbg: Option<DebugLoc>,
    },

    /// A recursive basis alias definition. Example syntax:
    /// ```text
    /// fourier[N] = fourier[N-1] // std.revolve
    /// ```
    BasisAliasRecDef {
        lhs: String,
        param: DimExpr,
        rhs: MetaBasis,
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
    // TODO: don't duplicate with classical.rs
    /// Extracts a plain-AST `@qpu` statement from this metaQwerty statement.
    pub fn extract(&self) -> Result<ast::Stmt<ast::qpu::Expr>, ExtractError> {
        match self {
            MetaStmt::Expr { expr } => expr.extract().map(|ast_expr| {
                ast::Stmt::Expr(ast::StmtExpr {
                    expr: ast_expr,
                    dbg: expr.get_dbg(),
                })
            }),
            MetaStmt::Assign { lhs, rhs, dbg } => rhs.extract().map(|ast_rhs| {
                ast::Stmt::Assign(ast::Assign {
                    lhs: lhs.to_string(),
                    rhs: ast_rhs,
                    dbg: dbg.clone(),
                })
            }),
            MetaStmt::UnpackAssign { lhs, rhs, dbg } => rhs.extract().map(|ast_rhs| {
                ast::Stmt::UnpackAssign(ast::UnpackAssign {
                    lhs: lhs.clone(),
                    rhs: ast_rhs,
                    dbg: dbg.clone(),
                })
            }),
            MetaStmt::Return { val, dbg } => val.extract().map(|ast_val| {
                ast::Stmt::Return(ast::Return {
                    val: ast_val,
                    dbg: dbg.clone(),
                })
            }),

            MetaStmt::MacroDef { dbg, .. }
            | MetaStmt::BasisGeneratorMacroDef { dbg, .. }
            | MetaStmt::VectorSymbolDef { dbg, .. }
            | MetaStmt::BasisAliasDef { dbg, .. }
            | MetaStmt::BasisAliasRecDef { dbg, .. } => Err(ExtractError {
                kind: ExtractErrorKind::NotFullyFolded,
                dbg: dbg.clone(),
            }),
        }
    }
}

// TODO: don't duplicate with ast.rs
impl fmt::Display for MetaStmt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MetaStmt::MacroDef {
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

/// A list of statements that are prepended to every `@qpu` kernel.
///
/// Example syntax:
/// ```text
/// @qpu_prelude
/// def example_prelude():
///     '0'.sym = __SYM_STD0__()
///     '1'.sym = __SYM_STD1__()
///     flip = {'0','1'} >> {'1','0'}
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct Prelude {
    pub body: Vec<MetaStmt>,
}
