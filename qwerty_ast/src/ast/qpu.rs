//! Expressions and bases for `@qpu` kernels.

use super::{
    Canonicalizable, ToPythonCode, Trivializable, angle_approx_total_cmp, angle_is_approx_zero,
    angles_are_approx_equal, canon_angle, equals_2_to_the_n,
};
use crate::dbg::DebugLoc;
use dashu::integer::UBig;
use itertools::Itertools;
use qwerty_ast_macros::{
    gen_rebuild, gen_rebuild_structs, rebuild, rewrite_match, rewrite_ty, visitor_write,
};
use std::cmp::Ordering;
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmbedKind {
    Sign,
    Xor,
    InPlace,
}

gen_rebuild_structs! {
    configs {
        canonicalize(
            rewrite(canonicalize_rewriter),
            recurse_attrs,
        ),
    }

    defs {
        /// See [`Expr::Variable`].
        #[derive(Debug, Clone, PartialEq)]
        pub struct Variable {
            pub name: String,
            pub dbg: Option<DebugLoc>,
        }

        /// See [`Expr::UnitLiteral`].
        #[derive(Debug, Clone, PartialEq)]
        pub struct UnitLiteral {
            pub dbg: Option<DebugLoc>,
        }

        /// See [`Expr::EmbedClassical`].
        #[derive(Debug, Clone, PartialEq)]
        pub struct EmbedClassical {
            pub func_name: String,
            #[gen_rebuild::skip_recurse(canonicalize)]
            pub embed_kind: EmbedKind,
            pub dbg: Option<DebugLoc>,
        }

        /// See [`Expr::Adjoint`].
        #[derive(Debug, Clone, PartialEq)]
        pub struct Adjoint {
            pub func: Box<Expr>,
            pub dbg: Option<DebugLoc>,
        }

        /// See [`Expr::Pipe`].
        #[derive(Debug, Clone, PartialEq)]
        pub struct Pipe {
            pub lhs: Box<Expr>,
            pub rhs: Box<Expr>,
            pub dbg: Option<DebugLoc>,
        }

        /// See [`Expr::Measure`].
        #[derive(Debug, Clone, PartialEq)]
        pub struct Measure {
            pub basis: Basis,
            pub dbg: Option<DebugLoc>,
        }

        /// See [`Expr::Discard`].
        #[derive(Debug, Clone, PartialEq)]
        pub struct Discard {
            pub dbg: Option<DebugLoc>,
        }

        /// See [`Expr::Tensor`].
        #[derive(Debug, Clone, PartialEq)]
        pub struct Tensor {
            pub vals: Vec<Expr>,
            pub dbg: Option<DebugLoc>,
        }

        /// See [`Expr::BasisTranslation`].
        #[derive(Debug, Clone, PartialEq)]
        pub struct BasisTranslation {
            pub bin: Basis,
            pub bout: Basis,
            pub dbg: Option<DebugLoc>,
        }

        /// See [`Expr::Predicated`].
        #[derive(Debug, Clone, PartialEq)]
        pub struct Predicated {
            pub then_func: Box<Expr>,
            pub else_func: Box<Expr>,
            pub pred: Basis,
            pub dbg: Option<DebugLoc>,
        }

        /// See [`Expr::NonUniformSuperpos`].
        #[derive(Debug, Clone, PartialEq)]
        pub struct NonUniformSuperpos {
            pub pairs: Vec<(f64, QLit)>,
            pub dbg: Option<DebugLoc>,
        }

        /// See [`Expr::Ensemble`].
        #[derive(Debug, Clone, PartialEq)]
        pub struct Ensemble {
            pub pairs: Vec<(f64, QLit)>,
            pub dbg: Option<DebugLoc>,
        }

        /// See [`Expr::Conditional`].
        #[derive(Debug, Clone, PartialEq)]
        pub struct Conditional {
            pub then_expr: Box<Expr>,
            pub else_expr: Box<Expr>,
            pub cond: Box<Expr>,
            pub dbg: Option<DebugLoc>,
        }

        /// See [`Expr::QLitExpr`].
        #[derive(Debug, Clone, PartialEq)]
        pub struct QLitExpr {
            pub qlit: QLit,
            pub dbg: Option<DebugLoc>,
        }

        /// See [`Expr::BitLiteral`].
        #[derive(Debug, Clone, PartialEq)]
        pub struct BitLiteral {
            pub val: UBig,
            pub n_bits: usize,
            pub dbg: Option<DebugLoc>,
        }

        /// See [`Expr::QubitRef`].
        #[derive(Debug, Clone, PartialEq)]
        pub struct QubitRef {
            pub index: usize,
        }

        #[derive(Debug, Clone, PartialEq)]
        pub enum Expr {
            /// A variable name used in an expression. Example syntax:
            /// ```text
            /// my_var
            /// ```
            Variable(Variable),

            /// A unit literal. Represents an empty register or void. Example syntax:
            /// ```text
            /// []
            /// ```
            UnitLiteral(UnitLiteral),

            /// Embeds a classical function into a quantum context. Example syntax:
            /// ```text
            /// my_classical_func.sign
            /// ```
            EmbedClassical(EmbedClassical),

            /// Takes the adjoint of a function value. Example syntax:
            /// ```text
            /// ~f
            /// ```
            Adjoint(Adjoint),

            /// Calls a function value. Example syntax for `f(x)`:
            /// ```text
            /// x | f
            /// ```
            Pipe(Pipe),

            /// A function value that measures its input when called. Example syntax:
            /// ```text
            /// {'0','1'}.measure
            /// ```
            Measure(Measure),

            /// A function value that discards its input when called. Example syntax:
            /// ```text
            /// discard
            /// ```
            Discard(Discard),

            /// A tensor product of function values or register values. Example syntax:
            /// ```text
            /// '0' * '1' * '0'
            /// ```
            Tensor(Tensor),

            /// The mighty basis translation. Example syntax:
            /// ```text
            /// {'0','1'} >> {'0',-'1'}
            /// ```
            BasisTranslation(BasisTranslation),

            /// A function value that, when called, runs a function value (`then_func`)
            /// in a proper subspace and another function (`else_func`) in the orthogonal
            /// complement of that subspace. Example syntax:
            /// ```text
            /// flip if {'1_'} else id
            /// ```
            Predicated(Predicated),

            /// A superposition of qubit literals that may not have uniform
            /// probabilities. Example syntax:
            /// ```text
            /// 0.25*'0' + 0.75*'1'
            /// ```
            NonUniformSuperpos(NonUniformSuperpos),

            /// An ensemble of qubit literals that may not have uniform
            /// probabilities. Example syntax:
            /// ```text
            /// 0.25*'0' ^ 0.75*'1'
            /// ```
            /// Another example:
            /// ```text
            /// '0' ^ '1'
            /// ```
            Ensemble(Ensemble),

            /// A classical conditional (ternary) expression. Example syntax:
            /// ```text
            /// flip if meas_result else id
            /// ```
            Conditional(Conditional),

            /// A qubit literal. Example syntax:
            /// ```text
            /// '0' + '1'
            /// ```
            QLitExpr(QLitExpr),

            /// A classical bit literal. Example syntax:
            /// ```text
            /// bit[4](0b1101)
            /// ```
            BitLiteral(BitLiteral),

            /// A reference to a qubit, q_i in Appendix A of arXiv:2404.12603. This is
            /// only involved in intermediate computations, so there is no Python DSL
            /// syntax that can (directly) produce this node. However, we implement the
            /// Display string as `q[i]`, although programmers should never see this
            /// node printed.
            QubitRef(QubitRef),
        }
    }
}

impl Expr {
    /// Returns the debug location for this expression.
    pub fn get_dbg(&self) -> Option<DebugLoc> {
        match self {
            Expr::Variable(Variable { dbg, .. })
            | Expr::UnitLiteral(UnitLiteral { dbg })
            | Expr::EmbedClassical(EmbedClassical { dbg, .. })
            | Expr::Adjoint(Adjoint { dbg, .. })
            | Expr::Pipe(Pipe { dbg, .. })
            | Expr::Measure(Measure { dbg, .. })
            | Expr::Discard(Discard { dbg, .. })
            | Expr::Tensor(Tensor { dbg, .. })
            | Expr::BasisTranslation(BasisTranslation { dbg, .. })
            | Expr::Predicated(Predicated { dbg, .. })
            | Expr::NonUniformSuperpos(NonUniformSuperpos { dbg, .. })
            | Expr::Ensemble(Ensemble { dbg, .. })
            | Expr::Conditional(Conditional { dbg, .. })
            | Expr::QLitExpr(QLitExpr { dbg, .. })
            | Expr::BitLiteral(BitLiteral { dbg, .. }) => dbg.clone(),

            Expr::QubitRef(_) => None,
        }
    }

    /// Called by `gen_rebuild` for the `canonicalize` config. The resulting
    /// rebuild code is called in the implementation of [`Canonicalize`] trait.
    pub(crate) fn canonicalize_rewriter(self) -> Self {
        match self {
            Expr::Tensor(Tensor { vals, dbg }) => {
                let vals: Vec<_> = vals
                    .into_iter()
                    .filter_map(|val| {
                        if let Expr::UnitLiteral(_) = val {
                            None
                        } else {
                            Some(val)
                        }
                    })
                    .collect();

                if vals.is_empty() {
                    Expr::UnitLiteral(UnitLiteral { dbg })
                } else {
                    Expr::Tensor(Tensor { vals, dbg })
                }
            }

            already_canon @ (Expr::Variable(_)
            | Expr::UnitLiteral(_)
            | Expr::EmbedClassical(_)
            | Expr::Adjoint(_)
            | Expr::Pipe(_)
            | Expr::Measure(_)
            | Expr::Discard(_)
            | Expr::BasisTranslation(_)
            | Expr::Predicated(_)
            | Expr::NonUniformSuperpos(_)
            | Expr::Ensemble(_)
            | Expr::Conditional(_)
            | Expr::QLitExpr(_)
            | Expr::BitLiteral(_)
            | Expr::QubitRef(_)) => already_canon,
        }
    }
}

impl Trivializable for Expr {
    /// Trivial expression is a unit literal.
    fn trivial(dbg: Option<DebugLoc>) -> Self {
        Self::UnitLiteral(UnitLiteral { dbg })
    }

    /// Returns true if this is a unit literal.
    fn is_trivial(&self) -> bool {
        if let Expr::UnitLiteral(_) = self {
            true
        } else {
            false
        }
    }
}

impl Canonicalizable for Expr {
    fn canonicalize(self) -> Self {
        rebuild!(Expr, self, canonicalize)
    }
}

impl fmt::Display for Expr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        visitor_write! { Expr, self,
            Expr::Variable(var) => write!(f, "{}", var), // Defer to impl in ast.rs
            Expr::UnitLiteral(UnitLiteral { .. }) => write!(f, "[]"),
            Expr::EmbedClassical(EmbedClassical {
                func_name,
                embed_kind: EmbedKind::Sign,
                ..
            }) => {
                write!(f, "{}.{}", func_name, "sign")
            }
            Expr::EmbedClassical(EmbedClassical {
                func_name,
                embed_kind: EmbedKind::Xor,
                ..
            }) => {
                write!(f, "{}.{}", func_name, "xor")
            }
            Expr::EmbedClassical(EmbedClassical {
                func_name,
                embed_kind: EmbedKind::InPlace,
                ..
            }) => {
                write!(f, "{}.{}", func_name, "inplace")
            }
            Expr::Adjoint(Adjoint { func, .. }) => write!(f, "~({!})", *func),
            Expr::Pipe(Pipe { lhs, rhs, .. }) => write!(f, "({!}) | ({!})", *lhs, *rhs),
            Expr::Measure(Measure { basis, .. }) => write!(f, "({}).measure", basis),
            Expr::Discard(Discard { .. }) => write!(f, "discard"),
            Expr::Tensor(Tensor { vals, ..}) => {
                write!(f, "({!:,})", vals.iter(), '*')
            }
            Expr::BasisTranslation(BasisTranslation { bin, bout, .. }) => {
                write!(f, "({}) >> ({})", bin, bout)
            }
            Expr::Predicated(Predicated {
                then_func,
                else_func,
                pred,
                ..
            }) => write!(f, "({!}) if ({}) else ({!})", then_func, pred, else_func),
            Expr::NonUniformSuperpos(NonUniformSuperpos{ pairs, ..}) => {
                write!(f, "{}",
                        pairs.iter()
                        .format_with(
                            " + ",
                            |(prob, qlit), f| {
                                f(&format_args!("{}*({})", prob, qlit))
                            }
                        )
                      )
            }
            Expr::Ensemble(Ensemble {pairs, ..} ) => {
                write!(
                    f, "{}",
                        pairs.iter()
                        .format_with(
                            "+",
                            |(prob, qlit), f | {
                                f(&format_args!("{}*({})",  prob, qlit))
                            }
                        ))
            }
            Expr::Conditional(Conditional {
                then_expr,
                else_expr,
                cond,
                ..
            }) => write!(f, "({!}) if ({!}) else ({!})", then_expr, cond, else_expr),
            Expr::QLitExpr(QLitExpr { qlit, .. }) => write!(f, "{}", qlit),
            Expr::BitLiteral(blit) => write!(f, "{}", blit), // Defer to impl in ast.rs
            Expr::QubitRef(QubitRef { index }) => write!(f, "q[{}]", index),
        }
    }
}

/* impl fmt::Display for Expr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Expr::Variable(var) => write!(f, "{}", var), // Defer to impl in ast.rs
            Expr::UnitLiteral(UnitLiteral { .. }) => write!(f, "[]"),
            Expr::EmbedClassical(EmbedClassical {
                func_name,
                embed_kind,
                ..
            }) => {
                let embed_kind_str = match embed_kind {
                    EmbedKind::Sign => "sign",
                    EmbedKind::Xor => "xor",
                    EmbedKind::InPlace => "inplace",
                };
                write!(f, "{}.{}", func_name, embed_kind_str)
            }
            Expr::Adjoint(Adjoint { func, .. }) => write!(f, "~({})", *func),
            Expr::Pipe(Pipe { lhs, rhs, .. }) => write!(f, "({}) | ({})", *lhs, *rhs),
            Expr::Measure(Measure { basis, .. }) => write!(f, "({}).measure", basis),
            Expr::Discard(Discard { .. }) => write!(f, "discard"),
            Expr::Tensor(Tensor { vals, .. }) => {
                for (i, val) in vals.iter().enumerate() {
                    if i > 0 {
                        write!(f, "*")?;
                    }
                    write!(f, "({})", val)?;
                }
                Ok(())
            }
            Expr::BasisTranslation(BasisTranslation { bin, bout, .. }) => {
                write!(f, "({}) >> ({})", bin, bout)
            }
            Expr::Predicated(Predicated {
                then_func,
                else_func,
                pred,
                ..
            }) => write!(f, "({}) if ({}) else ({})", then_func, pred, else_func),
            Expr::NonUniformSuperpos(NonUniformSuperpos { pairs, .. }) => {
                for (i, (prob, qlit)) in pairs.iter().enumerate() {
                    if i > 0 {
                        write!(f, " + ")?;
                    }
                    write!(f, "{}*({})", prob, qlit)?;
                }
                Ok(())
            }
            Expr::Ensemble(Ensemble { pairs, .. }) => {
                for (i, (prob, qlit)) in pairs.iter().enumerate() {
                    if i > 0 {
                        write!(f, " ^ ")?;
                    }
                    write!(f, "{}*({})", prob, qlit)?;
                }
                Ok(())
            }
            Expr::Conditional(Conditional {
                then_expr,
                else_expr,
                cond,
                ..
            }) => write!(f, "({}) if ({}) else ({})", then_expr, cond, else_expr),
            Expr::QLitExpr(QLitExpr { qlit, .. }) => write!(f, "{}", qlit),
            Expr::BitLiteral(blit) => write!(f, "{}", blit), // Defer to impl in ast.rs
            Expr::QubitRef(QubitRef { index }) => write!(f, "q[{}]", index),
        }
    }
}

*/

impl ToPythonCode for Expr {
    fn fmt_py(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Expr::Variable(Variable { name, .. }) => write!(f, "{}", name),
            Expr::UnitLiteral(UnitLiteral { .. }) => write!(f, "[]"),
            Expr::EmbedClassical(EmbedClassical {
                func_name,
                embed_kind,
                ..
            }) => {
                let embed_kind_str = match embed_kind {
                    EmbedKind::Sign => "SIGN",
                    EmbedKind::Xor => "XOR",
                    EmbedKind::InPlace => "INPLACE",
                };
                write!(f, "__EMBED_{}__({})", embed_kind_str, func_name)
            }
            Expr::Adjoint(Adjoint { func, .. }) => {
                write!(f, "~(")?;
                func.fmt_py(f)?;
                write!(f, ")")
            }
            Expr::Pipe(Pipe { lhs, rhs, .. }) => {
                write!(f, "(")?;
                lhs.fmt_py(f)?;
                write!(f, ") | (")?;
                rhs.fmt_py(f)?;
                write!(f, ")")
            }
            Expr::Measure(Measure { basis, .. }) => {
                write!(f, "__MEASURE__(")?;
                basis.fmt_py(f)?;
                write!(f, ")")
            }
            Expr::Discard(Discard { .. }) => write!(f, "__DISCARD__()"),
            Expr::Tensor(Tensor { vals, .. }) => {
                for (i, val) in vals.iter().enumerate() {
                    if i > 0 {
                        write!(f, "*")?;
                    }
                    write!(f, "(")?;
                    val.fmt_py(f)?;
                    write!(f, ")")?;
                }
                Ok(())
            }
            Expr::BasisTranslation(BasisTranslation { bin, bout, .. }) => {
                write!(f, "(")?;
                bin.fmt_py(f)?;
                write!(f, ") >> (")?;
                bout.fmt_py(f)?;
                write!(f, ")")
            }
            Expr::Predicated(Predicated {
                then_func,
                else_func,
                pred,
                ..
            }) => {
                write!(f, "(")?;
                then_func.fmt_py(f)?;
                write!(f, ") if (")?;
                pred.fmt_py(f)?;
                write!(f, ") else (")?;
                else_func.fmt_py(f)?;
                write!(f, ")")
            }
            Expr::NonUniformSuperpos(NonUniformSuperpos { pairs, .. }) => {
                for (i, (prob, qlit)) in pairs.iter().enumerate() {
                    if i > 0 {
                        write!(f, " + ")?;
                    }
                    write!(f, "{}*(", prob)?;
                    qlit.fmt_py(f)?;
                    write!(f, ")")?;
                }
                Ok(())
            }
            Expr::Ensemble(Ensemble { pairs, .. }) => {
                for (i, (prob, qlit)) in pairs.iter().enumerate() {
                    if i > 0 {
                        write!(f, " ^ ")?;
                    }
                    write!(f, "{}*(", prob)?;
                    qlit.fmt_py(f)?;
                    write!(f, ")")?;
                }
                Ok(())
            }
            Expr::Conditional(Conditional {
                then_expr,
                else_expr,
                cond,
                ..
            }) => {
                write!(f, "(")?;
                then_expr.fmt_py(f)?;
                write!(f, ") if (")?;
                cond.fmt_py(f)?;
                write!(f, ") else (")?;
                else_expr.fmt_py(f)?;
                write!(f, ")")
            }
            Expr::QLitExpr(QLitExpr { qlit, .. }) => qlit.fmt_py(f),
            // This is not completely trivial, so use the Display
            // implementation from ast.rs.
            Expr::BitLiteral(blit) => write!(f, "{}", blit),
            Expr::QubitRef(QubitRef { index }) => write!(f, "q[{}]", index),
        }
    }
}

// ----- Qubit Literals -----

#[gen_rebuild {
    into_basis_vector(
        rewrite(into_basis_vector_rewriter),
        rewrite_to(QLit => Vector),
    ),
}]
#[derive(Debug, Clone, PartialEq)]
pub enum QLit {
    ZeroQubit {
        dbg: Option<DebugLoc>,
    },
    OneQubit {
        dbg: Option<DebugLoc>,
    },
    QubitTilt {
        q: Box<QLit>,
        angle_deg: f64,
        dbg: Option<DebugLoc>,
    },
    UniformSuperpos {
        q1: Box<QLit>,
        q2: Box<QLit>,
        dbg: Option<DebugLoc>,
    },
    QubitTensor {
        qs: Vec<QLit>,
        dbg: Option<DebugLoc>,
    },
    QubitUnit {
        dbg: Option<DebugLoc>,
    },
}

impl QLit {
    /// Returns the debug location for this qubit literal.
    pub fn get_dbg(&self) -> Option<DebugLoc> {
        let (QLit::ZeroQubit { dbg }
        | QLit::OneQubit { dbg }
        | QLit::QubitTilt { dbg, .. }
        | QLit::UniformSuperpos { dbg, .. }
        | QLit::QubitTensor { dbg, .. }
        | QLit::QubitUnit { dbg }) = self;
        dbg.clone()
    }

    pub(crate) fn into_basis_vector_rewriter(
        rewritten: rewrite_ty!(QLit, into_basis_vector),
    ) -> Vector {
        rewrite_match! {QLit, into_basis_vector, rewritten,
            ZeroQubit { dbg } => Vector::ZeroVector { dbg },
            OneQubit { dbg } => Vector::OneVector { dbg },
            QubitTilt { q, angle_deg, dbg } => {
                let q = Box::new(q);
                Vector::VectorTilt { q, angle_deg, dbg }
            }
            UniformSuperpos { q1, q2, dbg } => {
                let q1 = Box::new(q1);
                let q2 = Box::new(q2);
                Vector::UniformVectorSuperpos { q1, q2, dbg }
            }
            QubitTensor { qs, dbg } => Vector::VectorTensor { qs, dbg },
            QubitUnit { dbg } => Vector::VectorUnit { dbg },
        }
    }

    /// Converts an owned qubit literal into a basis vector since in
    /// the spec, every ql is a bv.
    pub fn into_basis_vector(self) -> Vector {
        rebuild!(QLit, self, into_basis_vector)
    }

    /// Converts a reference to a qubit literal into a basis vector since in
    /// the spec, every ql is a bv.
    pub fn to_basis_vector(&self) -> Vector {
        self.clone().into_basis_vector()
    }

    /// Returns an equivalent qubit literal in canon form (see
    /// Vector::canonicalize()).
    pub fn canonicalize(self) -> QLit {
        self.into_basis_vector()
            .canonicalize()
            .try_into_qubit_literal()
            .expect(concat!(
                "converting a qlit to a basis vector should allow converting ",
                "back to a qlit"
            ))
    }
}

impl fmt::Display for QLit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            QLit::ZeroQubit { .. } => write!(f, "'0'"),
            QLit::OneQubit { .. } => write!(f, "'1'"),
            QLit::QubitTilt { q, angle_deg, .. } => {
                if angles_are_approx_equal(*angle_deg, 180.0) {
                    write!(f, "-{}", **q)
                } else {
                    write!(f, "({})@{}", **q, *angle_deg)
                }
            }
            QLit::UniformSuperpos { q1, q2, .. } => write!(f, "({}) + ({})", **q1, **q2),
            QLit::QubitTensor { qs, .. } => {
                for (i, q) in qs.iter().enumerate() {
                    if i > 0 {
                        write!(f, "*")?;
                    }
                    write!(f, "({})", q)?;
                }
                Ok(())
            }
            QLit::QubitUnit { .. } => write!(f, "''"),
        }
    }
}

impl ToPythonCode for QLit {
    fn fmt_py(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            QLit::ZeroQubit { .. } => write!(f, "__SYM_STD0__()"),
            QLit::OneQubit { .. } => write!(f, "__SYM_STD1__()"),
            QLit::QubitTilt { q, angle_deg, .. } => {
                write!(f, "(")?;
                q.fmt_py(f)?;
                write!(f, ")@{}", *angle_deg)
            }
            QLit::UniformSuperpos { q1, q2, .. } => {
                write!(f, "(")?;
                q1.fmt_py(f)?;
                write!(f, ") + (")?;
                q2.fmt_py(f)?;
                write!(f, ")")
            }
            QLit::QubitTensor { qs, .. } => {
                for (i, q) in qs.iter().enumerate() {
                    if i > 0 {
                        write!(f, "*")?;
                    }
                    write!(f, "(")?;
                    q.fmt_py(f)?;
                    write!(f, ")")?;
                }
                Ok(())
            }
            QLit::QubitUnit { .. } => write!(f, "''"),
        }
    }
}

// ----- Vector -----

/// Represents either a padding ('?') or a target ('_') vector atom. This is
/// "va" in the spec, except limited to the variants we actually use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VectorAtomKind {
    PadAtom,    // '?'
    TargetAtom, // '_'
}

impl fmt::Display for VectorAtomKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            VectorAtomKind::PadAtom => write!(f, "'?'"),
            VectorAtomKind::TargetAtom => write!(f, "'_'"),
        }
    }
}

#[gen_rebuild {
    try_into_qubit_literal(
        option,
        rewrite(try_into_qubit_literal_rewriter),
        rewrite_to(Vector => QLit),
    ),
    into_strip_dbg(
        rewrite(into_strip_dbg_rewriter),
    ),
    into_explicit(
        rewrite(into_explicit_rewriter),
    ),
    canonicalize(
        rewrite(canonicalize_rewriter),
    ),
}]
#[derive(Debug, Clone)]
pub enum Vector {
    /// The first standard basis vector, |0⟩. Example syntax:
    /// ```text
    /// '0'
    /// ```
    ZeroVector { dbg: Option<DebugLoc> },

    /// The second standard basis vector, |1⟩. Example syntax:
    /// ```text
    /// '1'
    /// ```
    OneVector { dbg: Option<DebugLoc> },

    /// The pad atom. Example syntax:
    /// ```text
    /// '?'
    /// ```
    PadVector { dbg: Option<DebugLoc> },

    /// The target atom. Example syntax:
    /// ```text
    /// '_'
    /// ```
    TargetVector { dbg: Option<DebugLoc> },

    /// Tilts a vector. Example syntax:
    /// ```text
    /// '1' @ 180
    /// ```
    VectorTilt {
        q: Box<Vector>,
        angle_deg: f64,
        dbg: Option<DebugLoc>,
    },

    /// A uniform vector superposition. Example syntax:
    /// ```text
    /// '0' + '1'
    /// ```
    UniformVectorSuperpos {
        q1: Box<Vector>,
        q2: Box<Vector>,
        dbg: Option<DebugLoc>,
    },

    /// A tensor product. Example syntax:
    /// ```text
    /// '0' * '1'
    /// ```
    VectorTensor {
        qs: Vec<Vector>,
        dbg: Option<DebugLoc>,
    },

    /// An empty vector. Example syntax:
    /// ```text
    /// ''
    /// ```
    VectorUnit { dbg: Option<DebugLoc> },
}

impl Vector {
    /// Returns the source code location for this vector.
    pub fn get_dbg(&self) -> Option<DebugLoc> {
        match self {
            Vector::ZeroVector { dbg }
            | Vector::OneVector { dbg }
            | Vector::PadVector { dbg }
            | Vector::TargetVector { dbg }
            | Vector::VectorTilt { dbg, .. }
            | Vector::UniformVectorSuperpos { dbg, .. }
            | Vector::VectorTensor { dbg, .. }
            | Vector::VectorUnit { dbg } => dbg.clone(),
        }
    }

    /// Converts this vector into a qubit literal. Returns None if this vector
    /// contains any padding or target atoms.
    pub fn try_into_qubit_literal(self) -> Option<QLit> {
        rebuild!(Vector, self, try_into_qubit_literal)
    }

    pub(crate) fn try_into_qubit_literal_rewriter(
        rewritten: rewrite_ty!(Vector, try_into_qubit_literal),
    ) -> Option<QLit> {
        rewrite_match! {Vector, try_into_qubit_literal, rewritten,
            ZeroVector { dbg } => Some(QLit::ZeroQubit { dbg }),

            OneVector { dbg } => Some(QLit::OneQubit { dbg }),

            VectorTilt { q, angle_deg, dbg } => {
                let q = Box::new(q);
                Some(QLit::QubitTilt { q, angle_deg, dbg })
            }

            UniformVectorSuperpos { q1, q2, dbg } => {
                let q1 = Box::new(q1);
                let q2 = Box::new(q2);
                Some(QLit::UniformSuperpos { q1, q2, dbg })
            }

            VectorTensor { qs, dbg } => Some(QLit::QubitTensor { qs, dbg }),

            VectorUnit { dbg } => Some(QLit::QubitUnit { dbg }),

            PadVector { .. } | TargetVector { .. } => None,
        }
    }

    /// Returns a version of this vector with no debug info. Useful for
    /// comparing vectors/bases without considering debug info.
    pub fn into_strip_dbg(self) -> Self {
        rebuild!(Vector, self, into_strip_dbg)
    }

    pub(crate) fn into_strip_dbg_rewriter(self) -> Self {
        match self {
            Vector::ZeroVector { .. } => Vector::ZeroVector { dbg: None },
            Vector::OneVector { .. } => Vector::OneVector { dbg: None },
            Vector::PadVector { .. } => Vector::PadVector { dbg: None },
            Vector::TargetVector { .. } => Vector::TargetVector { dbg: None },
            Vector::VectorTilt { q, angle_deg, .. } => Vector::VectorTilt {
                q,
                angle_deg,
                dbg: None,
            },
            Vector::UniformVectorSuperpos { q1, q2, .. } => {
                Vector::UniformVectorSuperpos { q1, q2, dbg: None }
            }
            Vector::VectorTensor { qs, .. } => Vector::VectorTensor { qs, dbg: None },
            Vector::VectorUnit { .. } => Vector::VectorUnit { dbg: None },
        }
    }

    /// Returns whether vectors are equivalent, using
    /// [`Vector::into_strip_dbg]` and [`angles_are_approx_equal`]
    pub fn approx_equal(&self, other: &Vector) -> bool {
        match (self, other) {
            (Vector::ZeroVector { .. }, Vector::ZeroVector { .. }) => true,
            (Vector::OneVector { .. }, Vector::OneVector { .. }) => true,
            (Vector::PadVector { .. }, Vector::PadVector { .. }) => true,
            (Vector::TargetVector { .. }, Vector::TargetVector { .. }) => true,
            (Vector::VectorUnit { .. }, Vector::VectorUnit { .. }) => true,
            (
                Vector::VectorTilt {
                    q: q1,
                    angle_deg: deg1,
                    ..
                },
                Vector::VectorTilt {
                    q: q2,
                    angle_deg: deg2,
                    ..
                },
            ) => angles_are_approx_equal(*deg1, *deg2) && q1.approx_equal(&q2),
            (
                Vector::UniformVectorSuperpos {
                    q1: q1a, q2: q2a, ..
                },
                Vector::UniformVectorSuperpos {
                    q1: q1b, q2: q2b, ..
                },
            ) => q1a.approx_equal(&q1b) && q2a.approx_equal(&q2b),
            (Vector::VectorTensor { qs: qs1, .. }, Vector::VectorTensor { qs: qs2, .. }) => {
                qs1.len() == qs2.len() && qs1.iter().zip(qs2).all(|(v1, v2)| v1.approx_equal(&v2))
            }
            _ => false,
        }
    }

    /// Returns number of non-target and non-padding qubits represented by a basis
    /// vector (`⌊bv⌋` in the spec) or None if the basis vector is malformed
    /// (currently, if both sides of a superposition have different dimensions
    ///  or if a tensor product has less than two vectors).
    pub fn get_explicit_dim(&self) -> Option<usize> {
        match self {
            Vector::ZeroVector { .. } | Vector::OneVector { .. } => Some(1),
            Vector::PadVector { .. } | Vector::TargetVector { .. } => Some(0),
            Vector::VectorTilt { q: inner_bv, .. } => inner_bv.get_explicit_dim(),
            Vector::UniformVectorSuperpos {
                q1: inner_bv_1,
                q2: inner_bv_2,
                ..
            } => match (inner_bv_1.get_explicit_dim(), inner_bv_2.get_explicit_dim()) {
                (Some(inner_dim1), Some(inner_dim2)) if inner_dim1 == inner_dim2 => {
                    Some(inner_dim1)
                }
                _ => None,
            },
            Vector::VectorTensor { qs: inner_bvs, .. } => {
                if inner_bvs.len() < 2 {
                    None
                } else {
                    inner_bvs
                        .iter()
                        .try_fold(0, |acc, v| v.get_explicit_dim().map(|dim| acc + dim))
                }
            }
            Vector::VectorUnit { .. } => Some(0),
        }
    }

    /// Returns number of qubits represented by a basis vector, including
    /// padding and target qubits. (This is |bv| in the spec.) Returns None
    /// if the basis vector is malformed (currently, if both sides of a
    /// superposition have different dimensions or if a tensor product has less
    /// than two vectors).
    pub fn get_dim(&self) -> Option<usize> {
        match self {
            Vector::ZeroVector { .. }
            | Vector::OneVector { .. }
            | Vector::PadVector { .. }
            | Vector::TargetVector { .. } => Some(1),
            Vector::VectorTilt { q: inner_bv, .. } => inner_bv.get_dim(),
            Vector::UniformVectorSuperpos {
                q1: inner_bv_1,
                q2: inner_bv_2,
                ..
            } => match (inner_bv_1.get_dim(), inner_bv_2.get_dim()) {
                (Some(inner_dim1), Some(inner_dim2)) if inner_dim1 == inner_dim2 => {
                    Some(inner_dim1)
                }
                _ => None,
            },
            Vector::VectorTensor { qs: inner_bvs, .. } => {
                if inner_bvs.len() < 2 {
                    None
                } else {
                    inner_bvs
                        .iter()
                        .try_fold(0, |acc, v| v.get_dim().map(|dim| acc + dim))
                }
            }
            Vector::VectorUnit { .. } => Some(0),
        }
    }

    /// Returns the zero-indexed qubit positions that are correspond to the
    /// provided atom. This is `Ξva[bv]` in the spec. Returning None indicates
    /// a malformed vector (currently, a superposition whose possibilities do
    /// not have matching indices or an empty tensor product).
    pub fn get_atom_indices(&self, atom: VectorAtomKind) -> Option<Vec<usize>> {
        match (self, atom) {
            (Vector::ZeroVector { .. }, _)
            | (Vector::OneVector { .. }, _)
            | (Vector::VectorUnit { .. }, _)
            | (Vector::PadVector { .. }, VectorAtomKind::TargetAtom)
            | (Vector::TargetVector { .. }, VectorAtomKind::PadAtom) => Some(vec![]),

            (Vector::PadVector { .. }, VectorAtomKind::PadAtom)
            | (Vector::TargetVector { .. }, VectorAtomKind::TargetAtom) => Some(vec![0]),

            (Vector::VectorTilt { q, .. }, _) => q.get_atom_indices(atom),

            (Vector::UniformVectorSuperpos { q1, q2, .. }, _) => {
                let left_indices = q1.get_atom_indices(atom)?;
                let right_indices = q2.get_atom_indices(atom)?;
                if left_indices == right_indices {
                    Some(left_indices)
                } else {
                    None
                }
            }

            (Vector::VectorTensor { qs, .. }, _) => {
                if qs.len() < 2 {
                    None
                } else {
                    let mut offset = 0;
                    let mut indices = vec![];
                    for vec in qs {
                        let vec_indices = vec.get_atom_indices(atom)?;
                        for idx in vec_indices {
                            indices.push(idx + offset);
                        }
                        let vec_dim = vec.get_dim()?;
                        offset += vec_dim;
                    }
                    Some(indices)
                }
            }
        }
    }

    /// Returns a version of a vector reference without any '?' or '_' atoms.
    /// Assumes the original vector is well-typed.
    pub fn to_explicit(&self) -> Self {
        self.clone().into_explicit()
    }

    /// Returns a version of an owned vector without any '?' or '_' atoms.
    /// Assumes the original vector is well-typed.
    pub fn into_explicit(self) -> Self {
        rebuild!(Vector, self, into_explicit)
    }

    pub(crate) fn into_explicit_rewriter(self) -> Self {
        match self {
            Vector::PadVector { dbg } | Vector::TargetVector { dbg } => Vector::VectorUnit { dbg },

            Vector::VectorTilt { q, angle_deg, dbg } => {
                if let Vector::VectorUnit { dbg } = *q {
                    Vector::VectorUnit { dbg }
                } else {
                    Vector::VectorTilt { q, angle_deg, dbg }
                }
            }

            Vector::VectorTensor { qs, dbg } => {
                let qs: Vec<_> = qs
                    .into_iter()
                    .filter(|vec| !matches!(vec, Vector::VectorUnit { .. }))
                    .collect();
                // Make an assumption that this is well-formed
                if qs.is_empty() {
                    Vector::VectorUnit { dbg }
                } else if qs.len() == 1 {
                    qs.into_iter().next().expect("qs is empty yet has length 1")
                } else {
                    Vector::VectorTensor { qs, dbg }
                }
            }

            already_explicit @ (Vector::ZeroVector { .. }
            | Vector::OneVector { .. }
            | Vector::VectorUnit { .. }
            | Vector::UniformVectorSuperpos { .. }) => already_explicit,
        }
    }

    /// Suppose left=(l1, l2) and right=(r1, r2). Then this function considers
    /// (l1 + l2) + (r1 + r2). If r1 = l1@180 and r2 = l2, up to some
    /// commutation of the terms of the inner superpositions, this returns two
    /// vectors (cons, dest). cons is the constructively interfering vector
    /// (r2/l2), and dest was the destructively interfering vector (l1).
    fn try_interference<'v>(
        left: (&'v Vector, &'v Vector),
        right: (&'v Vector, &'v Vector),
    ) -> Option<(&'v Vector, &'v Vector)> {
        match (left, right) {
            (
                (q1l, q2l),
                (
                    q1r,
                    Vector::VectorTilt {
                        q: bq2r, angle_deg, ..
                    },
                ),
            )
            | (
                (q1l, q2l),
                (
                    Vector::VectorTilt {
                        q: bq2r, angle_deg, ..
                    },
                    q1r,
                ),
            )
            | (
                (q2l, q1l),
                (
                    q1r,
                    Vector::VectorTilt {
                        q: bq2r, angle_deg, ..
                    },
                ),
            )
            | (
                (q2l, q1l),
                (
                    Vector::VectorTilt {
                        q: bq2r, angle_deg, ..
                    },
                    q1r,
                ),
            ) if q1l == q1r && q2l == &**bq2r && angles_are_approx_equal(*angle_deg, 180.0) => {
                Some((q1l, q2l))
            }

            _ => None,
        }
    }

    /// Simplifies the vector `left+right` by performing interference.
    pub fn interfere(left: &Vector, right: &Vector) -> Option<Vector> {
        match (left, right) {
            (
                Vector::UniformVectorSuperpos {
                    q1: lq1, q2: lq2, ..
                },
                Vector::UniformVectorSuperpos {
                    q1: rq1, q2: rq2, ..
                },
            ) => Vector::try_interference((&**lq1, &**lq2), (&**rq1, &**rq2))
                .or_else(|| Vector::try_interference((&**rq1, &**rq2), (&**lq1, &**lq2)))
                .map(|(cons, _dest)| cons.clone()),

            (
                Vector::UniformVectorSuperpos {
                    q1: lq1, q2: lq2, ..
                },
                Vector::VectorTilt {
                    q: rq,
                    angle_deg: rangle_deg,
                    dbg,
                },
            )
            | (
                Vector::VectorTilt {
                    q: rq,
                    angle_deg: rangle_deg,
                    dbg,
                },
                Vector::UniformVectorSuperpos {
                    q1: lq1, q2: lq2, ..
                },
            ) if angles_are_approx_equal(*rangle_deg, 180.0) => match &**rq {
                Vector::UniformVectorSuperpos {
                    q1: rq1, q2: rq2, ..
                } => Vector::try_interference((&**lq1, &**lq2), (&**rq1, &**rq2))
                    .map(|(_cons, dest)| dest.clone())
                    .or_else(|| {
                        Vector::try_interference((&**rq1, &**rq2), (&**lq1, &**lq2)).map(
                            |(_cons, dest)| Vector::VectorTilt {
                                q: Box::new(dest.clone()),
                                angle_deg: 180.0,
                                dbg: dbg.clone(),
                            },
                        )
                    }),

                _ => None,
            },

            _ => None,
        }
    }

    /// Returns an equivalent vector such that:
    /// 1. No tensor product contains a []
    /// 2. No tensor product contains another tensor product
    /// 3. All @s are propagated to the outside
    /// 4. All tensor product angles in canon form (i.e., in the interval
    ///    [0.0, 360.0))
    /// 5. The terms q1 and q2 of a superposition q1+q2 are ordered such that
    ///    q1 <= q2.
    /// 6. Basic interference is performed:
    ///        (q1 + q2) + (q1 - q2) -> q1
    ///    and
    ///        (q1 + q2) + -(q1 - q2) -> q1
    /// Assumes this vector is well-typed.
    pub fn canonicalize(self) -> Vector {
        rebuild!(Vector, self, canonicalize)
    }

    pub(crate) fn canonicalize_rewriter(self) -> Self {
        match self {
            Vector::VectorTilt { q, angle_deg, dbg } => {
                if let Vector::VectorTilt {
                    q: q_inner,
                    angle_deg: angle_deg_inner,
                    ..
                } = *q
                {
                    let angle_deg_sum = angle_deg + angle_deg_inner;
                    if angle_is_approx_zero(canon_angle(angle_deg_sum)) {
                        *q_inner
                    } else {
                        Vector::VectorTilt {
                            q: q_inner,
                            angle_deg: canon_angle(angle_deg_sum),
                            dbg,
                        }
                    }
                } else if angle_is_approx_zero(canon_angle(angle_deg)) {
                    *q
                } else {
                    Vector::VectorTilt {
                        q,
                        angle_deg: canon_angle(angle_deg),
                        dbg,
                    }
                }
            }

            Vector::UniformVectorSuperpos { q1, q2, dbg } => {
                let (first_q, second_q) = if *q2 < *q1 { (*q2, *q1) } else { (*q1, *q2) };

                if let Some(vec) = Vector::interfere(&first_q, &second_q) {
                    // Call canonicalize() here in case there is
                    // e.g. a nested tilt
                    // TODO: gen_rebuild needs to allow rewriters to signal a
                    //       "retrigger" or "retry" somehow instead of doing this.
                    vec.canonicalize()
                } else {
                    Vector::UniformVectorSuperpos {
                        q1: Box::new(first_q),
                        q2: Box::new(second_q),
                        dbg,
                    }
                }
            }

            Vector::VectorTensor { qs, dbg } => {
                let mut new_qs = vec![];
                let mut angle_deg_sum = 0.0;
                let mut new_tilt_dbg = None;
                for vec in qs {
                    if let Vector::VectorTilt {
                        q,
                        angle_deg,
                        dbg: tilt_dbg,
                    } = vec
                    {
                        angle_deg_sum += angle_deg;
                        new_tilt_dbg = new_tilt_dbg.or(tilt_dbg);

                        if let Vector::VectorUnit { .. } = *q {
                            // Skip units
                        } else if let Vector::VectorTensor { qs: inner_qs, .. } = *q {
                            // No need to look for tilts here because we can
                            // inductively assume they were moved to the
                            // outside and we just found them
                            new_qs.extend(inner_qs);
                        } else {
                            new_qs.push(*q);
                        }
                    } else if let Vector::VectorUnit { .. } = vec {
                        // Skip units
                    } else if let Vector::VectorTensor { qs: inner_qs, .. } = vec {
                        // No need to look for tilts here because we can
                        // inductively assume they would've been moved to the
                        // outside and found above
                        new_qs.extend(inner_qs);
                    } else {
                        new_qs.push(vec);
                    }
                }

                let new_tensor = if new_qs.is_empty() {
                    Vector::VectorUnit { dbg: dbg.clone() }
                } else if new_qs.len() == 1 {
                    new_qs
                        .into_iter()
                        .next()
                        .expect("new_qs is empty yet has length 1")
                } else {
                    Vector::VectorTensor {
                        qs: new_qs,
                        dbg: dbg.clone(),
                    }
                };

                if angle_is_approx_zero(canon_angle(angle_deg_sum)) {
                    new_tensor
                } else {
                    Vector::VectorTilt {
                        q: Box::new(new_tensor),
                        angle_deg: canon_angle(angle_deg_sum),
                        dbg: new_tilt_dbg.or(dbg),
                    }
                }
            }

            already_canon @ (Vector::ZeroVector { .. }
            | Vector::OneVector { .. }
            | Vector::PadVector { .. }
            | Vector::TargetVector { .. }
            | Vector::VectorUnit { .. }) => already_canon,
        }
    }

    /// Removes phases (except those directly inside superpositions). This is
    /// intended for use in span equivalence checking. The vector should be
    /// type-checked and canonicalized first.
    pub fn normalize(self) -> Vector {
        match self {
            Vector::VectorTilt { q, .. } => *q,
            already_normal => already_normal,
        }
    }

    /// Tries to factor `prefix` from this vector and returns the remainder.
    /// Expects a canonicalized and normalized vector first, but does not
    /// return a canonicalized vector.
    pub fn factor(&self, prefix: &Vector) -> Option<Vector> {
        if self == prefix {
            Some(Vector::VectorUnit {
                dbg: self.get_dbg(),
            })
        } else if let Vector::VectorTensor { qs, dbg } = self {
            assert!(qs.len() >= 2, "empty tensor products are not allowed");
            let remainder = qs[0].factor(prefix)?;
            let new_qs = std::iter::once(remainder)
                .chain(qs.iter().skip(1).cloned())
                .collect();
            Some(Vector::VectorTensor {
                qs: new_qs,
                dbg: dbg.clone(),
            })
        } else {
            // TODO: support more cases
            None
        }
    }

    /// Converts a Vector into a Rust Vec of Vectors to remove
    /// VectorTensor and VectorUnits from the state.
    /// Similar to Basis::to_vec
    pub fn to_vec(self) -> Vec<Vector> {
        match self {
            Vector::VectorTensor { qs, .. } => qs,
            Vector::VectorUnit { .. } => vec![],
            other => vec![other],
        }
    }
}

impl Ord for Vector {
    fn cmp(&self, other: &Self) -> Ordering {
        match (self, other) {
            (Vector::VectorUnit { dbg: dbgl }, Vector::VectorUnit { dbg: dbgr })
            | (Vector::ZeroVector { dbg: dbgl }, Vector::ZeroVector { dbg: dbgr })
            | (Vector::OneVector { dbg: dbgl }, Vector::OneVector { dbg: dbgr })
            | (Vector::PadVector { dbg: dbgl }, Vector::PadVector { dbg: dbgr })
            | (Vector::TargetVector { dbg: dbgl }, Vector::TargetVector { dbg: dbgr }) => {
                dbgl.cmp(dbgr)
            }

            // [] is the least element
            (Vector::VectorUnit { .. }, _) => Ordering::Less,
            (_, Vector::VectorUnit { .. }) => Ordering::Greater,

            // '0' is the next least
            (Vector::ZeroVector { .. }, _) => Ordering::Less,
            (_, Vector::ZeroVector { .. }) => Ordering::Greater,

            // Then '1' is the next least
            (Vector::OneVector { .. }, _) => Ordering::Less,
            (_, Vector::OneVector { .. }) => Ordering::Greater,

            // Then '?' is the next least
            (Vector::PadVector { .. }, _) => Ordering::Less,
            (_, Vector::PadVector { .. }) => Ordering::Greater,

            // Then '_' is the next least
            (Vector::TargetVector { .. }, _) => Ordering::Less,
            (_, Vector::TargetVector { .. }) => Ordering::Greater,

            // Then @
            (
                Vector::VectorTilt {
                    q: ql,
                    angle_deg: angle_degl,
                    dbg: dbgl,
                },
                Vector::VectorTilt {
                    q: qr,
                    angle_deg: angle_degr,
                    dbg: dbgr,
                },
            ) => ql
                .cmp(qr)
                .then(angle_approx_total_cmp(*angle_degl, *angle_degr))
                .then(dbgl.cmp(dbgr)),
            (Vector::VectorTilt { .. }, _) => Ordering::Less,
            (_, Vector::VectorTilt { .. }) => Ordering::Greater,

            // Then +
            (
                Vector::UniformVectorSuperpos {
                    q1: q1l,
                    q2: q2l,
                    dbg: dbgl,
                },
                Vector::UniformVectorSuperpos {
                    q1: q1r,
                    q2: q2r,
                    dbg: dbgr,
                },
            ) => q1l.cmp(q1r).then(q2l.cmp(q2r)).then(dbgl.cmp(dbgr)),
            (Vector::UniformVectorSuperpos { .. }, _) => Ordering::Less,
            (_, Vector::UniformVectorSuperpos { .. }) => Ordering::Greater,

            // Then *
            (
                Vector::VectorTensor { qs: qsl, dbg: dbgl },
                Vector::VectorTensor { qs: qsr, dbg: dbgr },
            ) => qsl
                .iter()
                .zip(qsr.iter())
                .fold(Ordering::Equal, |acc, (l, r)| acc.then(l.cmp(r)))
                .then(dbgl.cmp(dbgr)),
        }
    }
}

impl PartialOrd for Vector {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

// Unfortunately, we can't rely on #[derive] to implement this because NaN !=
// NaN unless we intentionally use f64::total_cmp() as we do below.
impl PartialEq for Vector {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Vector::ZeroVector { dbg: dbgl }, Vector::ZeroVector { dbg: dbgr }) => dbgl == dbgr,
            (Vector::OneVector { dbg: dbgl }, Vector::OneVector { dbg: dbgr }) => dbgl == dbgr,
            (Vector::PadVector { dbg: dbgl }, Vector::PadVector { dbg: dbgr }) => dbgl == dbgr,
            (Vector::TargetVector { dbg: dbgl }, Vector::TargetVector { dbg: dbgr }) => {
                dbgl == dbgr
            }
            (
                Vector::VectorTilt {
                    q: ql,
                    angle_deg: angle_degl,
                    dbg: dbgl,
                },
                Vector::VectorTilt {
                    q: qr,
                    angle_deg: angle_degr,
                    dbg: dbgr,
                },
            ) => ql == qr && angles_are_approx_equal(*angle_degl, *angle_degr) && dbgl == dbgr,
            (
                Vector::UniformVectorSuperpos {
                    q1: q1l,
                    q2: q2l,
                    dbg: dbgl,
                },
                Vector::UniformVectorSuperpos {
                    q1: q1r,
                    q2: q2r,
                    dbg: dbgr,
                },
            ) => q1l == q1r && q2l == q2r && dbgl == dbgr,
            (
                Vector::VectorTensor { qs: qsl, dbg: dbgl },
                Vector::VectorTensor { qs: qsr, dbg: dbgr },
            ) => qsl == qsr && dbgl == dbgr,
            (Vector::VectorUnit { dbg: dbgl }, Vector::VectorUnit { dbg: dbgr }) => dbgl == dbgr,
            _ => false,
        }
    }
}

impl Eq for Vector {}

impl fmt::Display for Vector {
    /// Represents a vector in a human-readable form for error messages sent
    /// back to the programmer.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Vector::ZeroVector { .. } => write!(f, "'0'"),
            Vector::OneVector { .. } => write!(f, "'1'"),
            Vector::PadVector { .. } => write!(f, "'?'"),
            Vector::TargetVector { .. } => write!(f, "'_'"),
            Vector::VectorTilt { q, angle_deg, .. } => {
                if angles_are_approx_equal(*angle_deg, 180.0) {
                    write!(f, "-{}", **q)
                } else {
                    write!(f, "({})@{}", **q, *angle_deg)
                }
            }
            Vector::UniformVectorSuperpos { q1, q2, .. } => write!(f, "({}) + ({})", **q1, **q2),
            Vector::VectorTensor { qs, .. } => {
                for (i, q) in qs.iter().enumerate() {
                    if i > 0 {
                        write!(f, "*")?;
                    }
                    write!(f, "({})", q)?;
                }
                Ok(())
            }
            Vector::VectorUnit { .. } => write!(f, "''"),
        }
    }
}

impl ToPythonCode for Vector {
    fn fmt_py(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Vector::ZeroVector { .. } => write!(f, "__SYM_STD0__()"),
            Vector::OneVector { .. } => write!(f, "__SYM_STD1__()"),
            Vector::PadVector { .. } => write!(f, "__SYM_PAD__()"),
            Vector::TargetVector { .. } => write!(f, "__SYM_TARGET__()"),
            Vector::VectorTilt { q, angle_deg, .. } => {
                write!(f, "(")?;
                q.fmt_py(f)?;
                write!(f, ")@{}", *angle_deg)
            }
            Vector::UniformVectorSuperpos { q1, q2, .. } => {
                write!(f, "(")?;
                q1.fmt_py(f)?;
                write!(f, ") + (")?;
                q2.fmt_py(f)?;
                write!(f, ")")
            }
            Vector::VectorTensor { qs, .. } => {
                for (i, q) in qs.iter().enumerate() {
                    if i > 0 {
                        write!(f, "*")?;
                    }
                    write!(f, "(")?;
                    q.fmt_py(f)?;
                    write!(f, ")")?;
                }
                Ok(())
            }
            Vector::VectorUnit { .. } => write!(f, "''"),
        }
    }
}

// ----- Basis -----

#[gen_rebuild {
    into_strip_dbg(
        rewrite(into_strip_dbg_rewriter),
        recurse_attrs,
    ),
    canonicalize(
        recurse_attrs,
    ),
}]
#[derive(Debug, Clone, PartialEq)]
pub enum BasisGenerator {
    /// A revolve generator, used to define the Fourier basis. Example syntax:
    /// ```text
    /// __REVOLVE__('0', '1')
    /// ```
    Revolve {
        v1: Vector,
        v2: Vector,
        dbg: Option<DebugLoc>,
    },
}

impl BasisGenerator {
    /// Returns a version of this basis generator with no debug info. Useful
    /// for comparing vectors/bases without considering debug info.
    pub fn into_strip_dbg(self) -> Self {
        rebuild!(BasisGenerator, self, into_strip_dbg)
    }

    pub(crate) fn into_strip_dbg_rewriter(self) -> Self {
        match self {
            BasisGenerator::Revolve { v1, v2, dbg: _ } => {
                BasisGenerator::Revolve { v1, v2, dbg: None }
            }
        }
    }

    /// Get the number of qubits by which this basis generator would extend any
    /// basis.
    pub fn get_dim(&self) -> usize {
        match self {
            BasisGenerator::Revolve { .. } => 1,
        }
    }

    /// Returns `true` if applying this generator to **a fully spanning basis**
    /// would yield another fully spanning basis.
    pub fn fully_spans(&self) -> bool {
        match self {
            BasisGenerator::Revolve { .. } => true,
        }
    }

    /// Returns this `BasisGenerator` with any contained vectors canonicalized.
    pub fn canonicalize(self) -> Self {
        rebuild!(BasisGenerator, self, canonicalize)
    }
}

impl fmt::Display for BasisGenerator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BasisGenerator::Revolve { v1, v2, .. } => write!(f, "{{{},{}}}.revolve", v1, v2),
        }
    }
}

impl ToPythonCode for BasisGenerator {
    fn fmt_py(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BasisGenerator::Revolve { v1, v2, .. } => {
                write!(f, "__REVOLVE__(")?;
                v1.fmt_py(f)?;
                write!(f, ", ")?;
                v2.fmt_py(f)?;
                write!(f, ")")
            }
        }
    }
}

#[gen_rebuild {
    into_strip_dbg(
        rewrite(into_strip_dbg_rewriter),
        recurse_attrs,
    ),
    into_explicit(
        rewrite(into_explicit_rewriter),
        recurse_attrs,
    ),
    canonicalize(
        rewrite(canonicalize_rewriter),
        recurse_attrs,
    ),
    normalize(
        rewrite(normalize_rewriter),
        recurse_attrs,
    ),
}]
#[derive(Debug, Clone, PartialEq)]
pub enum Basis {
    /// A basis literal. Example syntax:
    /// ```text
    /// {'0', '1'}
    /// ```
    BasisLiteral {
        vecs: Vec<Vector>,
        dbg: Option<DebugLoc>,
    },

    /// An empty basis literal. Example syntax:
    /// ```text
    /// {}
    /// ```
    EmptyBasisLiteral { dbg: Option<DebugLoc> },

    /// Tensor product of bases. Example syntax:
    /// ```text
    /// {'0', '1'} * {'0'+'1', '0'-'1'} * {'0', '1'}
    /// ```
    BasisTensor {
        bases: Vec<Basis>,
        dbg: Option<DebugLoc>,
    },

    /// Apply a basis generator. Example syntax:
    /// ```text
    /// {'0'+'1', '0'-'1'} // __REVOLVE__('0', '1')
    /// ```
    ApplyBasisGenerator {
        basis: Box<Basis>,
        #[gen_rebuild::skip_recurse(into_explicit, normalize)]
        generator: BasisGenerator,
        dbg: Option<DebugLoc>,
    },
}

impl Basis {
    /// Returns the n-qubit standard basis, where n = dim.
    pub fn std(dim: usize, dbg: Option<DebugLoc>) -> Basis {
        let mut bases = vec![];
        for _ in 0..dim {
            bases.push(Basis::BasisLiteral {
                vecs: vec![
                    Vector::ZeroVector { dbg: dbg.clone() },
                    Vector::OneVector { dbg: dbg.clone() },
                ],
                dbg: dbg.clone(),
            });
        }
        Basis::BasisTensor { bases, dbg: dbg }
    }

    /// Returns a version of this basis with no debug info. Useful for
    /// comparing vectors/bases without considering debug info.
    pub fn into_strip_dbg(self) -> Self {
        rebuild!(Basis, self, into_strip_dbg)
    }

    pub(crate) fn into_strip_dbg_rewriter(self) -> Self {
        match self {
            Basis::BasisLiteral { vecs, dbg: _ } => Basis::BasisLiteral { vecs, dbg: None },
            Basis::EmptyBasisLiteral { dbg: _ } => Basis::EmptyBasisLiteral { dbg: None },
            Basis::BasisTensor { bases, dbg: _ } => Basis::BasisTensor { bases, dbg: None },
            Basis::ApplyBasisGenerator {
                basis,
                generator,
                dbg: _,
            } => Basis::ApplyBasisGenerator {
                basis,
                generator,
                dbg: None,
            },
        }
    }

    /// Returns whether bases are equivalent, using [`Basis::into_strip_dbg`]
    /// and [`angles_are_approx_equal`].
    pub fn approx_equal(&self, other: &Basis) -> bool {
        match (self, other) {
            (Basis::EmptyBasisLiteral { .. }, Basis::EmptyBasisLiteral { .. }) => true,
            (Basis::BasisLiteral { vecs: v1, .. }, Basis::BasisLiteral { vecs: v2, .. }) => {
                v1.len() == v2.len() && v1.iter().zip(v2).all(|(a, b)| a.approx_equal(&b))
            }
            (Basis::BasisTensor { bases: b1, .. }, Basis::BasisTensor { bases: b2, .. }) => {
                b1.len() == b2.len() && b1.iter().zip(b2).all(|(a, b)| a.approx_equal(&b))
            }
            (
                Basis::ApplyBasisGenerator {
                    basis: b1,
                    generator: g1,
                    ..
                },
                Basis::ApplyBasisGenerator {
                    basis: b2,
                    generator: g2,
                    ..
                },
            ) => {
                (match (g1, g2) {
                    (
                        BasisGenerator::Revolve {
                            v1: v1a, v2: v2a, ..
                        },
                        BasisGenerator::Revolve {
                            v1: v1b, v2: v2b, ..
                        },
                    ) => v1a.approx_equal(&v1b) && v2a.approx_equal(&v2b),
                }) && b1.approx_equal(&b2)
            }
            _ => false,
        }
    }

    /// Returns the source code location for this node.
    pub fn get_dbg(&self) -> Option<DebugLoc> {
        match self {
            Basis::BasisLiteral { dbg, .. } => dbg.clone(),
            Basis::EmptyBasisLiteral { dbg } => dbg.clone(),
            Basis::BasisTensor { dbg, .. } => dbg.clone(),
            Basis::ApplyBasisGenerator { dbg, .. } => dbg.clone(),
        }
    }

    /// Returns number of qubits represented by a basis (`|b|` in the spec)
    /// or None if any basis vector involved is malformed (see Vector::get_dim()).
    pub fn get_dim(&self) -> Option<usize> {
        match self {
            Basis::BasisLiteral { vecs, .. } => {
                if vecs.is_empty() {
                    None
                } else {
                    vecs[0].get_dim().and_then(|first_dim| {
                        if vecs[1..]
                            .iter()
                            .all(|vec| vec.get_dim().is_some_and(|vec_dim| vec_dim == first_dim))
                        {
                            Some(first_dim)
                        } else {
                            None
                        }
                    })
                }
            }

            Basis::EmptyBasisLiteral { .. } => Some(0),

            Basis::BasisTensor { bases, .. } => {
                if bases.len() < 2 {
                    None
                } else {
                    bases
                        .iter()
                        .try_fold(0, |acc, v| v.get_dim().map(|dim| acc + dim))
                }
            }

            Basis::ApplyBasisGenerator {
                basis, generator, ..
            } => basis
                .get_dim()
                .map(|basis_dim| basis_dim + generator.get_dim()),
        }
    }

    /// Returns the zero-indexed qubit positions that are correspond to the
    /// provided atom. This is `Ξva[bv]` in the spec. Returning None indicates
    /// a malformed basis (currently, when basis vectors in a basis literal do
    /// not have matching indices or when [`Vector::get_atom_indices`]
    /// considers any vector malformed).
    pub fn get_atom_indices(&self, atom: VectorAtomKind) -> Option<Vec<usize>> {
        match self {
            Basis::BasisLiteral { vecs, .. } => {
                if vecs.is_empty() {
                    None
                } else {
                    vecs[0]
                        .get_atom_indices(atom)
                        .and_then(|first_vec_indices| {
                            if vecs[1..].iter().all(|vec| {
                                vec.get_atom_indices(atom)
                                    .is_some_and(|vec_indices| vec_indices == first_vec_indices)
                            }) {
                                Some(first_vec_indices)
                            } else {
                                None
                            }
                        })
                }
            }

            Basis::EmptyBasisLiteral { .. } => Some(vec![]),

            Basis::BasisTensor { bases, .. } => {
                if bases.len() < 2 {
                    None
                } else {
                    let mut offset = 0;
                    let mut indices = vec![];
                    for basis in bases {
                        let basis_indices = basis.get_atom_indices(atom)?;
                        for idx in basis_indices {
                            indices.push(idx + offset);
                        }
                        let basis_dim = basis.get_dim()?;
                        offset += basis_dim;
                    }
                    Some(indices)
                }
            }

            // The basis generator extends the basis on the right, and '?' and
            // '_' and atoms are banned inside basis generators right now, so
            // we can pass through the indices of the inner basis.
            Basis::ApplyBasisGenerator { basis, .. } => basis.get_atom_indices(atom),
        }
    }

    /// Returns a version of a basis reference without any '?' or '_' atoms.
    /// Assumes the original basis is well-typed.
    pub fn to_explicit(&self) -> Self {
        self.clone().into_explicit()
    }

    /// Returns a version of an owned basis without any '?' or '_' atoms.
    /// Assumes the original basis is well-typed.
    pub fn into_explicit(self) -> Self {
        rebuild!(Basis, self, into_explicit)
    }

    pub(crate) fn into_explicit_rewriter(self) -> Self {
        match self {
            Basis::BasisLiteral { vecs, dbg } => {
                // This is a little bit overpowered: according to the
                // orthogonality rules, a basis literal can contain at most one
                // lonely '?' or '_'. (That is, {'?'} would typecheck but {'?',
                // '?'+'?' would not.) This code will do the job, though.
                let vecs: Vec<_> = vecs
                    .into_iter()
                    .filter(|vec| !matches!(vec, Vector::VectorUnit { .. }))
                    .collect();
                if vecs.is_empty() {
                    Basis::EmptyBasisLiteral { dbg }
                } else {
                    Basis::BasisLiteral { vecs, dbg }
                }
            }

            Basis::BasisTensor { bases, dbg } => {
                let bases: Vec<_> = bases
                    .into_iter()
                    .filter(|basis| !matches!(basis, Basis::EmptyBasisLiteral { .. }))
                    .collect();
                // Make an assumption that this is well-formed
                if bases.is_empty() {
                    Basis::EmptyBasisLiteral { dbg }
                } else if bases.len() == 1 {
                    bases
                        .into_iter()
                        .next()
                        .expect("bases is empty yet has length 1")
                } else {
                    Basis::BasisTensor { bases, dbg }
                }
            }

            // The '?' and '_' atoms are banned in basis generators
            already_explicit @ (Basis::EmptyBasisLiteral { .. }
            | Basis::ApplyBasisGenerator { .. }) => already_explicit,
        }
    }

    /// Returns an equivalent basis such that:
    /// 1. All vectors are in the form promised by Vector::canonicalize()
    /// 2. No tensor product contains a []
    /// 3. No tensor product contains another tensor product
    /// 4. No basis literal consistingly only of an empty vector, {[]}, are
    ///    replaced with an empty basis literal []
    pub fn canonicalize(self) -> Self {
        rebuild!(Basis, self, canonicalize)
    }

    pub(crate) fn canonicalize_rewriter(self) -> Self {
        match self {
            Basis::BasisLiteral { vecs, dbg } => {
                if vecs.len() == 1 && matches!(&vecs[0], Vector::VectorUnit { .. }) {
                    Basis::EmptyBasisLiteral { dbg }
                } else {
                    Basis::BasisLiteral { vecs, dbg }
                }
            }

            Basis::BasisTensor { bases, dbg } => {
                let bases_canon: Vec<_> = bases
                    .into_iter()
                    .flat_map(|basis| match basis {
                        solo_basis @ (Basis::BasisLiteral { .. }
                        | Basis::ApplyBasisGenerator { .. }) => {
                            vec![solo_basis]
                        }
                        Basis::EmptyBasisLiteral { .. } => vec![],
                        Basis::BasisTensor {
                            bases: inner_bases, ..
                        } => inner_bases,
                    })
                    .collect();
                if bases_canon.is_empty() {
                    Basis::EmptyBasisLiteral { dbg }
                } else if bases_canon.len() == 1 {
                    bases_canon.into_iter().next().expect("basis is empty")
                } else {
                    Basis::BasisTensor {
                        bases: bases_canon,
                        dbg,
                    }
                }
            }

            already_canon @ (Basis::EmptyBasisLiteral { .. }
            | Basis::ApplyBasisGenerator { .. }) => already_canon,
        }
    }

    /// Converts to a vector of basis elements in order.
    pub fn to_vec(self) -> Vec<Basis> {
        match self {
            Basis::EmptyBasisLiteral { .. } => vec![],

            Basis::BasisTensor { bases, .. } => bases,

            other @ (Basis::BasisLiteral { .. } | Basis::ApplyBasisGenerator { .. }) => vec![other],
        }
    }

    /// Converts to a stack of basis elements such that the first element
    /// popped is the leftmost.
    pub fn to_stack(self) -> Vec<Basis> {
        let mut vec = self.to_vec();
        vec.reverse();
        vec
    }

    /// Returns true if this basis fully spans the n-qubit space (if
    /// self.get_dim() == n). Behavior is undefined if this basis is not
    /// well-typed.
    pub fn fully_spans(&self) -> bool {
        match self {
            // A matter of definition, but let's say no. You can't write
            // [] >> [] anyway.
            Basis::EmptyBasisLiteral { .. } => false,

            Basis::BasisLiteral { vecs, .. } => {
                if let Some(dim) = self.get_dim() {
                    // Assumes that all vectors are mutually orthogonal, as
                    // checked by type checking.
                    equals_2_to_the_n(vecs.len(), dim.try_into().unwrap())
                } else {
                    // Conservatively assume not
                    false
                }
            }

            Basis::BasisTensor { bases, .. } => bases.iter().all(Basis::fully_spans),

            Basis::ApplyBasisGenerator {
                basis, generator, ..
            } => basis.fully_spans() && generator.fully_spans(),
        }
    }

    /// Removes phases (except those directly inside superpositions) and sorts
    /// all vectors in basis literals. The resulting basis has the same span
    /// but is not necessarily equivalent. This is intended for use in span
    /// equivalence checking. The basis should be type-checked and
    /// canonicalized first.
    pub fn normalize(self) -> Self {
        rebuild!(Basis, self, normalize)
    }

    pub(crate) fn normalize_rewriter(self) -> Self {
        match self {
            Basis::BasisLiteral { mut vecs, dbg } => {
                vecs.sort();
                Basis::BasisLiteral { vecs, dbg }
            }

            // Pass through basis generators unchanged because they should be
            // treated as a giant inseparable basis
            already_normal @ (Basis::EmptyBasisLiteral { .. }
            | Basis::ApplyBasisGenerator { .. }
            | Basis::BasisTensor { .. }) => already_normal,
        }
    }
}

impl fmt::Display for Basis {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Basis::BasisLiteral { vecs, .. } => {
                write!(f, "{{")?;
                for (i, vec) in vecs.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", vec)?;
                }
                write!(f, "}}")
            }
            Basis::EmptyBasisLiteral { .. } => write!(f, "{{}}"),
            Basis::BasisTensor { bases, .. } => {
                for (i, b) in bases.iter().enumerate() {
                    if i > 0 {
                        write!(f, "*")?;
                    }
                    write!(f, "({})", b)?;
                }
                Ok(())
            }
            Basis::ApplyBasisGenerator {
                basis, generator, ..
            } => write!(f, "({}) // ({})", basis, generator),
        }
    }
}

impl ToPythonCode for Basis {
    fn fmt_py(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Basis::BasisLiteral { vecs, .. } => {
                write!(f, "{{")?;
                for (i, vec) in vecs.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    vec.fmt_py(f)?;
                }
                write!(f, "}}")
            }
            Basis::EmptyBasisLiteral { .. } => write!(f, "{{}}"),
            Basis::BasisTensor { bases, .. } => {
                for (i, b) in bases.iter().enumerate() {
                    if i > 0 {
                        write!(f, "*")?;
                    }
                    write!(f, "(")?;
                    b.fmt_py(f)?;
                    write!(f, ")")?;
                }
                Ok(())
            }
            Basis::ApplyBasisGenerator {
                basis, generator, ..
            } => {
                write!(f, "(")?;
                basis.fmt_py(f)?;
                write!(f, ") // (")?;
                generator.fmt_py(f)?;
                write!(f, ")")
            }
        }
    }
}
