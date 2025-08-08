use crate::{
    ast::{
        self,
        classical::{BinaryOpKind, UnaryOpKind},
        qpu::EmbedKind,
        RegKind,
    },
    dbg::DebugLoc,
};
use dashu::integer::UBig;
use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub enum DimExpr {
    /// Dimension variable. Example syntax:
    /// ```text
    /// N
    /// ```
    DimVar { name: String, dbg: Option<DebugLoc> },

    /// A constant dimension variable value. Example syntax:
    /// ```text
    /// 2
    /// ```
    DimConst { val: UBig, dbg: Option<DebugLoc> },

    /// Sum of dimension variable values. Example syntax:
    /// ```text
    /// N + 1
    /// ```
    DimSum {
        left: Box<DimExpr>,
        right: Box<DimExpr>,
        dbg: Option<DebugLoc>,
    },

    /// Product of dimension variable values. Example syntax:
    /// ```text
    /// 2*N
    /// ```
    DimProd {
        left: Box<DimExpr>,
        right: Box<DimExpr>,
        dbg: Option<DebugLoc>,
    },

    /// Negation of a dimension variable value. Example syntax:
    /// ```text
    /// -N
    /// ```
    DimNeg {
        val: Box<DimExpr>,
        dbg: Option<DebugLoc>,
    },
}

impl fmt::Display for DimExpr {
    /// Returns a representation of a dimension variable expression that
    /// matches the syntax in the Python DSL.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DimExpr::DimVar { name, .. } => write!(f, "{}", name),
            DimExpr::DimConst { val, .. } => write!(f, "{}", val),
            DimExpr::DimSum { left, right, .. } => write!(f, "({})+({})", left, right),
            DimExpr::DimProd { left, right, .. } => write!(f, "({})*({})", left, right),
            DimExpr::DimNeg { val, .. } => write!(f, "-({})", val),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum MetaType {
    FuncType {
        in_ty: Box<MetaType>,
        out_ty: Box<MetaType>,
    },
    RevFuncType {
        in_out_ty: Box<MetaType>,
    },
    RegType {
        elem_ty: RegKind,
        dim: DimExpr,
    },
    TupleType {
        tys: Vec<MetaType>,
    },
    UnitType,
}

// TODO: Don't duplicate this with ast.rs
impl fmt::Display for MetaType {
    /// Returns a representation of a type that matches the syntax for the
    /// Python DSL.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MetaType::FuncType { in_ty, out_ty } => match (&**in_ty, &**out_ty) {
                (
                    MetaType::RegType {
                        elem_ty: in_elem_ty,
                        dim: in_dim,
                    },
                    MetaType::RegType {
                        elem_ty: out_elem_ty,
                        dim: out_dim,
                    },
                ) if *in_elem_ty != RegKind::Basis && *out_elem_ty != RegKind::Basis => {
                    let prefix = match (in_elem_ty, out_elem_ty) {
                        (RegKind::Qubit, RegKind::Qubit) => "q",
                        (RegKind::Qubit, RegKind::Bit) => "qb",
                        (RegKind::Bit, RegKind::Qubit) => "bq",
                        (RegKind::Bit, RegKind::Bit) => "b",
                        (RegKind::Basis, _) | (_, RegKind::Basis) => {
                            unreachable!("bases cannot be function arguments/results")
                        }
                    };
                    write!(f, "{}func[", prefix)?;
                    if in_elem_ty == out_elem_ty && in_dim == out_dim {
                        write!(f, "{}]", in_dim)
                    } else {
                        write!(f, "{},{}]", in_dim, out_dim)
                    }
                }
                _ => write!(f, "func[{},{}]", in_ty, out_ty),
            },
            MetaType::RevFuncType { in_out_ty } => match &**in_out_ty {
                MetaType::RegType {
                    elem_ty: RegKind::Qubit,
                    dim,
                } => write!(f, "rev_qfunc[{}]", dim),
                MetaType::RegType {
                    elem_ty: RegKind::Bit,
                    dim,
                } => write!(f, "rev_bfunc[{}]", dim),
                _ => write!(f, "rev_func[{}]", in_out_ty),
            },
            MetaType::RegType { elem_ty, dim } => match elem_ty {
                RegKind::Qubit => write!(f, "qubit[{}]", dim),
                RegKind::Bit => write!(f, "bit[{}]", dim),
                RegKind::Basis => write!(f, "basis[{}]", dim),
            },
            MetaType::TupleType { tys } => {
                write!(f, "(")?;
                for (i, ty) in tys.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", ty)?;
                }
                write!(f, ")")
            }
            MetaType::UnitType => write!(f, "None"),
        }
    }
}

/// A function (kernel) definition.
///
/// Example syntax:
/// ```text
/// @qpu[[N]]
/// def get_zero() -> qubit[N]:
///     return '0'**N
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct MetaFunctionDef<S> {
    pub name: String,
    pub args: Vec<(MetaType, String)>,
    pub ret_type: MetaType,
    pub body: Vec<S>,
    pub is_rev: bool,
    pub dim_vars: Vec<String>,
    pub dbg: Option<DebugLoc>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum MetaFunc {
    /// A `@qpu` kernel.
    Qpu(MetaFunctionDef<qpu::MetaStmt>),
    /// A `@classical` function.
    Classical(MetaFunctionDef<classical::MetaStmt>),
}

/// The top-level node in a Qwerty program that holds all function defintiions.
///
/// In the current implementation, there is only one of these per Python
/// interpreter.
#[derive(Debug, Clone, PartialEq)]
pub struct MetaProgram {
    pub funcs: Vec<MetaFunc>,
    pub dbg: Option<DebugLoc>,
}

impl MetaProgram {
    pub fn expand(&self) -> ast::Program {
        todo!("MetaProgram::expand()")
    }
}

pub mod qpu {
    use super::*;

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
}

pub mod classical {
    use super::*;

    #[derive(Debug, Clone, PartialEq)]
    pub enum MetaExpr {
        /// A variable name used in an expression. Example syntax:
        /// ```text
        /// my_var
        /// ```
        Variable { name: String, dbg: Option<DebugLoc> },

        /// A unary bitwise operation. Example syntax:
        /// ```text
        /// ~x
        /// ```
        UnaryOp {
            kind: UnaryOpKind,
            val: Box<MetaExpr>,
            dbg: Option<DebugLoc>,
        },

        /// A binary bitwise operation. Example syntax:
        /// ```text
        /// x & y
        /// ```
        BinaryOp {
            kind: BinaryOpKind,
            left: Box<MetaExpr>,
            right: Box<MetaExpr>,
            dbg: Option<DebugLoc>,
        },

        /// A logical reduction operation. Example syntax:
        /// ```text
        /// x.xor_reduce()
        /// ```
        ReduceOp {
            kind: BinaryOpKind,
            val: Box<MetaExpr>,
            dbg: Option<DebugLoc>,
        },

        /// A constant bit register. Example syntax:
        /// ```text
        /// bit[N](0b0)
        /// ```
        BitLiteral {
            val: UBig,
            n_bits: DimExpr,
            dbg: Option<DebugLoc>,
        },
    }

    // TODO: don't duplicate this code with classical.rs
    impl fmt::Display for MetaExpr {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            match self {
                MetaExpr::Variable { name, .. } => write!(f, "{}", name),
                MetaExpr::UnaryOp { kind, val, .. } => {
                    let kind_str = match kind {
                        UnaryOpKind::Not => "~",
                    };
                    write!(f, "{}({})", kind_str, *val)
                }
                MetaExpr::BinaryOp {
                    kind, left, right, ..
                } => {
                    let kind_str = match kind {
                        BinaryOpKind::And => "&",
                        BinaryOpKind::Or => "|",
                        BinaryOpKind::Xor => "^",
                    };
                    write!(f, "({}) {} ({})", *left, kind_str, *right)
                }
                MetaExpr::ReduceOp { kind, val, .. } => {
                    let kind_str = match kind {
                        BinaryOpKind::And => "and",
                        BinaryOpKind::Or => "or",
                        BinaryOpKind::Xor => "xor",
                    };
                    write!(f, "({}).{}_reduce()", kind_str, *val)
                }
                MetaExpr::BitLiteral { val, n_bits, .. } => {
                    write!(f, "bit[{}](0b{:b})", n_bits, val)
                }
            }
        }
    }

    #[derive(Debug, Clone, PartialEq)]
    pub enum MetaStmt {
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

    // TODO: don't duplicate with ast.rs
    impl fmt::Display for MetaStmt {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            match self {
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
}
