use crate::{
    ast::{qpu::EmbedKind, RegKind},
    dbg::DebugLoc,
};
use dashu::integer::UBig;

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

/// A function (kernel) definition.
///
/// Example syntax:
/// ```text
/// @qpu[[N]]
/// def get_zero() -> qubit[N]:
///     return '0'**N
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct FunctionDef<S> {
    pub name: String,
    pub args: Vec<(MetaType, String)>,
    pub ret_type: MetaType,
    pub body: Vec<S>,
    pub is_rev: bool,
    pub dbg: Option<DebugLoc>,
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
        /// 0.25*'p' + 0.75*'m'
        /// ```
        NonUniformSuperpos {
            pairs: Vec<(f64, MetaVector)>,
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
            n_bits: usize,
            dbg: Option<DebugLoc>,
        },
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
