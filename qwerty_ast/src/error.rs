use crate::{ast::qpu::VectorAtomKind, dbg::DebugLoc};
use dashu::integer::IBig;
use std::fmt;

// For type checking

#[derive(Debug, Clone, PartialEq)]
pub enum TypeErrorKind {
    UndefinedVariable(String),
    RedefinedVariable(String),
    UninitializedVariable(String),
    ImmutableAssignment(String),
    MismatchedTypes { expected: String, found: String },
    WrongArity { expected: usize, found: usize },
    NotCallable(String),
    InvalidType(String),
    InvalidOperation { op: String, ty: String },
    TypeInferenceFailure,
    EmptyLiteral,
    DimMismatch,
    InvalidFloat { float: f64 },
    ReturnNotLastStatement,
    ReturnOutsideFunction,
    InvalidIntermediateComputation,
    // Quantum-specific errors:
    LinearVariableUsedTwice(String),
    LinearVariableUnused(String),
    LinearVariableUseMismatch(String),
    MismatchedAtoms { atom_kind: VectorAtomKind },
    InvalidBasis,
    SpanMismatch,
    DoesNotFullySpan,
    UnsupportedTensorProduct,
    NotOrthogonal { left: String, right: String },
    InvalidQubitOperation(String),
    NonReversibleOperationInReversibleFunction(String),
    ProbabilitiesDoNotSumToOne,
}

impl fmt::Display for TypeErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TypeErrorKind::UndefinedVariable(var) => write!(
                f,
                "Variable '{var}' is used but not declared or is out of scope."
            ),
            TypeErrorKind::RedefinedVariable(var) => {
                write!(f, "Variable '{var}' is already defined.")
            }
            TypeErrorKind::UninitializedVariable(var) => {
                write!(f, "Variable '{var}' is used before being assigned a value.")
            }
            TypeErrorKind::ImmutableAssignment(var) => write!(
                f,
                "Attempt to assign to a variable '{var}' that is immutable."
            ),
            TypeErrorKind::MismatchedTypes { expected, found } => write!(
                f,
                "Value type '{found}' does not match the expected type '{expected}'."
            ),
            TypeErrorKind::WrongArity { expected, found } => write!(
                f,
                "Number of arguments in a call {found} does not match the expected number {expected}."
            ),
            TypeErrorKind::NotCallable(what) => {
                write!(f, "Attempt to call '{what}' that is not a function.")
            }
            TypeErrorKind::InvalidType(ty) => {
                write!(f, "Type '{ty}' is not supported or is malformed.")
            }
            TypeErrorKind::InvalidOperation { op, ty } => write!(
                f,
                "Operation '{op}' is not valid for the given type '{ty}'."
            ),
            // TODO: needs more details
            TypeErrorKind::TypeInferenceFailure => {
                write!(f, "Compiler could not determine the type.")
            }
            TypeErrorKind::EmptyLiteral => write!(f, "Literal is unexpectedly empty."),
            // TODO: needs more details
            TypeErrorKind::DimMismatch => write!(f, "Dimension mismatch."),
            TypeErrorKind::InvalidFloat { float } => write!(
                f,
                "The float {float} is invalid. Floats used in Qwerty must be finite and not NaN."
            ),
            TypeErrorKind::ReturnNotLastStatement => {
                write!(f, "Functions must end with a return statement.")
            }
            TypeErrorKind::ReturnOutsideFunction => write!(
                f,
                "The return statement can only be written inside a function."
            ),
            TypeErrorKind::InvalidIntermediateComputation => write!(
                f,
                "Qubit References should only be intermediate computations."
            ),
            TypeErrorKind::LinearVariableUsedTwice(var) => write!(
                f,
                "The qubit register variable '{var}' must be used exactly once, but a second usage was encountered."
            ),
            TypeErrorKind::LinearVariableUnused(var) => write!(
                f,
                "The qubit register variable '{var}' must be used exactly once, but it was never used."
            ),
            TypeErrorKind::LinearVariableUseMismatch(var) => write!(
                f,
                "The qubit register variable '{var}' is used in one branch of a classical conditional and not the other branch."
            ),
            TypeErrorKind::MismatchedAtoms { atom_kind } => write!(
                f,
                "The location of {atom_kind} does not match between both bases."
            ),
            // TODO: needs more details (maybe?)
            TypeErrorKind::InvalidBasis => write!(f, "Invalid basis."),
            // TODO: need more detail
            TypeErrorKind::SpanMismatch => write!(f, "Bases do not have matching spans."),
            TypeErrorKind::DoesNotFullySpan => {
                write!(f, "Basis does not span the entire n-qubit space.")
            }
            // TODO: need more detail. not quite a mismatch but something like
            //       f*f where f : (qubit[1]->bit[1]) -> unit. The types match
            //       but that's not a valid tensor product.
            TypeErrorKind::UnsupportedTensorProduct => write!(f, "Unsupported tensor product"),
            // TODO: say something more specific than "constructs"?
            TypeErrorKind::NotOrthogonal { left, right } => write!(
                f,
                "The constructs `{left}` and `{right}` are not orthogonal."
            ),
            TypeErrorKind::InvalidQubitOperation(op) => {
                write!(f, "Invalid operation '{op}' performed on a qubit.")
            }
            TypeErrorKind::NonReversibleOperationInReversibleFunction(func_name) => write!(
                f,
                "Function '{}' is declared @reversible but contains non-reversible operations.",
                func_name
            ),
            TypeErrorKind::ProbabilitiesDoNotSumToOne => {
                write!(f, "Probabilities of superposition do not add up to 100%.")
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct TypeError {
    pub kind: TypeErrorKind,
    pub dbg: Option<DebugLoc>,
}

// For extracting a plain AST from a meta-AST

#[derive(Debug, Clone, PartialEq)]
pub enum LowerErrorKind {
    // TODO: add more details
    NotFullyFolded,
    Malformed,
    IntegerTooBig { offender: IBig },
    NegativeInteger { offender: IBig },
    DivisionByZero,
    Stuck,
    MissingFuncTypeAnnotation,
    TypeError { kind: TypeErrorKind },
}

impl fmt::Display for LowerErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LowerErrorKind::NotFullyFolded => write!(
                f,
                "Cannot extract since metaQwerty constructs are not fully expanded."
            ),
            LowerErrorKind::Malformed => write!(
                f,
                concat!(
                    "metaQwerty is malformed in a way that should be caught by its ",
                    "type checker. This is a compiler bug."
                )
            ),
            LowerErrorKind::IntegerTooBig { offender } => {
                write!(
                    f,
                    concat!(
                        "Computed dimension variable expression is {}, ",
                        "which is too big to fit in a native integer"
                    ),
                    offender
                )
            }
            LowerErrorKind::NegativeInteger { offender } => {
                write!(
                    f,
                    "Computed dimension variable expression is {}, a negative value",
                    offender
                )
            }
            LowerErrorKind::DivisionByZero => write!(f, "Division by zero"),
            LowerErrorKind::Stuck => write!(
                f,
                concat!(
                    "metaQwerty lowering is stuck. ",
                    "Some dimension variables may need to be provided explicitly."
                )
            ),
            LowerErrorKind::MissingFuncTypeAnnotation => {
                write!(f, "Function type annotation missing")
            }
            LowerErrorKind::TypeError { kind } => write!(f, "{}", kind),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct LowerError {
    pub kind: LowerErrorKind,
    pub dbg: Option<DebugLoc>,
}
