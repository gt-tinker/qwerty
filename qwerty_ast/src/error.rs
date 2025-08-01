use crate::ast::VectorAtomKind;
use crate::dbg::DebugLoc;
use std::fmt;

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
    ReturnOutsideFunction,
    InvalidIntermediateComputation,
    // Quantum-specific errors:
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
            TypeErrorKind::UndefinedVariable(var) => write!(f, "Variable '{var}' is used but not declared or is out of scope."),
            TypeErrorKind::RedefinedVariable(var) => write!(f, "Variable '{var}' declared more than once in the same scope."),
            TypeErrorKind::UninitializedVariable(var) => write!(f, "Variable '{var}' is used before being assigned a value."),
            TypeErrorKind::ImmutableAssignment(var) => write!(f, "Attempt to assign to a variable '{var}' that is immutable."),
            TypeErrorKind::MismatchedTypes { expected, found } => write!(f, "Value type '{found}' does not match the expected type '{expected}'."),
            TypeErrorKind::WrongArity { expected, found } => write!(f, "Number of arguments in a call {found} does not match the expected number {expected}."),
            TypeErrorKind::NotCallable(what) => write!(f, "Attempt to call '{what}' that is not a function."),
            TypeErrorKind::InvalidType(ty) => write!(f, "Type '{ty}' is not supported or is malformed."),
            TypeErrorKind::InvalidOperation { op, ty } => write!(f, "Operation '{op}' is not valid for the given type '{ty}'."),
            // TODO: needs more details
            TypeErrorKind::TypeInferenceFailure => write!(f, "Compiler could not determine the type."),
            TypeErrorKind::EmptyLiteral => write!(f, "Literal is unexpectedly empty."),
            // TODO: needs more details
            TypeErrorKind::DimMismatch => write!(f, "Dimension mismatch."),
            TypeErrorKind::InvalidFloat { float } => write!(f, "The float {float} is invalid. Floats used in Qwerty must be finite and not NaN."),
            TypeErrorKind::ReturnOutsideFunction => write!(f, "The return statement can only be written inside a function."),
            TypeErrorKind::InvalidIntermediateComputation => write!(f, "Qubit References should only be intermediate computations."),
            TypeErrorKind::MismatchedAtoms { atom_kind } => write!(f, "The location of {atom_kind} does not match between both bases."),
            // TODO: needs more details (maybe?)
            TypeErrorKind::InvalidBasis => write!(f, "Invalid basis."),
            // TODO: need more detail
            TypeErrorKind::SpanMismatch => write!(f, "Bases do not have matching spans."),
            TypeErrorKind::DoesNotFullySpan => write!(f, "Basis does not span the entire n-qubit space."),
            // TODO: need more detail. not quite a mismatch but something like
            //       f*f where f : (qubit[1]->bit[1]) -> unit. The types match
            //       but that's not a valid tensor product.
            TypeErrorKind::UnsupportedTensorProduct => write!(f, "Unsupported tensor product"),
            // TODO: say something more specific than "constructs"?
            TypeErrorKind::NotOrthogonal { left, right } => write!(f, "The constructs `{left}` and `{right}` are not orthogonal."),
            TypeErrorKind::InvalidQubitOperation(op) => write!(f, "Invalid operation '{op}' performed on a qubit."),
            TypeErrorKind::NonReversibleOperationInReversibleFunction(func_name) => write!(
                f,
                "Function '{}' is declared @reversible but contains non-reversible operations.",
                func_name
            ),
            TypeErrorKind::ProbabilitiesDoNotSumToOne => write!(f, "Probabilities of superposition do not add up to 100%."),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct TypeError {
    pub kind: TypeErrorKind,
    pub dbg: Option<DebugLoc>,
}
