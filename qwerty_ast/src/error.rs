use crate::{ast::qpu::VectorAtomKind, dbg::DebugLoc};
use dashu::integer::IBig;
use std::{convert::From, fmt};

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
    NonBitArgToClassicalFunc(String),
    NonBitRetFromClassicalFunc(String),
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
    UnsupportedTilt(String),
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
            TypeErrorKind::NonBitArgToClassicalFunc(arg_ty) => write!(
                f,
                "Arguments to @classical function must be nonempty registers of bits (bit[N]), not {}",
                arg_ty
            ),
            TypeErrorKind::NonBitRetFromClassicalFunc(ret_ty) => write!(
                f,
                "Every @classical function must return a nonempty register of bits (bit[N]), not {}",
                ret_ty
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
            TypeErrorKind::UnsupportedTilt(ty) => {
                write!(f, "Cannot tilt an expression of type {}", ty)
            }
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
    // TODO: remove Malformed
    Malformed,
    IntegerTooBig {
        offender: IBig,
    },
    NegativeInteger {
        offender: IBig,
    },
    DivisionByZero,
    RepeatMustBeOnRightOfPipe,
    Stuck,
    MissingFuncTypeAnnotation,
    TypeError {
        kind: TypeErrorKind,
    },
    CannotInferDimVar {
        dim_var_names: Vec<String>,
    },
    DimVarConflict {
        dim_var_name: String,
        first_val: IBig,
        second_val: IBig,
    },
    CannotInstantiate {
        func_name: String,
    },
    DuplicateQubitSymbolDef {
        sym: char,
    },
    DuplicateBasisAliasDef {
        alias_name: String,
    },
    DuplicateBasisAliasRecDef {
        alias_name: String,
    },
    DuplicateMacroDef {
        macro_name: String,
    },
    UndefinedMacro {
        macro_name: String,
    },
    UndefinedAlias {
        alias_name: String,
    },
    UndefinedQubitSymbol {
        sym: char,
    },
    IllegalQubitSymbolInQubitLiteral,
    WrongMacroKind {
        macro_name: String,
    },
    WrongAliasKind {
        alias_name: String,
    },
    MissingAliasRecursiveStep {
        alias_name: String,
    },
    MacroDoesNotMatch {
        macro_name: String,
    },
    ModNotPowerOf2 {
        bad_divisor: usize,
    },
    InvalidEmbedOperand,
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
            LowerErrorKind::RepeatMustBeOnRightOfPipe => {
                write!(
                    f,
                    concat!(
                        "The repeat construct `(... for i in ...)` must be on the right-hand side ",
                        "of a pipe `|`."
                    ),
                )
            }
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
            LowerErrorKind::CannotInferDimVar { dim_var_names } => {
                write!(f, "Cannot infer the following dimension variables: ")?;
                for (i, dim_var_name) in dim_var_names.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", dim_var_name)?;
                }
                Ok(())
            }
            LowerErrorKind::DimVarConflict {
                dim_var_name,
                first_val,
                second_val,
            } => {
                write!(
                    f,
                    "Conflict when inferring dimension variable {}: value {} versus value {}",
                    dim_var_name, first_val, second_val
                )
            }
            LowerErrorKind::CannotInstantiate { func_name } => {
                write!(f, "Cannot instantiate function {}", func_name)
            }
            LowerErrorKind::DuplicateQubitSymbolDef { sym } => {
                write!(f, "Qubit symbol {} is already defined", sym)
            }
            LowerErrorKind::DuplicateBasisAliasDef { alias_name } => {
                write!(f, "Basis alias {} is already defined", alias_name)
            }
            LowerErrorKind::DuplicateBasisAliasRecDef { alias_name } => {
                write!(
                    f,
                    "Recursive step of recursive basis alias {} is already defined",
                    alias_name
                )
            }
            LowerErrorKind::DuplicateMacroDef { macro_name } => {
                write!(f, "Macro {} is already defined", macro_name)
            }
            LowerErrorKind::UndefinedMacro { macro_name } => {
                write!(f, "Macro {} is not defined", macro_name)
            }
            LowerErrorKind::UndefinedAlias { alias_name } => {
                write!(f, "Alias {} is not defined", alias_name)
            }
            LowerErrorKind::UndefinedQubitSymbol { sym } => {
                write!(f, "Qubit symbol {} is not defined", sym)
            }
            LowerErrorKind::IllegalQubitSymbolInQubitLiteral => {
                write!(f, "Illegal qubit symbol in qubit literal")
            }
            LowerErrorKind::WrongMacroKind { macro_name } => {
                write!(f, "Macro {} is the wrong kind", macro_name)
            }
            LowerErrorKind::WrongAliasKind { alias_name } => {
                write!(f, "Alias {} is the wrong kind", alias_name)
            }
            LowerErrorKind::MissingAliasRecursiveStep { alias_name } => {
                write!(
                    f,
                    "Recursive alias {} is missing the recursive step",
                    alias_name
                )
            }
            LowerErrorKind::MacroDoesNotMatch { macro_name } => {
                write!(f, "Macro {} does not match input", macro_name)
            }
            LowerErrorKind::ModNotPowerOf2 { bad_divisor } => {
                write!(
                    f,
                    "Modulus divisor must be a power of 2, not {}",
                    bad_divisor
                )
            }
            LowerErrorKind::InvalidEmbedOperand => {
                write!(f, "Embed operand must be the name of a classical function")
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct LowerError {
    pub kind: LowerErrorKind,
    pub dbg: Option<DebugLoc>,
}

/// This is is useful when you call something that can return `Err(TypeError)`
/// inside a function/method that returns a `Result<T, LowerError>`.
impl From<TypeError> for LowerError {
    fn from(err: TypeError) -> Self {
        LowerError {
            kind: LowerErrorKind::TypeError { kind: err.kind },
            dbg: err.dbg,
        }
    }
}
