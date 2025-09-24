//! Defines the Qwerty Abstract Syntax Tree (AST) and helpful methods for
//! manipulating them (canonicalizing, getting their dimension, etc.)

use crate::dbg::DebugLoc;
use dashu::integer::UBig;
use std::cmp::Ordering;
use std::fmt;

// ----- Types -----

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RegKind {
    /// Classical bit register
    Bit,
    /// Quantum bit register
    Qubit,
    /// A basis, represented as a register for implementation convenience
    Basis,
}

impl fmt::Display for RegKind {
    /// Returns a representation of a type that matches the syntax for the
    /// Python DSL.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RegKind::Bit => write!(f, "bit"),
            RegKind::Qubit => write!(f, "qubit"),
            RegKind::Basis => write!(f, "basis"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Type {
    FuncType { in_ty: Box<Type>, out_ty: Box<Type> },
    RevFuncType { in_out_ty: Box<Type> },
    RegType { elem_ty: RegKind, dim: usize },
    TupleType { tys: Vec<Type> },
    UnitType,
}

impl Type {
    pub fn tuple(tys: Vec<Type>) -> Result<Type, String> {
        if tys.len() < 2 {
            Err(format!(
                "TupleType must contain at least 2 types, found {}",
                tys.len()
            ))
        } else {
            Ok(Type::TupleType { tys })
        }
    }

    /// Returns true if this is a linear type, i.e., must be used exactly once.
    pub fn is_linear(&self) -> bool {
        matches!(self, Type::RegType { elem_ty: RegKind::Qubit, dim } if *dim > 0)
    }

    /// Helper for creating a BitRegType (classical bit register)
    pub fn bit_reg(dim: usize) -> Type {
        Type::RegType {
            elem_ty: RegKind::Bit,
            dim,
        }
    }

    /// Helper for creating a QubitRegType (quantum qubit register)
    pub fn qubit_reg(dim: usize) -> Type {
        Type::RegType {
            elem_ty: RegKind::Qubit,
            dim,
        }
    }

    /// Helper for creating a BasisRegType (quantum basis register)
    pub fn basis_reg(dim: usize) -> Type {
        Type::RegType {
            elem_ty: RegKind::Basis,
            dim,
        }
    }
}

impl fmt::Display for Type {
    /// Returns a representation of a type that matches the syntax for the
    /// Python DSL.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Type::FuncType { in_ty, out_ty } => match (&**in_ty, &**out_ty) {
                (
                    Type::RegType {
                        elem_ty: in_elem_ty,
                        dim: in_dim,
                    },
                    Type::RegType {
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
            Type::RevFuncType { in_out_ty } => match &**in_out_ty {
                Type::RegType {
                    elem_ty: RegKind::Qubit,
                    dim,
                } => write!(f, "rev_qfunc[{}]", dim),
                Type::RegType {
                    elem_ty: RegKind::Bit,
                    dim,
                } => write!(f, "rev_bfunc[{}]", dim),
                _ => write!(f, "rev_func[{}]", in_out_ty),
            },
            Type::RegType { elem_ty, dim } => match elem_ty {
                RegKind::Qubit => write!(f, "qubit[{}]", dim),
                RegKind::Bit => write!(f, "bit[{}]", dim),
                RegKind::Basis => write!(f, "basis[{}]", dim),
            },
            Type::TupleType { tys } => {
                write!(f, "(")?;
                for (i, ty) in tys.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", ty)?;
                }
                write!(f, ")")
            }
            Type::UnitType => write!(f, "None"),
        }
    }
}

// ----- Shared Expressions (QPU and Classical) -----

/// See [`qpu::Expr::Variable`] or [`classical::Expr::Variable`].
#[derive(Debug, Clone, PartialEq)]
pub struct Variable {
    pub name: String,
    pub dbg: Option<DebugLoc>,
}

impl fmt::Display for Variable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name)
    }
}

/// See [`qpu::Expr::BitLiteral`] or [`classical::Expr::BitLiteral`].
#[derive(Debug, Clone, PartialEq)]
pub struct BitLiteral {
    pub val: UBig,
    pub n_bits: usize,
    pub dbg: Option<DebugLoc>,
}

impl fmt::Display for BitLiteral {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "bit[{}](0b{:b})", self.n_bits, self.val)
    }
}

// ----- Expressions (QPU) -----

pub mod qpu;

// ----- Expressions (Classical) -----

pub mod classical;

// ----- Statements (Generic over Expression Type) -----

// Structs for Stmt variants
/// See [`Stmt::Expr`].
#[derive(Debug, Clone, PartialEq)]
pub struct StmtExpr<E> {
    pub expr: E,
    pub dbg: Option<DebugLoc>,
}

/// See [`Stmt::Assign`].
#[derive(Debug, Clone, PartialEq)]
pub struct Assign<E> {
    pub lhs: String,
    pub rhs: E,
    pub dbg: Option<DebugLoc>,
}

/// See [`Stmt::UnpackAssign`].
#[derive(Debug, Clone, PartialEq)]
pub struct UnpackAssign<E> {
    pub lhs: Vec<String>,
    pub rhs: E,
    pub dbg: Option<DebugLoc>,
}

/// See [`Stmt::Return`].
#[derive(Debug, Clone, PartialEq)]
pub struct Return<E> {
    pub val: E,
    pub dbg: Option<DebugLoc>,
}

/// Allows creating a trivial version of a statement or expression.
/// (That is, one that does nothing.)
pub trait Trivializable {
    fn trivial(dbg: Option<DebugLoc>) -> Self;
}

#[derive(Debug, Clone, PartialEq)]
pub enum Stmt<E> {
    /// An expression statement. Example syntax:
    /// ```text
    /// f(x)
    /// ```
    Expr(StmtExpr<E>),

    /// An assignment statement. Example syntax:
    /// ```text
    /// q = '0'
    /// ```
    Assign(Assign<E>),

    /// A register-unpacking assignment statement. Example syntax:
    /// ```text
    /// q1, q2 = '01'
    /// ```
    UnpackAssign(UnpackAssign<E>),

    /// A return statement. Example syntax:
    /// ```text
    /// return q
    /// ```
    Return(Return<E>),
}

impl<E: Trivializable> Trivializable for Stmt<E> {
    /// Trivial statement is an expression statement containing a trivial
    /// expression.
    fn trivial(dbg: Option<DebugLoc>) -> Self {
        Self::Expr(StmtExpr {
            expr: E::trivial(dbg.clone()),
            dbg,
        })
    }
}

impl<E: fmt::Display> fmt::Display for Stmt<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Stmt::Expr(StmtExpr { expr, .. }) => write!(f, "{}", expr),
            Stmt::Assign(Assign { lhs, rhs, .. }) => write!(f, "{} = {}", lhs, rhs),
            Stmt::UnpackAssign(UnpackAssign { lhs, rhs, .. }) => {
                for (i, name) in lhs.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", name)?;
                }
                write!(f, " = {}", rhs)
            }
            Stmt::Return(Return { val, .. }) => write!(f, "return {}", val),
        }
    }
}

// ----- Functions (Generic over Expression Type) -----

/// A function (kernel) definition.
///
/// Example syntax:
/// ```text
/// @qpu
/// def get_zero() -> qubit:
///     return '0'
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct FunctionDef<E> {
    pub name: String,
    pub args: Vec<(Type, String)>,
    pub ret_type: Type,
    pub body: Vec<Stmt<E>>,
    pub is_rev: bool,
    pub dbg: Option<DebugLoc>,
}

impl<E> FunctionDef<E> {
    pub fn new(
        name: String,
        args: Vec<(Type, String)>,
        ret_type: Type,
        body: Vec<Stmt<E>>,
        is_rev: bool, // passed from the parser
        dbg: Option<DebugLoc>,
    ) -> Self {
        Self {
            name,
            args,
            ret_type,
            body,
            is_rev,
            dbg,
        }
    }

    /// Returns true if the function was explicitly annotated as reversible.
    pub fn is_reversible(&self) -> bool {
        self.is_rev
    }

    /// Reconstructs the full function type (FuncType or RevFuncType) from the
    /// FunctionDef's arguments, value return type, and reversibility flag.
    pub fn get_type(&self) -> Type {
        let in_ty = if self.args.is_empty() {
            Type::UnitType
        } else if self.args.len() == 1 {
            self.args[0].0.clone()
        } else {
            let arg_types: Vec<Type> = self.args.iter().map(|(ty, _)| ty.clone()).collect();
            Type::tuple(arg_types)
                .expect("Function with multiple arguments must form a valid TupleType")
        };

        if self.is_rev {
            Type::RevFuncType {
                in_out_ty: Box::new(self.ret_type.clone()),
            }
        } else {
            Type::FuncType {
                in_ty: Box::new(in_ty),
                out_ty: Box::new(self.ret_type.clone()),
            }
        }
    }
}

impl<E: std::fmt::Display> std::fmt::Display for FunctionDef<E> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "fn {}(", self.name)?;
        for (i, (ty, arg_name)) in self.args.iter().enumerate() {
            write!(f, "{arg_name} {ty}")?;
            if i + 1 != self.args.len() {
                write!(f, ", ")?;
            }
        }
        write!(f, ") -> {}", self.ret_type)?;
        if self.is_rev {
            write!(f, " rev")?;
        }
        writeln!(f, " {{")?;

        // Print body, one stmt per line with indentation
        for stmt in &self.body {
            writeln!(f, "    {stmt}")?;
        }

        write!(f, "}}")?;

        if let Some(dbg) = &self.dbg {
            writeln!(f, "dbg: {dbg}")?;
        }
        Ok(())
    }
}

// ----- Program (Top-Level Function Container) -----

#[derive(Debug, Clone, PartialEq)]
pub enum Func {
    /// A `@qpu` kernel.
    Qpu(FunctionDef<qpu::Expr>),
    /// A `@classical` function.
    Classical(FunctionDef<classical::Expr>),
}

impl Func {
    /// Returns the name of this function.
    pub fn get_name(&self) -> String {
        match self {
            Func::Qpu(func_def) => func_def.name.to_string(),
            Func::Classical(func_def) => func_def.name.to_string(),
        }
    }

    /// Constructs the type of this function.
    pub fn get_type(&self) -> Type {
        match self {
            Func::Qpu(func_def) => func_def.get_type(),
            Func::Classical(func_def) => func_def.get_type(),
        }
    }
}

impl std::fmt::Display for Func {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Func::Qpu(def) => write!(f, "@qpu {def}"),
            Func::Classical(def) => write!(f, "@classical {def}"),
        }
    }
}

/// The top-level node in a Qwerty program that holds all function defintiions.
///
/// In the current implementation, there is only one of these per Python
/// interpreter.
#[derive(Debug, Clone, PartialEq)]
pub struct Program {
    pub funcs: Vec<Func>,
    pub dbg: Option<DebugLoc>,
}

impl std::fmt::Display for Program {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for func in &self.funcs {
            writeln!(f, "{func}")?;
        }
        if let Some(dbg) = &self.dbg {
            writeln!(f, "{dbg}")?;
        }
        Ok(())
    }
}

// ----- Miscellaneous math for angles and bits -----

/// Tolerance for floating point comparison
const ATOL: f64 = 1e-12;

/// Returns a canon form of this angle in the range [0.0, 360.0).
pub fn canon_angle(angle_deg: f64) -> f64 {
    // angle_deg % 360 could be negative. This will always be nonnegative.
    angle_deg.rem_euclid(360.0)
}

/// Returns true if two angles are approximately equal.
pub fn angles_are_approx_equal(angle_deg1: f64, angle_deg2: f64) -> bool {
    (angle_deg1 - angle_deg2).abs() < ATOL
}

/// Returns true if an angle is approximately 0 degrees.
pub fn angle_is_approx_zero(angle_deg: f64) -> bool {
    angles_are_approx_equal(angle_deg, 0.0)
}

/// Returns true iff the two phases are the same angle (up to a multiple of 360)
pub fn in_phase(angle_deg1: f64, angle_deg2: f64) -> bool {
    let diff = angle_deg1 - angle_deg2;
    let mod360 = canon_angle(diff);
    mod360.abs() < ATOL
}

/// Returns true iff the two phases differ by 180 degrees (up to a multiple of
/// 360)
pub fn anti_phase(angle_deg1: f64, angle_deg2: f64) -> bool {
    let diff = angle_deg1 - angle_deg2;
    let mod360 = canon_angle(diff);
    (mod360 - 180.0).abs() < ATOL
}

/// A total ordering of angles that accounts for floats being noisy
pub fn angle_approx_total_cmp(angle_deg1: f64, angle_deg2: f64) -> Ordering {
    if angles_are_approx_equal(angle_deg1, angle_deg2) {
        Ordering::Equal
    } else if angle_deg1 < angle_deg2 {
        Ordering::Less
    } else {
        // angle_deg1 > angle_deg2
        Ordering::Greater
    }
}

/// Returns `None` if `num` is not a power of two or `Some(log2(num))`
/// otherwise.
pub fn try_log2(num: usize) -> Option<usize> {
    if num.count_ones() == 1 {
        Some(num.trailing_zeros() as usize)
    } else {
        None
    }
}

/// Returns true iff num == 2**n.
pub fn equals_2_to_the_n(num: usize, n: usize) -> bool {
    try_log2(num).is_some_and(|pow| pow == n)
}

/// Try to convert a `usize` into an `f64`. Returns `None` if precision was
/// lost in the conversion. In the future, if we need more of these kinds of
/// conversions, we can use the `conv2` crate. But for now, we use this simple
/// code from [a reddit post](https://www.reddit.com/r/learnrust/comments/df65g9/comment/f31d3ho/).
pub fn usize_try_into_angle(num: usize) -> Option<f64> {
    let ret = num as f64;
    if ret as usize == num { Some(ret) } else { None }
}

/// Return a [`UBig`] with value `~(-1 << n)`.
pub fn ubig_with_n_lower_bits_set(n: usize) -> UBig {
    let mut ret = UBig::ZERO;
    for i in (0..n).rev() {
        ret.set_bit(i)
    }
    ret
}

//
// ─── UNIT TESTS ─────────────────────────────────────────────────────────────────
//

#[cfg(test)]
mod test_ast_basis;
#[cfg(test)]
mod test_ast_vec_qlit;
