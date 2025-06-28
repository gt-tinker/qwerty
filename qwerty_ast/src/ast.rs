/*
 * QWERTY Programming Language Compiler
 * Abstract Syntax Tree (AST) Definitions
 *
 * This module defines the Abstract Syntax Tree (AST) structures
 * used for parsing and representing QWERTY programs.
 *
 * Version: 1.0
 */

use crate::dbg::DebugLoc;

// ----- Types -----

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Type {
    FuncType { in_ty: Box<Type>, out_ty: Box<Type> },
    RevFuncType { in_out_ty: Box<Type> },
    RegType { elem_ty: RegKind, dim: u64 },
    UnitType,
}

// ----- Registers -----

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RegKind {
    Bit,   // Classical bit register
    Qubit, // Quantum bit register
    Basis, // Register for basis states
}

// ----- Qubit Literals -----

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
}

// ----- Vector -----

#[derive(Debug, Clone, PartialEq)]
pub enum Vector {
    ZeroVector {
        dbg: Option<DebugLoc>,
    },
    OneVector {
        dbg: Option<DebugLoc>,
    },
    PadVector {
        dbg: Option<DebugLoc>,
    },
    TargetVector {
        dbg: Option<DebugLoc>,
    },
    VectorTilt {
        q: Box<QLit>,
        angle_deg: f64,
        dbg: Option<DebugLoc>,
    },
    UniformVectorSuperpos {
        q1: Box<QLit>,
        q2: Box<QLit>,
        dbg: Option<DebugLoc>,
    },
    VectorTensor {
        qs: Vec<QLit>,
        dbg: Option<DebugLoc>,
    },
}

// ----- Basis -----

#[derive(Debug, Clone, PartialEq)]
pub enum Basis {
    BasisLiteral {
        vecs: Vec<Vector>,
        dbg: Option<DebugLoc>,
    },
    EmptyBasisLiteral {
        dbg: Option<DebugLoc>,
    },
    BasisTensor {
        bases: Vec<Basis>,
        dbg: Option<DebugLoc>,
    },
}

// ----- Expressions -----

#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    Variable {
        name: String,
        dbg: Option<DebugLoc>,
    },
    UnitLiteral {
        dbg: Option<DebugLoc>,
    },
    Adjoint {
        func: Box<Expr>,
        dbg: Option<DebugLoc>,
    },
    Pipe {
        lhs: Box<Expr>,
        rhs: Box<Expr>,
        dbg: Option<DebugLoc>,
    },
    Measure {
        basis: Basis,
        dbg: Option<DebugLoc>,
    },
    Discard {
        dbg: Option<DebugLoc>,
    },
    Tensor {
        vals: Vec<Expr>,
        dbg: Option<DebugLoc>,
    },
    BasisTranslation {
        bin: Basis,
        bout: Basis,
        dbg: Option<DebugLoc>,
    },
    Predicated {
        then_func: Box<Expr>,
        else_func: Box<Expr>,
        pred: Basis,
        dbg: Option<DebugLoc>,
    },
    NonUniformSuperpos {
        pairs: Vec<(f64, QLit)>,
        dbg: Option<DebugLoc>,
    },
    Conditional {
        then_expr: Box<Expr>,
        else_expr: Box<Expr>,
        cond: Box<Expr>,
        dbg: Option<DebugLoc>,
    },
    QLit(QLit),
}

// ----- Statements -----

#[derive(Debug, Clone, PartialEq)]
pub enum Stmt {
    Assign {
        lhs: String,
        rhs: Expr,
        dbg: Option<DebugLoc>,
    },
    UnpackAssign {
        lhs: Vec<String>,
        rhs: Expr,
        dbg: Option<DebugLoc>,
    },
    Return {
        val: Expr,
        dbg: Option<DebugLoc>,
    },
}

// ----- Functions -----

#[derive(Debug, Clone, PartialEq)]
pub struct FunctionDef {
    pub name: String,
    pub args: Vec<(Type, String)>,
    pub ret_type: Type,
    pub body: Vec<Stmt>,
    pub dbg: Option<DebugLoc>,
}

// ----- Program -----

#[derive(Debug, Clone, PartialEq)]
pub struct Program {
    pub funcs: Vec<FunctionDef>,
    pub dbg: Option<DebugLoc>,
}
