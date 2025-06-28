/*
 * QWERTY Programming Language Compiler
 * Abstract Syntax Tree (AST) Definitions
 *
 * This module defines the Abstract Syntax Tree (AST) structures
 * used for parsing and representing QWERTY programs.
 *
 * Version: 1.0
 */

use crate::dbg::DebugInfo;

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
        dbg: Option<DebugInfo>,
    },
    OneQubit {
        dbg: Option<DebugInfo>,
    },
    QubitTilt {
        q: Box<QLit>,
        angle_deg: f64,
        dbg: Option<DebugInfo>,
    },
    UniformSuperpos {
        q1: Box<QLit>,
        q2: Box<QLit>,
        dbg: Option<DebugInfo>,
    },
    QubitTensor {
        qs: Vec<QLit>,
        dbg: Option<DebugInfo>,
    },
}

// ----- Vector -----

#[derive(Debug, Clone, PartialEq)]
pub enum Vector {
    ZeroVector {
        dbg: Option<DebugInfo>,
    },
    OneVector {
        dbg: Option<DebugInfo>,
    },
    PadVector {
        dbg: Option<DebugInfo>,
    },
    TargetVector {
        dbg: Option<DebugInfo>,
    },
    VectorTilt {
        q: Box<QLit>,
        angle_deg: f64,
        dbg: Option<DebugInfo>,
    },
    UniformVectorSuperpos {
        q1: Box<QLit>,
        q2: Box<QLit>,
        dbg: Option<DebugInfo>,
    },
    VectorTensor {
        qs: Vec<QLit>,
        dbg: Option<DebugInfo>,
    },
}

// ----- Basis -----

#[derive(Debug, Clone, PartialEq)]
pub enum Basis {
    BasisLiteral {
        vecs: Vec<Vector>,
        dbg: Option<DebugInfo>,
    },
    EmptyBasisLiteral {
        dbg: Option<DebugInfo>,
    },
    BasisTensor {
        bases: Vec<Basis>,
        dbg: Option<DebugInfo>,
    },
}

// ----- Expressions -----

#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    Variable {
        name: String,
        dbg: Option<DebugInfo>,
    },
    UnitLiteral {
        dbg: Option<DebugInfo>,
    },
    Adjoint {
        func: Box<Expr>,
        dbg: Option<DebugInfo>,
    },
    Pipe {
        lhs: Box<Expr>,
        rhs: Box<Expr>,
        dbg: Option<DebugInfo>,
    },
    Measure {
        basis: Basis,
        dbg: Option<DebugInfo>,
    },
    Discard {
        dbg: Option<DebugInfo>,
    },
    Tensor {
        vals: Vec<Expr>,
        dbg: Option<DebugInfo>,
    },
    BasisTranslation {
        bin: Basis,
        bout: Basis,
        dbg: Option<DebugInfo>,
    },
    Predicated {
        then_func: Box<Expr>,
        else_func: Box<Expr>,
        pred: Basis,
        dbg: Option<DebugInfo>,
    },
    NonUniformSuperpos {
        pairs: Vec<(f64, QLit)>,
        dbg: Option<DebugInfo>,
    },
    Conditional {
        then_expr: Box<Expr>,
        else_expr: Box<Expr>,
        cond: Box<Expr>,
        dbg: Option<DebugInfo>,
    },
    QLit(QLit),
}

// ----- Statements -----

#[derive(Debug, Clone, PartialEq)]
pub enum Stmt {
    Assign {
        lhs: String,
        rhs: Expr,
        dbg: Option<DebugInfo>,
    },
    UnpackAssign {
        lhs: Vec<String>,
        rhs: Expr,
        dbg: Option<DebugInfo>,
    },
    Return {
        val: Expr,
        dbg: Option<DebugInfo>,
    },
}

// ----- Functions -----

#[derive(Debug, Clone, PartialEq)]
pub struct FunctionDef {
    pub name: String,
    pub args: Vec<(Type, String)>,
    pub ret_type: Type,
    pub body: Vec<Stmt>,
    pub dbg: Option<DebugInfo>,
}

// ----- Program -----

#[derive(Debug, Clone, PartialEq)]
pub struct Program {
    pub funcs: Vec<FunctionDef>,
    pub dbg: Option<DebugInfo>,
}
