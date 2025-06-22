/*
 * QWERTY Programming Language Compiler
 * Abstract Syntax Tree (AST) Definitions
 * 
 * This module defines the Abstract Syntax Tree (AST) structures
 * used for parsing and representing QWERTY programs.
 * 
 * Version: 1.0
 */

use crate::span::SourceSpan;

// ----- Types -----

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Type {
    FuncType {
        in_ty: Box<Type>,
        out_ty: Box<Type>,
    },
    RevFuncType {
        in_out_ty: Box<Type>,
    },
    RegType {
        elem_ty: RegKind,
        dim: u64,
    },
    UnitType,
}

// ----- Registers -----

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RegKind {
    Bit,    // Classical bit register
    Qubit,  // Quantum bit register
    Basis,  // Register for basis states
}

// ----- Qubit Literals -----

#[derive(Debug, Clone, PartialEq)]
pub enum QLit {
    ZeroQubit { span: Option<SourceSpan> },
    OneQubit { span: Option<SourceSpan> },
    QubitTilt {
        q: Box<QLit>,
        angle_deg: f64,
        span: Option<SourceSpan>,
    },
    UniformSuperpos {
        q1: Box<QLit>,
        q2: Box<QLit>,
        span: Option<SourceSpan>,
    },
    QubitTensor {
        qs: Vec<QLit>,
        span: Option<SourceSpan>,
    },
}

// ----- Vector -----

#[derive(Debug, Clone, PartialEq)]
pub enum Vector {
    ZeroVector { span: Option<SourceSpan> },
    OneVector { span: Option<SourceSpan> },
    PadVector { span: Option<SourceSpan> },
    TargetVector { span: Option<SourceSpan> },
    VectorTilt {
        q: Box<QLit>,
        angle_deg: f64,
        span: Option<SourceSpan>,
    },
    UniformVectorSuperpos {
        q1: Box<QLit>,
        q2: Box<QLit>,
        span: Option<SourceSpan>,
    },
    VectorTensor {
        qs: Vec<QLit>,
        span: Option<SourceSpan>,
    },
}

// ----- Basis -----

#[derive(Debug, Clone, PartialEq)]
pub enum Basis {
    BasisLiteral {
        vecs: Vec<Vector>,
        span: Option<SourceSpan>,
    },
    EmptyBasisLiteral { span: Option<SourceSpan> },
    BasisTensor {
        bases: Vec<Basis>,
        span: Option<SourceSpan>,
    },
}

// ----- Expressions -----

#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    Variable {
        name: String,
        span: Option<SourceSpan>,
    },
    UnitLiteral {
        span: Option<SourceSpan>,
    },
    Adjoint {
        func: Box<Expr>,
        span: Option<SourceSpan>,
    },
    Pipe {
        lhs: Box<Expr>,
        rhs: Box<Expr>,
        span: Option<SourceSpan>,
    },
    Measure {
        basis: Basis,
        span: Option<SourceSpan>,
    },
    Discard {
        span: Option<SourceSpan>,
    },
    Tensor {
        vals: Vec<Expr>,
        span: Option<SourceSpan>,
    },
    BasisTranslation {
        bin: Basis,
        bout: Basis,
        span: Option<SourceSpan>,
    },
    Predicated {
        then_func: Box<Expr>,
        else_func: Box<Expr>,
        pred: Basis,
        span: Option<SourceSpan>,
    },
    NonUniformSuperpos {
        pairs: Vec<(f64, QLit)>,
        span: Option<SourceSpan>,
    },
    Conditional {
        then_expr: Box<Expr>,
        else_expr: Box<Expr>,
        cond: Box<Expr>,
        span: Option<SourceSpan>,
    },
    QLit(QLit),
}

// ----- Statements -----

#[derive(Debug, Clone, PartialEq)]
pub enum Stmt {
    Assign {
        lhs: String,
        rhs: Expr,
        span: Option<SourceSpan>,
    },
    UnpackAssign {
        lhs: Vec<String>,
        rhs: Expr,
        span: Option<SourceSpan>,
    },
    Return {
        val: Expr,
        span: Option<SourceSpan>,
    },
}

// ----- Functions -----

#[derive(Debug, Clone, PartialEq)]
pub struct FunctionDef {
    pub name: String,
    pub args: Vec<(Type, String)>,
    pub ret_type: Type,
    pub body: Vec<Stmt>,
    pub span: Option<SourceSpan>,
}

// ----- Program -----

#[derive(Debug, Clone, PartialEq)]
pub struct Program {
    pub funcs: Vec<FunctionDef>,
    pub span: Option<SourceSpan>,
}
