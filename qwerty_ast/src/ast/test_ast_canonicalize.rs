//! Unit tests for canonicalization

use super::*;

#[test]
fn test_canonicalize_removes_trivial_qpu_statements() {
    let prog = Program {
        funcs: vec![Func::Qpu(FunctionDef {
            name: "kernel".to_string(),
            args: vec![],
            ret_type: Type::RegType {
                elem_ty: RegKind::Bit,
                dim: 2,
            },
            body: vec![
                Stmt::Expr(StmtExpr {
                    expr: qpu::Expr::UnitLiteral(qpu::UnitLiteral { dbg: None }),
                    dbg: None,
                }),
                Stmt::Return(Return {
                    val: qpu::Expr::BitLiteral(BitLiteral {
                        val: UBig::ZERO,
                        n_bits: 2,
                        dbg: None,
                    }),
                    dbg: None,
                }),
            ],
            is_rev: false,
            dbg: None,
        })],
        dbg: None,
    };

    let expected = Program {
        funcs: vec![Func::Qpu(FunctionDef {
            name: "kernel".to_string(),
            args: vec![],
            ret_type: Type::RegType {
                elem_ty: RegKind::Bit,
                dim: 2,
            },
            body: vec![
                // Note that unit literal statement is gone
                Stmt::Return(Return {
                    val: qpu::Expr::BitLiteral(BitLiteral {
                        val: UBig::ZERO,
                        n_bits: 2,
                        dbg: None,
                    }),
                    dbg: None,
                }),
            ],
            is_rev: false,
            dbg: None,
        })],
        dbg: None,
    };

    let actual = prog.canonicalize();
    assert_eq!(expected, actual);
}

#[test]
fn test_canonicalize_removes_trivial_classical_statements() {
    let prog = Program {
        funcs: vec![Func::Classical(FunctionDef {
            name: "oracle".to_string(),
            args: vec![],
            ret_type: Type::RegType {
                elem_ty: RegKind::Bit,
                dim: 2,
            },
            body: vec![
                Stmt::Expr(StmtExpr {
                    expr: classical::Expr::UnaryOp(classical::UnaryOp {
                        kind: classical::UnaryOpKind::Not,
                        val: Box::new(classical::Expr::BitLiteral(BitLiteral {
                            val: UBig::ZERO,
                            n_bits: 3,
                            dbg: None,
                        })),
                        dbg: None,
                    }),
                    dbg: None,
                }),
                Stmt::Return(Return {
                    val: classical::Expr::BitLiteral(BitLiteral {
                        val: UBig::ZERO,
                        n_bits: 2,
                        dbg: None,
                    }),
                    dbg: None,
                }),
            ],
            is_rev: false,
            dbg: None,
        })],
        dbg: None,
    };

    let expected = Program {
        funcs: vec![Func::Classical(FunctionDef {
            name: "oracle".to_string(),
            args: vec![],
            ret_type: Type::RegType {
                elem_ty: RegKind::Bit,
                dim: 2,
            },
            body: vec![
                // Note that unhelpful ~bit[3](0b000) is gone
                Stmt::Return(Return {
                    val: classical::Expr::BitLiteral(BitLiteral {
                        val: UBig::ZERO,
                        n_bits: 2,
                        dbg: None,
                    }),
                    dbg: None,
                }),
            ],
            is_rev: false,
            dbg: None,
        })],
        dbg: None,
    };

    let actual = prog.canonicalize();
    assert_eq!(expected, actual);
}
