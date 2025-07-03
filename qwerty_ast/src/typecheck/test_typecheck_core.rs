// Unit tests for typecheck module

use super::*;

#[test]
fn test_typecheck_var_and_assign() {
    let prog = Program {
        funcs: vec![FunctionDef {
            name: "main".into(),
            args: vec![(Type::UnitType, "x".into())],
            ret_type: Type::UnitType,
            body: vec![Stmt::Assign {
                lhs: "y".into(),
                rhs: Expr::Variable {
                    name: "x".into(),
                    dbg: None,
                },
                dbg: None,
            }],
            dbg: None,
        }],
        dbg: None,
    };
    let result = typecheck_program(&prog);
    assert!(result.is_ok());
}

#[test]
fn test_unpack_assign_typing() {
    use crate::ast::*;

    // Case 1: Legal unpack (2 variables, qubit[2])
    let legal_prog = Program {
        funcs: vec![FunctionDef {
            name: "main".into(),
            args: vec![],
            ret_type: Type::UnitType,
            body: vec![Stmt::UnpackAssign {
                lhs: vec!["a".into(), "b".into()],
                rhs: Expr::Variable {
                    name: "payload".into(),
                    dbg: None,
                },
                dbg: None,
            }],
            dbg: None,
        }],
        dbg: None,
    };

    // The environment must bind payload to a qubit[2] for this to pass
    let mut env = TypeEnv::new();
    env.insert_var(
        "payload",
        Type::RegType {
            elem_ty: RegKind::Qubit,
            dim: 2,
        },
    );

    // The typechecker expects to build the environment itself, so let's test the stmt directly:
    let stmt = &legal_prog.funcs[0].body[0];
    let result = super::typecheck_stmt(stmt, &mut env, &Type::UnitType);
    assert!(result.is_ok(), "Legal unpack failed typechecking");

    // Case 2: Illegal unpack (3 variables, qubit[2])
    let illegal_stmt = Stmt::UnpackAssign {
        lhs: vec!["a".into(), "b".into(), "c".into()],
        rhs: Expr::Variable {
            name: "payload".into(),
            dbg: None,
        },
        dbg: None,
    };
    let mut env2 = TypeEnv::new();
    env2.insert_var(
        "payload",
        Type::RegType {
            elem_ty: RegKind::Qubit,
            dim: 2,
        },
    );

    let result2 = super::typecheck_stmt(&illegal_stmt, &mut env2, &Type::UnitType);
    assert!(
        matches!(
            result2,
            Err(TypeError {
                kind: TypeErrorKind::WrongArity { .. },
                ..
            })
        ),
        "Unpack of mismatched arity did not fail as expected"
    );
}

#[test]
fn test_unpack_assign_non_register_rhs() {
    use crate::ast::*;

    // Trying to unpack a UnitType (not a register!)
    let stmt = Stmt::UnpackAssign {
        lhs: vec!["a".into()],
        rhs: Expr::Variable {
            name: "not_a_reg".into(),
            dbg: None,
        },
        dbg: None,
    };
    let mut env = TypeEnv::new();
    env.insert_var("not_a_reg", Type::UnitType);

    let result = super::typecheck_stmt(&stmt, &mut env, &Type::UnitType);
    assert!(
        matches!(
            result,
            Err(TypeError {
                kind: TypeErrorKind::InvalidType(_),
                ..
            })
        ),
        "Unpack of non-register type did not fail as expected"
    );
}
