// Unit tests for typecheck module

use super::*;

#[test]
fn test_typecheck_var_and_assign() {
    let prog = Program {
        funcs: vec![FunctionDef::new(
            "main".into(),
            vec![(Type::UnitType, "x".into())],
            Type::UnitType,
            vec![Stmt::Assign(Assign {
                lhs: "y".into(),
                rhs: Expr::Variable(Variable {
                    name: "x".into(),
                    dbg: None,
                }),
                dbg: None,
            })],
            false,
            None,
        )],
        dbg: None,
    };

    let result = prog.typecheck();
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
            body: vec![Stmt::UnpackAssign(UnpackAssign {
                lhs: vec!["a".into(), "b".into()],
                rhs: Expr::Variable(Variable {
                    name: "payload".into(),
                    dbg: None,
                }),
                dbg: None,
            })],
            is_rev: true,
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
    let result = stmt.typecheck(&mut env, Some(Type::UnitType));
    assert!(result.is_ok(), "Legal unpack failed typechecking");

    // Case 2: Illegal unpack (3 variables, qubit[2])
    let illegal_stmt = Stmt::UnpackAssign(UnpackAssign {
        lhs: vec!["a".into(), "b".into(), "c".into()],
        rhs: Expr::Variable(Variable {
            name: "payload".into(),
            dbg: None,
        }),
        dbg: None,
    });
    let mut env2 = TypeEnv::new();
    env2.insert_var(
        "payload",
        Type::RegType {
            elem_ty: RegKind::Qubit,
            dim: 2,
        },
    );

    let result2 = illegal_stmt.typecheck(&mut env2, Some(Type::UnitType));
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
    let stmt = Stmt::UnpackAssign(UnpackAssign {
        lhs: vec!["a".into()],
        rhs: Expr::Variable(Variable {
            name: "not_a_reg".into(),
            dbg: None,
        }),
        dbg: None,
    });
    let mut env = TypeEnv::new();
    env.insert_var("not_a_reg", Type::UnitType);

    let result = stmt.typecheck(&mut env, Some(Type::UnitType));
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

#[test]
fn test_valid_tupletype_construction() {
    use crate::ast::*;
    let tys = vec![Type::UnitType, Type::UnitType];
    let tuple = Type::tuple(tys);
    assert!(tuple.is_ok());
}

#[test]
fn test_invalid_tupletype_construction() {
    use crate::ast::*;
    // This test checks that constructing a TupleType with fewer than 2 types fails.
    let tys = vec![Type::UnitType];
    let tuple = Type::tuple(tys);
    assert!(tuple.is_err());
}

#[test]
fn test_functiondef_get_type_tuple_args() {
    use crate::ast::*;
    let func = FunctionDef::new(
        "f_tuple".to_string(),
        vec![
            (Type::UnitType, "x".to_string()),
            (Type::UnitType, "y".to_string()),
        ],
        Type::UnitType,
        vec![],
        false,
        None,
    );
    let ty = func.get_type();
    if let Type::FuncType { in_ty, .. } = ty {
        assert!(matches!(*in_ty, Type::TupleType { .. }));
    } else {
        panic!("Expected FuncType");
    }
}
