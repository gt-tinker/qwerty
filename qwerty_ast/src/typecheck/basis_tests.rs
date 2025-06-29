// Unit tests for typechecking bases

use super::*;

#[test]
fn test_typecheck_basis_std() {
    let mut type_env = TypeEnv::new();
    // {'0', '1'} : basis[1] because '0' _|_ '1'
    let ast = Basis::BasisLiteral {
        vecs: vec![
            Vector::ZeroVector { dbg: None },
            Vector::OneVector { dbg: None },
        ],
        dbg: None,
    };
    let result = typecheck_basis(&ast, &mut type_env);
    assert_eq!(
        result,
        Ok(Type::RegType {
            elem_ty: RegKind::Basis,
            dim: 1
        })
    );
    assert!(type_env.is_empty());
}

#[test]
fn test_typecheck_basis_not_ortho() {
    let dbg = DebugLoc {
        file: "skippy.py".to_string(),
        line: 42,
        col: 420,
    };
    let mut type_env = TypeEnv::new();
    // {'0', '0'} !: basis[1] because '0' !_|_ '0'
    let ast = Basis::BasisLiteral {
        vecs: vec![
            Vector::ZeroVector { dbg: None },
            Vector::ZeroVector { dbg: None },
        ],
        dbg: Some(dbg.clone()),
    };
    let result = typecheck_basis(&ast, &mut type_env);
    assert_eq!(
        result,
        Err(TypeError {
            kind: TypeErrorKind::NotOrthogonal {
                left: "'0'".to_string(),
                right: "'0'".to_string()
            },
            dbg: Some(dbg)
        })
    );
    assert!(type_env.is_empty());
}

#[test]
fn test_typecheck_basis_not_ortho_tilt() {
    let dbg = DebugLoc {
        file: "skippy.py".to_string(),
        line: 42,
        col: 420,
    };
    let mut type_env = TypeEnv::new();
    // {'0', -'0'} !: basis[1] because '0' !_|_ -'0'
    let ast = Basis::BasisLiteral {
        vecs: vec![
            Vector::ZeroVector { dbg: None },
            Vector::VectorTilt {
                q: Box::new(Vector::ZeroVector { dbg: None }),
                angle_deg: 180.0,
                dbg: None,
            },
        ],
        dbg: Some(dbg.clone()),
    };
    let result = typecheck_basis(&ast, &mut type_env);
    assert_eq!(
        result,
        Err(TypeError {
            kind: TypeErrorKind::NotOrthogonal {
                left: "'0'".to_string(),
                right: "('0' @ 180)".to_string()
            },
            dbg: Some(dbg)
        })
    );
    assert!(type_env.is_empty());
}

#[test]
fn test_typecheck_basis_empty() {
    let dbg = DebugLoc {
        file: "skippy.py".to_string(),
        line: 42,
        col: 420,
    };
    let mut type_env = TypeEnv::new();
    // {} !: basis[1] because it's empty
    let ast = Basis::BasisLiteral {
        vecs: vec![],
        dbg: Some(dbg.clone()),
    };
    let result = typecheck_basis(&ast, &mut type_env);
    assert_eq!(
        result,
        Err(TypeError {
            kind: TypeErrorKind::EmptyLiteral,
            dbg: Some(dbg)
        })
    );
    assert!(type_env.is_empty());
}

#[test]
fn test_typecheck_basis_empty_vector() {
    let dbg = DebugLoc {
        file: "skippy.py".to_string(),
        line: 42,
        col: 420,
    };
    let mut type_env = TypeEnv::new();
    // {[]} !: basis[0] because [] is an empty vector
    let ast = Basis::BasisLiteral {
        vecs: vec![Vector::VectorUnit { dbg: None }],
        dbg: Some(dbg.clone()),
    };
    let result = typecheck_basis(&ast, &mut type_env);
    assert_eq!(
        result,
        Err(TypeError {
            kind: TypeErrorKind::EmptyLiteral,
            dbg: Some(dbg)
        })
    );
    assert!(type_env.is_empty());
}

#[test]
fn test_typecheck_basis_mixed_vector_dims() {
    let dbg = DebugLoc {
        file: "skippy.py".to_string(),
        line: 42,
        col: 420,
    };
    let mut type_env = TypeEnv::new();
    // {'0', '0'+'1'} !: basis[0] because vector dimensions differ
    let ast = Basis::BasisLiteral {
        vecs: vec![
            Vector::ZeroVector { dbg: None },
            Vector::VectorTensor {
                qs: vec![
                    Vector::ZeroVector { dbg: None },
                    Vector::OneVector { dbg: None },
                ],
                dbg: None,
            },
        ],
        dbg: Some(dbg.clone()),
    };
    let result = typecheck_basis(&ast, &mut type_env);
    assert_eq!(
        result,
        Err(TypeError {
            kind: TypeErrorKind::DimMismatch,
            dbg: Some(dbg)
        })
    );
    assert!(type_env.is_empty());
}
