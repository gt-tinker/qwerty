// Unit tests for typechecking bases

use super::*;
use crate::dbg::DebugLoc;

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

#[test]
fn test_typecheck_expr_btrans_empty() {
    let dbg = DebugLoc {
        file: "skippy.py".to_string(),
        line: 42,
        col: 420,
    };
    let mut type_env = TypeEnv::new();
    // () >> () !: qubit[0] rev-> qubit[0] because empty bases are not allowed
    let ast = Expr::BasisTranslation {
        bin: Basis::EmptyBasisLiteral {
            dbg: Some(dbg.clone()),
        },
        bout: Basis::EmptyBasisLiteral { dbg: None },
        dbg: None,
    };

    let result = typecheck_expr(&ast, &mut type_env);
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
fn test_typecheck_expr_btrans_dim_mismatch() {
    let dbg = DebugLoc {
        file: "skippy.py".to_string(),
        line: 42,
        col: 420,
    };
    let mut type_env = TypeEnv::new();
    // {'0'} >> {'0'+'0'} !: qubit[1] rev-> qubit[1]
    // because basis dimensions do not align
    let ast = Expr::BasisTranslation {
        bin: Basis::BasisLiteral {
            vecs: vec![Vector::ZeroVector { dbg: None }],
            dbg: None,
        },
        bout: Basis::BasisLiteral {
            vecs: vec![Vector::VectorTensor {
                qs: vec![
                    Vector::ZeroVector { dbg: None },
                    Vector::ZeroVector { dbg: None },
                ],
                dbg: None,
            }],
            dbg: None,
        },
        dbg: Some(dbg.clone()),
    };

    let result = typecheck_expr(&ast, &mut type_env);
    assert_eq!(
        result,
        Err(TypeError {
            kind: TypeErrorKind::DimMismatch,
            dbg: Some(dbg)
        })
    );
    assert!(type_env.is_empty());
}

#[test]
fn test_typecheck_expr_btrans_left_err_propagate() {
    let dbg = DebugLoc {
        file: "skippy.py".to_string(),
        line: 42,
        col: 420,
    };
    let mut type_env = TypeEnv::new();
    // {} >> {'0'} !: qubit[0] rev-> qubit[0]
    // because {} is not a valid basis
    let ast = Expr::BasisTranslation {
        bin: Basis::BasisLiteral {
            vecs: vec![],
            dbg: Some(dbg.clone()),
        },
        bout: Basis::BasisLiteral {
            vecs: vec![Vector::ZeroVector { dbg: None }],
            dbg: None,
        },
        dbg: None,
    };

    let result = typecheck_expr(&ast, &mut type_env);
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
fn test_typecheck_expr_btrans_right_err_propagate() {
    let dbg = DebugLoc {
        file: "skippy.py".to_string(),
        line: 42,
        col: 420,
    };
    let mut type_env = TypeEnv::new();
    // {'0'} >> {} !: qubit[1] rev-> qubit[1]
    // because {} is not a valid basis
    let ast = Expr::BasisTranslation {
        bin: Basis::BasisLiteral {
            vecs: vec![Vector::ZeroVector { dbg: None }],
            dbg: None,
        },
        bout: Basis::BasisLiteral {
            vecs: vec![],
            dbg: Some(dbg.clone()),
        },
        dbg: None,
    };

    let result = typecheck_expr(&ast, &mut type_env);
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
fn test_typecheck_expr_btrans_zero_identity() {
    let mut type_env = TypeEnv::new();
    // {'0'} >> {'0'} : qubit[1] rev-> qubit[1]
    let ast = Expr::BasisTranslation {
        bin: Basis::BasisLiteral {
            vecs: vec![Vector::ZeroVector { dbg: None }],
            dbg: None,
        },
        bout: Basis::BasisLiteral {
            vecs: vec![Vector::ZeroVector { dbg: None }],
            dbg: None,
        },
        dbg: None,
    };

    let result = typecheck_expr(&ast, &mut type_env);
    assert_eq!(
        result,
        Ok(Type::RevFuncType {
            in_out_ty: Box::new(Type::RegType {
                elem_ty: RegKind::Qubit,
                dim: 1,
            })
        })
    );
    assert!(type_env.is_empty());
}
