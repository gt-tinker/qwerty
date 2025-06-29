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
fn test_qlits_are_ortho_sym() {
    // Base cases: '0' _|_ '1'
    assert!(qlits_are_ortho(
        &QLit::ZeroQubit { dbg: None },
        &QLit::OneQubit { dbg: None }
    ));
    assert!(qlits_are_ortho(
        &QLit::OneQubit { dbg: None },
        &QLit::ZeroQubit { dbg: None }
    ));
    assert!(!qlits_are_ortho(
        &QLit::ZeroQubit { dbg: None },
        &QLit::ZeroQubit { dbg: None }
    ));

    // '0' and '1' _|_ '0' and -'1'
    assert!(qlits_are_ortho(
        &QLit::UniformSuperpos {
            q1: Box::new(QLit::ZeroQubit { dbg: None }),
            q2: Box::new(QLit::OneQubit { dbg: None }),
            dbg: None
        },
        &QLit::UniformSuperpos {
            q1: Box::new(QLit::ZeroQubit { dbg: None }),
            q2: Box::new(QLit::QubitTilt {
                q: Box::new(QLit::OneQubit { dbg: None }),
                angle_deg: 180.0,
                dbg: None
            }),
            dbg: None
        }
    ));
    // '0' and -'1' _|_ '0' and '1'
    assert!(qlits_are_ortho(
        &QLit::UniformSuperpos {
            q1: Box::new(QLit::ZeroQubit { dbg: None }),
            q2: Box::new(QLit::QubitTilt {
                q: Box::new(QLit::OneQubit { dbg: None }),
                angle_deg: 180.0,
                dbg: None
            }),
            dbg: None
        },
        &QLit::UniformSuperpos {
            q1: Box::new(QLit::ZeroQubit { dbg: None }),
            q2: Box::new(QLit::OneQubit { dbg: None }),
            dbg: None
        }
    ));
    // '0' and '1' !_|_ '0' and '1'
    assert!(!qlits_are_ortho(
        &QLit::UniformSuperpos {
            q1: Box::new(QLit::ZeroQubit { dbg: None }),
            q2: Box::new(QLit::OneQubit { dbg: None }),
            dbg: None
        },
        &QLit::UniformSuperpos {
            q1: Box::new(QLit::ZeroQubit { dbg: None }),
            q2: Box::new(QLit::OneQubit { dbg: None }),
            dbg: None
        }
    ));

    // '0'@45 and '1'@45 _|_ '0'@45 and '1'@225
    assert!(qlits_are_ortho(
        &QLit::UniformSuperpos {
            q1: Box::new(QLit::QubitTilt {
                q: Box::new(QLit::ZeroQubit { dbg: None }),
                angle_deg: 45.0,
                dbg: None
            }),
            q2: Box::new(QLit::QubitTilt {
                q: Box::new(QLit::OneQubit { dbg: None }),
                angle_deg: 45.0,
                dbg: None
            }),
            dbg: None
        },
        &QLit::UniformSuperpos {
            q1: Box::new(QLit::QubitTilt {
                q: Box::new(QLit::ZeroQubit { dbg: None }),
                angle_deg: 45.0,
                dbg: None
            }),
            q2: Box::new(QLit::QubitTilt {
                q: Box::new(QLit::OneQubit { dbg: None }),
                angle_deg: 225.0,
                dbg: None
            }),
            dbg: None
        }
    ));

    // '0' and '1'@0 _|_ '0'@180 and '1'
    assert!(qlits_are_ortho(
        &QLit::UniformSuperpos {
            q1: Box::new(QLit::ZeroQubit { dbg: None }),
            q2: Box::new(QLit::QubitTilt {
                q: Box::new(QLit::OneQubit { dbg: None }),
                angle_deg: 0.0,
                dbg: None
            }),
            dbg: None
        },
        &QLit::UniformSuperpos {
            q1: Box::new(QLit::QubitTilt {
                q: Box::new(QLit::ZeroQubit { dbg: None }),
                angle_deg: 180.0,
                dbg: None
            }),
            q2: Box::new(QLit::OneQubit { dbg: None }),
            dbg: None
        }
    ));

    // '0' and '1'@5 !_|_ '0'@180 and '1'
    assert!(!qlits_are_ortho(
        &QLit::UniformSuperpos {
            q1: Box::new(QLit::ZeroQubit { dbg: None }),
            q2: Box::new(QLit::QubitTilt {
                q: Box::new(QLit::OneQubit { dbg: None }),
                angle_deg: 5.0,
                dbg: None
            }),
            dbg: None
        },
        &QLit::UniformSuperpos {
            q1: Box::new(QLit::QubitTilt {
                q: Box::new(QLit::ZeroQubit { dbg: None }),
                angle_deg: 180.0,
                dbg: None
            }),
            q2: Box::new(QLit::OneQubit { dbg: None }),
            dbg: None
        }
    ));

    // '0'@45 + '1'@225 _|_ '0'@135 + '1'@135
    assert!(qlits_are_ortho(
        &QLit::UniformSuperpos {
            q1: Box::new(QLit::QubitTilt {
                q: Box::new(QLit::ZeroQubit { dbg: None }),
                angle_deg: 45.0,
                dbg: None
            }),
            q2: Box::new(QLit::QubitTilt {
                q: Box::new(QLit::OneQubit { dbg: None }),
                angle_deg: 225.0,
                dbg: None
            }),
            dbg: None
        },
        &QLit::UniformSuperpos {
            q1: Box::new(QLit::QubitTilt {
                q: Box::new(QLit::ZeroQubit { dbg: None }),
                angle_deg: 135.0,
                dbg: None
            }),
            q2: Box::new(QLit::QubitTilt {
                q: Box::new(QLit::OneQubit { dbg: None }),
                angle_deg: 135.0,
                dbg: None
            }),
            dbg: None
        }
    ));
    // '0'@45 + '1'@225 !_|_ '0'@0 + '1'@180
    assert!(!qlits_are_ortho(
        &QLit::UniformSuperpos {
            q1: Box::new(QLit::QubitTilt {
                q: Box::new(QLit::ZeroQubit { dbg: None }),
                angle_deg: 45.0,
                dbg: None
            }),
            q2: Box::new(QLit::QubitTilt {
                q: Box::new(QLit::OneQubit { dbg: None }),
                angle_deg: 225.0,
                dbg: None
            }),
            dbg: None
        },
        &QLit::UniformSuperpos {
            q1: Box::new(QLit::QubitTilt {
                q: Box::new(QLit::ZeroQubit { dbg: None }),
                angle_deg: 0.0,
                dbg: None
            }),
            q2: Box::new(QLit::QubitTilt {
                q: Box::new(QLit::OneQubit { dbg: None }),
                angle_deg: 180.0,
                dbg: None
            }),
            dbg: None
        }
    ));
    // '0'@45 + '1'@225 !_|_ '0' + '1'@180
    assert!(!qlits_are_ortho(
        &QLit::UniformSuperpos {
            q1: Box::new(QLit::QubitTilt {
                q: Box::new(QLit::ZeroQubit { dbg: None }),
                angle_deg: 45.0,
                dbg: None
            }),
            q2: Box::new(QLit::QubitTilt {
                q: Box::new(QLit::OneQubit { dbg: None }),
                angle_deg: 225.0,
                dbg: None
            }),
            dbg: None
        },
        &QLit::UniformSuperpos {
            q1: Box::new(QLit::ZeroQubit { dbg: None }),
            q2: Box::new(QLit::QubitTilt {
                q: Box::new(QLit::OneQubit { dbg: None }),
                angle_deg: 180.0,
                dbg: None
            }),
            dbg: None
        }
    ));
    // '0'@45 + '1'@225 !_|_ '0' + '1'@37
    assert!(!qlits_are_ortho(
        &QLit::UniformSuperpos {
            q1: Box::new(QLit::QubitTilt {
                q: Box::new(QLit::ZeroQubit { dbg: None }),
                angle_deg: 45.0,
                dbg: None
            }),
            q2: Box::new(QLit::QubitTilt {
                q: Box::new(QLit::OneQubit { dbg: None }),
                angle_deg: 225.0,
                dbg: None
            }),
            dbg: None
        },
        &QLit::UniformSuperpos {
            q1: Box::new(QLit::ZeroQubit { dbg: None }),
            q2: Box::new(QLit::QubitTilt {
                q: Box::new(QLit::OneQubit { dbg: None }),
                angle_deg: 37.0,
                dbg: None
            }),
            dbg: None
        }
    ));

    // '0'*'1' _|_ '0'*'0'
    assert!(qlits_are_ortho(
        &QLit::QubitTensor {
            qs: vec![QLit::ZeroQubit { dbg: None }, QLit::OneQubit { dbg: None },],
            dbg: None
        },
        &QLit::QubitTensor {
            qs: vec![QLit::ZeroQubit { dbg: None }, QLit::ZeroQubit { dbg: None },],
            dbg: None
        }
    ));

    // '0'*'1' !_|_ '0'*'1'
    assert!(!qlits_are_ortho(
        &QLit::QubitTensor {
            qs: vec![QLit::ZeroQubit { dbg: None }, QLit::OneQubit { dbg: None },],
            dbg: None
        },
        &QLit::QubitTensor {
            qs: vec![QLit::ZeroQubit { dbg: None }, QLit::OneQubit { dbg: None },],
            dbg: None
        }
    ));

    // '0'*'1' _|_ ('0'*'0')@45
    assert!(qlits_are_ortho(
        &QLit::QubitTensor {
            qs: vec![QLit::ZeroQubit { dbg: None }, QLit::OneQubit { dbg: None },],
            dbg: None
        },
        &QLit::QubitTilt {
            q: Box::new(QLit::QubitTensor {
                qs: vec![QLit::ZeroQubit { dbg: None }, QLit::ZeroQubit { dbg: None },],
                dbg: None
            }),
            angle_deg: 45.0,
            dbg: None
        }
    ));

    // '0'*'0' + '0'*'1' _|_ '1'*'0' + '1'*'1'
    assert!(qlits_are_ortho(
        &QLit::UniformSuperpos {
            q1: Box::new(QLit::QubitTensor {
                qs: vec![QLit::ZeroQubit { dbg: None }, QLit::ZeroQubit { dbg: None },],
                dbg: None
            }),
            q2: Box::new(QLit::QubitTensor {
                qs: vec![QLit::ZeroQubit { dbg: None }, QLit::OneQubit { dbg: None },],
                dbg: None
            }),
            dbg: None
        },
        &QLit::UniformSuperpos {
            q1: Box::new(QLit::QubitTensor {
                qs: vec![QLit::OneQubit { dbg: None }, QLit::ZeroQubit { dbg: None },],
                dbg: None
            }),
            q2: Box::new(QLit::QubitTensor {
                qs: vec![QLit::OneQubit { dbg: None }, QLit::OneQubit { dbg: None },],
                dbg: None
            }),
            dbg: None
        }
    ));

    // '0'*'0' + '1'*'1' !_|_ '1'*'0' + '1'*'1'
    assert!(!qlits_are_ortho(
        &QLit::UniformSuperpos {
            q1: Box::new(QLit::QubitTensor {
                qs: vec![QLit::ZeroQubit { dbg: None }, QLit::ZeroQubit { dbg: None },],
                dbg: None
            }),
            q2: Box::new(QLit::QubitTensor {
                qs: vec![QLit::OneQubit { dbg: None }, QLit::OneQubit { dbg: None },],
                dbg: None
            }),
            dbg: None
        },
        &QLit::UniformSuperpos {
            q1: Box::new(QLit::QubitTensor {
                qs: vec![QLit::OneQubit { dbg: None }, QLit::ZeroQubit { dbg: None },],
                dbg: None
            }),
            q2: Box::new(QLit::QubitTensor {
                qs: vec![QLit::OneQubit { dbg: None }, QLit::OneQubit { dbg: None },],
                dbg: None
            }),
            dbg: None
        }
    ));

    // '0'*'0' + '0'*'1' _|_ '1'*'0' + ('1'*'1')@45
    assert!(qlits_are_ortho(
        &QLit::UniformSuperpos {
            q1: Box::new(QLit::QubitTensor {
                qs: vec![QLit::ZeroQubit { dbg: None }, QLit::ZeroQubit { dbg: None },],
                dbg: None
            }),
            q2: Box::new(QLit::QubitTensor {
                qs: vec![QLit::ZeroQubit { dbg: None }, QLit::OneQubit { dbg: None },],
                dbg: None
            }),
            dbg: None
        },
        &QLit::UniformSuperpos {
            q1: Box::new(QLit::QubitTensor {
                qs: vec![QLit::OneQubit { dbg: None }, QLit::ZeroQubit { dbg: None },],
                dbg: None
            }),
            q2: Box::new(QLit::QubitTilt {
                q: Box::new(QLit::QubitTensor {
                    qs: vec![QLit::OneQubit { dbg: None }, QLit::OneQubit { dbg: None },],
                    dbg: None
                }),
                angle_deg: 45.0,
                dbg: None
            }),
            dbg: None
        }
    ));

    // ('0'*'0' + '0'*'1') * '0' !_|_ '0'*'0'*'0'
    assert!(!qlits_are_ortho(
        &QLit::QubitTensor {
            qs: vec![
                QLit::UniformSuperpos {
                    q1: Box::new(QLit::QubitTensor {
                        qs: vec![QLit::ZeroQubit { dbg: None }, QLit::ZeroQubit { dbg: None },],
                        dbg: None
                    }),
                    q2: Box::new(QLit::QubitTensor {
                        qs: vec![QLit::ZeroQubit { dbg: None }, QLit::OneQubit { dbg: None },],
                        dbg: None
                    }),
                    dbg: None
                },
                QLit::ZeroQubit { dbg: None }
            ],
            dbg: None
        },
        &QLit::QubitTensor {
            qs: vec![
                QLit::ZeroQubit { dbg: None },
                QLit::ZeroQubit { dbg: None },
                QLit::ZeroQubit { dbg: None }
            ],
            dbg: None
        }
    ));

    // ('0'*'0' + '0'*'1') * '0' !_|_ '0'*'0'
    assert!(!qlits_are_ortho(
        &QLit::QubitTensor {
            qs: vec![
                QLit::UniformSuperpos {
                    q1: Box::new(QLit::QubitTensor {
                        qs: vec![QLit::ZeroQubit { dbg: None }, QLit::ZeroQubit { dbg: None },],
                        dbg: None
                    }),
                    q2: Box::new(QLit::QubitTensor {
                        qs: vec![QLit::ZeroQubit { dbg: None }, QLit::OneQubit { dbg: None },],
                        dbg: None
                    }),
                    dbg: None
                },
                QLit::ZeroQubit { dbg: None }
            ],
            dbg: None
        },
        &QLit::QubitTensor {
            qs: vec![QLit::ZeroQubit { dbg: None }, QLit::ZeroQubit { dbg: None }],
            dbg: None
        }
    ));

    // _ * _ !_|_ _ * _
    // \___/      \____/
    //   \           /
    //  empty tensor products
    assert!(!qlits_are_ortho(
        &QLit::QubitTensor {
            qs: vec![],
            dbg: None
        },
        &QLit::QubitTensor {
            qs: vec![],
            dbg: None
        }
    ));

    // '0'*'0' _|_ ('0'@45)*'1'
    assert!(qlits_are_ortho(
        &QLit::QubitTensor {
            qs: vec![QLit::ZeroQubit { dbg: None }, QLit::ZeroQubit { dbg: None },],
            dbg: None
        },
        &QLit::QubitTensor {
            qs: vec![
                QLit::QubitTilt {
                    q: Box::new(QLit::ZeroQubit { dbg: None }),
                    angle_deg: 45.0,
                    dbg: None
                },
                QLit::OneQubit { dbg: None },
            ],
            dbg: None
        },
    ));

    // ('0'+'1')*'0' _|_ ('0'+-'1')*'0'
    assert!(qlits_are_ortho(
        &QLit::QubitTensor {
            qs: vec![
                QLit::UniformSuperpos {
                    q1: Box::new(QLit::ZeroQubit { dbg: None }),
                    q2: Box::new(QLit::OneQubit { dbg: None }),
                    dbg: None
                },
                QLit::ZeroQubit { dbg: None }
            ],
            dbg: None
        },
        &QLit::QubitTensor {
            qs: vec![
                QLit::UniformSuperpos {
                    q1: Box::new(QLit::ZeroQubit { dbg: None }),
                    q2: Box::new(QLit::QubitTilt {
                        q: Box::new(QLit::OneQubit { dbg: None }),
                        angle_deg: 180.0,
                        dbg: None
                    }),
                    dbg: None
                },
                QLit::ZeroQubit { dbg: None }
            ],
            dbg: None
        },
    ));

    // (('0'*'0')+'1')*'0' !_|_ ('0'+-'1')*'0'
    assert!(!qlits_are_ortho(
        &QLit::QubitTensor {
            qs: vec![
                QLit::UniformSuperpos {
                    q1: Box::new(QLit::QubitTensor {
                        qs: vec![QLit::ZeroQubit { dbg: None }, QLit::ZeroQubit { dbg: None }],
                        dbg: None
                    }),
                    q2: Box::new(QLit::OneQubit { dbg: None }),
                    dbg: None
                },
                QLit::ZeroQubit { dbg: None }
            ],
            dbg: None
        },
        &QLit::QubitTensor {
            qs: vec![
                QLit::UniformSuperpos {
                    q1: Box::new(QLit::ZeroQubit { dbg: None }),
                    q2: Box::new(QLit::QubitTilt {
                        q: Box::new(QLit::OneQubit { dbg: None }),
                        angle_deg: 180.0,
                        dbg: None
                    }),
                    dbg: None
                },
                QLit::ZeroQubit { dbg: None }
            ],
            dbg: None
        },
    ));

    // ((_ * _)+'1')*'0' !_|_ ('0'+-'1')*'0'
    //  \_____/
    //     \
    //      empty tensor product
    assert!(!qlits_are_ortho(
        &QLit::QubitTensor {
            qs: vec![
                QLit::UniformSuperpos {
                    q1: Box::new(QLit::QubitTensor {
                        qs: vec![],
                        dbg: None
                    }),
                    q2: Box::new(QLit::OneQubit { dbg: None }),
                    dbg: None
                },
                QLit::ZeroQubit { dbg: None }
            ],
            dbg: None
        },
        &QLit::QubitTensor {
            qs: vec![
                QLit::UniformSuperpos {
                    q1: Box::new(QLit::ZeroQubit { dbg: None }),
                    q2: Box::new(QLit::QubitTilt {
                        q: Box::new(QLit::OneQubit { dbg: None }),
                        angle_deg: 180.0,
                        dbg: None
                    }),
                    dbg: None
                },
                QLit::ZeroQubit { dbg: None }
            ],
            dbg: None
        },
    ));
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
