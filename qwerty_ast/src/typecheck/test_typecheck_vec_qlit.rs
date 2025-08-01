// Unit tests for vectors and qubit literals

use super::*;

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

    // '0'@0 + '1'@90 _|_ '0'@0 + '1'@270
    assert!(qlits_are_ortho(
        &QLit::UniformSuperpos {
            q1: Box::new(QLit::QubitTilt {
                q: Box::new(QLit::ZeroQubit { dbg: None }),
                angle_deg: 0.0,
                dbg: None
            }),
            q2: Box::new(QLit::QubitTilt {
                q: Box::new(QLit::OneQubit { dbg: None }),
                angle_deg: 90.0,
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
                angle_deg: 270.0,
                dbg: None
            }),
            dbg: None
        }
    ));

    // '0' + '1'@90 _|_ '0' + '1'@270
    assert!(qlits_are_ortho(
        &QLit::UniformSuperpos {
            q1: Box::new(QLit::ZeroQubit { dbg: None }),
            q2: Box::new(QLit::QubitTilt {
                q: Box::new(QLit::OneQubit { dbg: None }),
                angle_deg: 90.0,
                dbg: None
            }),
            dbg: None
        },
        &QLit::UniformSuperpos {
            q1: Box::new(QLit::ZeroQubit { dbg: None }),
            q2: Box::new(QLit::QubitTilt {
                q: Box::new(QLit::OneQubit { dbg: None }),
                angle_deg: 270.0,
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
