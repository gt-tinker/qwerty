// Unit tests for vectors and qubit literals

use super::*;

#[test]
fn test_vec_to_programmer_str_zero() {
    let vec = Vector::ZeroVector { dbg: None };
    assert_eq!(vec.to_programmer_str(), "'0'");
}

#[test]
fn test_vec_to_programmer_str_one() {
    let vec = Vector::OneVector { dbg: None };
    assert_eq!(vec.to_programmer_str(), "'1'");
}

#[test]
fn test_vec_to_programmer_str_pad() {
    let vec = Vector::PadVector { dbg: None };
    assert_eq!(vec.to_programmer_str(), "'?'");
}

#[test]
fn test_vec_to_programmer_str_tgt() {
    let vec = Vector::TargetVector { dbg: None };
    assert_eq!(vec.to_programmer_str(), "'_'");
}

#[test]
fn test_vec_to_programmer_str_tilt_180() {
    let vec = Vector::VectorTilt {
        q: Box::new(Vector::OneVector { dbg: None }),
        angle_deg: 180.0,
        dbg: None,
    };
    assert_eq!(vec.to_programmer_str(), "-'1'");
}

#[test]
fn test_vec_to_programmer_str_tilt_non_180() {
    let vec = Vector::VectorTilt {
        q: Box::new(Vector::OneVector { dbg: None }),
        angle_deg: 1.23456,
        dbg: None,
    };
    assert_eq!(vec.to_programmer_str(), "('1'@1.23456)");
}

#[test]
fn test_vec_to_programmer_str_superpos() {
    let vec = Vector::UniformVectorSuperpos {
        q1: Box::new(Vector::ZeroVector { dbg: None }),
        q2: Box::new(Vector::OneVector { dbg: None }),
        dbg: None,
    };
    assert_eq!(vec.to_programmer_str(), "('0' + '1')");
}

#[test]
fn test_vec_to_programmer_str_tensor_01() {
    let vec = Vector::VectorTensor {
        qs: vec![
            Vector::ZeroVector { dbg: None },
            Vector::OneVector { dbg: None },
        ],
        dbg: None,
    };
    assert_eq!(vec.to_programmer_str(), "('0' * '1')");
}

#[test]
fn test_vec_to_programmer_str_tensor_01pad() {
    let vec = Vector::VectorTensor {
        qs: vec![
            Vector::ZeroVector { dbg: None },
            Vector::OneVector { dbg: None },
            Vector::PadVector { dbg: None },
        ],
        dbg: None,
    };
    assert_eq!(vec.to_programmer_str(), "('0' * '1' * '?')");
}

#[test]
fn test_vec_to_programmer_str_vector_unit() {
    let vec = Vector::VectorUnit { dbg: None };
    assert_eq!(vec.to_programmer_str(), "[]");
}

#[test]
fn test_vec_get_explicit_dim_zero() {
    // ⌊'0'⌋ = 1
    let vec = Vector::ZeroVector { dbg: None };
    assert_eq!(vec.get_explicit_dim(), Some(1));
}

#[test]
fn test_vec_get_explicit_dim_one() {
    // ⌊'1'⌋ == 1
    let vec = Vector::OneVector { dbg: None };
    assert_eq!(vec.get_explicit_dim(), Some(1));
}

#[test]
fn test_vec_get_explicit_dim_pad() {
    // ⌊'?'⌋ == 0
    let vec = Vector::PadVector { dbg: None };
    assert_eq!(vec.get_explicit_dim(), Some(0));
}

#[test]
fn test_vec_get_explicit_dim_tgt() {
    // ⌊'_'⌋ == 0
    let vec = Vector::TargetVector { dbg: None };
    assert_eq!(vec.get_explicit_dim(), Some(0));
}

#[test]
fn test_vec_get_explicit_dim_tilt() {
    // ⌊-'1'⌋ == 1
    let vec = Vector::VectorTilt {
        q: Box::new(Vector::OneVector { dbg: None }),
        angle_deg: 180.0,
        dbg: None,
    };
    assert_eq!(vec.get_explicit_dim(), Some(1));
}

#[test]
fn test_vec_get_explicit_dim_superpos_p() {
    // ⌊'0'+'1'⌋ == 1
    let vec = Vector::UniformVectorSuperpos {
        q1: Box::new(Vector::ZeroVector { dbg: None }),
        q2: Box::new(Vector::OneVector { dbg: None }),
        dbg: None,
    };
    assert_eq!(vec.get_explicit_dim(), Some(1));
}

#[test]
fn test_vec_get_explicit_dim_superpos_mismatch() {
    // ⌊'0'+'?'⌋ is undefined
    let vec = Vector::UniformVectorSuperpos {
        q1: Box::new(Vector::ZeroVector { dbg: None }),
        q2: Box::new(Vector::PadVector { dbg: None }),
        dbg: None,
    };
    assert_eq!(vec.get_explicit_dim(), None);
}

#[test]
fn test_vec_get_explicit_dim_tensor() {
    // ⌊'0'*'1'⌋ == 2
    let vec = Vector::VectorTensor {
        qs: vec![
            Vector::ZeroVector { dbg: None },
            Vector::OneVector { dbg: None },
        ],
        dbg: None,
    };
    assert_eq!(vec.get_explicit_dim(), Some(2));
}

#[test]
fn test_vec_get_explicit_dim_tensor_empty() {
    // ⌊ * ⌋ == 2
    //  ^^^
    // empty tensor
    let vec = Vector::VectorTensor {
        qs: vec![],
        dbg: None,
    };
    assert_eq!(vec.get_explicit_dim(), None);
}

#[test]
fn test_vec_get_explicit_dim_vector_unit() {
    // ⌊[]⌋ == 0
    let vec = Vector::VectorUnit { dbg: None };
    assert_eq!(vec.get_explicit_dim(), Some(0));
}

#[test]
fn test_vec_get_dim_zero() {
    // |'0'| = 1
    let vec = Vector::ZeroVector { dbg: None };
    assert_eq!(vec.get_dim(), Some(1));
}

#[test]
fn test_vec_get_dim_one() {
    // |'1'| == 1
    let vec = Vector::OneVector { dbg: None };
    assert_eq!(vec.get_dim(), Some(1));
}

#[test]
fn test_vec_get_dim_pad() {
    // |'?'| == 1
    let vec = Vector::PadVector { dbg: None };
    assert_eq!(vec.get_dim(), Some(1));
}

#[test]
fn test_vec_get_dim_tgt() {
    // |'_'| == 1
    let vec = Vector::TargetVector { dbg: None };
    assert_eq!(vec.get_dim(), Some(1));
}

#[test]
fn test_vec_get_dim_tilt() {
    // ⌊-'1'⌋ == 1
    let vec = Vector::VectorTilt {
        q: Box::new(Vector::OneVector { dbg: None }),
        angle_deg: 180.0,
        dbg: None,
    };
    assert_eq!(vec.get_dim(), Some(1));
}

#[test]
fn test_vec_get_dim_superpos_p() {
    // |'0'+'1'| == 1
    let vec = Vector::UniformVectorSuperpos {
        q1: Box::new(Vector::ZeroVector { dbg: None }),
        q2: Box::new(Vector::OneVector { dbg: None }),
        dbg: None,
    };
    assert_eq!(vec.get_dim(), Some(1));
}

#[test]
fn test_vec_get_dim_superpos_mismatch() {
    // |'0'+[]| is undefined
    let vec = Vector::UniformVectorSuperpos {
        q1: Box::new(Vector::ZeroVector { dbg: None }),
        q2: Box::new(Vector::VectorUnit { dbg: None }),
        dbg: None,
    };
    assert_eq!(vec.get_dim(), None);
}

#[test]
fn test_vec_get_dim_tensor() {
    // |'0'*'1'| == 2
    let vec = Vector::VectorTensor {
        qs: vec![
            Vector::ZeroVector { dbg: None },
            Vector::OneVector { dbg: None },
        ],
        dbg: None,
    };
    assert_eq!(vec.get_dim(), Some(2));
}

#[test]
fn test_vec_get_dim_tensor_empty() {
    // ⌊ * ⌋ == 2
    //  ^^^
    // empty tensor
    let vec = Vector::VectorTensor {
        qs: vec![],
        dbg: None,
    };
    assert_eq!(vec.get_dim(), None);
}

#[test]
fn test_vec_get_dim_vector_unit() {
    // |[]| == 0
    let vec = Vector::VectorUnit { dbg: None };
    assert_eq!(vec.get_dim(), Some(0));
}

#[test]
fn test_vec_get_atom_indices_zero() {
    // Ξ'?'['0'] = Ξ'_'['0'] = empty list
    let vec = Vector::ZeroVector { dbg: None };
    assert_eq!(vec.get_atom_indices(VectorAtomKind::PadAtom), Some(vec![]));
    assert_eq!(
        vec.get_atom_indices(VectorAtomKind::TargetAtom),
        Some(vec![])
    );
}

#[test]
fn test_vec_get_atom_indices_one() {
    // Ξ'?'['1'] = Ξ'_'['1'] = empty list
    let vec = Vector::OneVector { dbg: None };
    assert_eq!(vec.get_atom_indices(VectorAtomKind::PadAtom), Some(vec![]));
    assert_eq!(
        vec.get_atom_indices(VectorAtomKind::TargetAtom),
        Some(vec![])
    );
}

#[test]
fn test_vec_get_atom_indices_unit() {
    // Ξ'?'[[]] = Ξ'_'[[]] = empty list
    let vec = Vector::VectorUnit { dbg: None };
    assert_eq!(vec.get_atom_indices(VectorAtomKind::PadAtom), Some(vec![]));
    assert_eq!(
        vec.get_atom_indices(VectorAtomKind::TargetAtom),
        Some(vec![])
    );
}

#[test]
fn test_vec_get_atom_indices_pad() {
    // Ξ'?'['?'] = 0
    // Ξ'_'['?'] = empty list
    let vec = Vector::PadVector { dbg: None };
    assert_eq!(vec.get_atom_indices(VectorAtomKind::PadAtom), Some(vec![0]));
    assert_eq!(
        vec.get_atom_indices(VectorAtomKind::TargetAtom),
        Some(vec![])
    );
}

#[test]
fn test_vec_get_atom_indices_tgt() {
    // Ξ'?'['_'] = empty list
    // Ξ'_'['_'] = 0
    let vec = Vector::TargetVector { dbg: None };
    assert_eq!(vec.get_atom_indices(VectorAtomKind::PadAtom), Some(vec![]));
    assert_eq!(
        vec.get_atom_indices(VectorAtomKind::TargetAtom),
        Some(vec![0])
    );
}

#[test]
fn test_vec_get_atom_indices_tilt() {
    // Ξ'?'[-('1'*'?')] = 1
    // Ξ'_'[-('1'*'?')] = empty list
    let vec = Vector::VectorTilt {
        q: Box::new(Vector::VectorTensor {
            qs: vec![
                Vector::OneVector { dbg: None },
                Vector::PadVector { dbg: None },
            ],
            dbg: None,
        }),
        angle_deg: 180.0,
        dbg: None,
    };
    assert_eq!(vec.get_atom_indices(VectorAtomKind::PadAtom), Some(vec![1]));
    assert_eq!(
        vec.get_atom_indices(VectorAtomKind::TargetAtom),
        Some(vec![])
    );
}

#[test]
fn test_vec_get_atom_indices_superpos() {
    // Ξ'?'[('0'*'?') + ('1'*'?')] = 1
    // Ξ'_'[('0'*'?') + ('1'*'?')] = empty list
    let vec = Vector::UniformVectorSuperpos {
        q1: Box::new(Vector::VectorTensor {
            qs: vec![
                Vector::ZeroVector { dbg: None },
                Vector::PadVector { dbg: None },
            ],
            dbg: None,
        }),
        q2: Box::new(Vector::VectorTensor {
            qs: vec![
                Vector::OneVector { dbg: None },
                Vector::PadVector { dbg: None },
            ],
            dbg: None,
        }),
        dbg: None,
    };
    assert_eq!(vec.get_atom_indices(VectorAtomKind::PadAtom), Some(vec![1]));
    assert_eq!(
        vec.get_atom_indices(VectorAtomKind::TargetAtom),
        Some(vec![])
    );
}

#[test]
fn test_vec_get_atom_indices_superpos_undefined() {
    // Ξ'?'[('0'*'?') + ('1'*'_')] is undefined
    // Ξ'_'[('0'*'?') + ('1'*'_')] is undefined
    let vec = Vector::UniformVectorSuperpos {
        q1: Box::new(Vector::VectorTensor {
            qs: vec![
                Vector::ZeroVector { dbg: None },
                Vector::PadVector { dbg: None },
            ],
            dbg: None,
        }),
        q2: Box::new(Vector::VectorTensor {
            qs: vec![
                Vector::OneVector { dbg: None },
                Vector::TargetVector { dbg: None },
            ],
            dbg: None,
        }),
        dbg: None,
    };
    assert_eq!(vec.get_atom_indices(VectorAtomKind::PadAtom), None);
    assert_eq!(vec.get_atom_indices(VectorAtomKind::TargetAtom), None);
}

#[test]
fn test_vec_get_atom_indices_empty_tensor() {
    // Ξ'?'[( * )] is undefined
    // Ξ'_'[( * )] is undefined
    //       ^^^
    //   empty tensor
    let vec = Vector::VectorTensor {
        qs: vec![],
        dbg: None,
    };
    assert_eq!(vec.get_atom_indices(VectorAtomKind::PadAtom), None);
    assert_eq!(vec.get_atom_indices(VectorAtomKind::TargetAtom), None);
}

#[test]
fn test_vec_make_explicit_pad() {
    let dbg = Some(DebugLoc {
        file: "skippy.py".to_string(),
        line: 42,
        col: 420,
    });
    // '?' -> []
    let vec = Vector::PadVector { dbg: dbg.clone() };
    assert_eq!(vec.make_explicit(), Vector::VectorUnit { dbg });
}

#[test]
fn test_vec_make_explicit_tgt() {
    let dbg = Some(DebugLoc {
        file: "skippy.py".to_string(),
        line: 42,
        col: 420,
    });
    // '_' -> []
    let vec = Vector::TargetVector { dbg: dbg.clone() };
    assert_eq!(vec.make_explicit(), Vector::VectorUnit { dbg });
}

#[test]
fn test_vec_make_explicit_zero() {
    let dbg = Some(DebugLoc {
        file: "skippy.py".to_string(),
        line: 42,
        col: 420,
    });
    // '0' -> '0'
    let vec = Vector::ZeroVector { dbg: dbg.clone() };
    assert_eq!(vec.make_explicit(), vec);
}

#[test]
fn test_vec_make_explicit_one() {
    let dbg = Some(DebugLoc {
        file: "skippy.py".to_string(),
        line: 42,
        col: 420,
    });
    // '1' -> '1'
    let vec = Vector::OneVector { dbg: dbg.clone() };
    assert_eq!(vec.make_explicit(), vec);
}

#[test]
fn test_vec_make_explicit_unit() {
    let dbg = Some(DebugLoc {
        file: "skippy.py".to_string(),
        line: 42,
        col: 420,
    });
    // [] -> []
    let vec = Vector::VectorUnit { dbg: dbg.clone() };
    assert_eq!(vec.make_explicit(), vec);
}

#[test]
fn test_vec_make_explicit_tilt_one() {
    let dbg = Some(DebugLoc {
        file: "skippy.py".to_string(),
        line: 42,
        col: 420,
    });
    // -'1' -> -'1'
    let vec = Vector::VectorTilt {
        q: Box::new(Vector::OneVector { dbg: dbg.clone() }),
        angle_deg: 180.0,
        dbg: dbg.clone(),
    };
    assert_eq!(vec.make_explicit(), vec);
}

#[test]
fn test_vec_make_explicit_tilt_pad() {
    let dbg = Some(DebugLoc {
        file: "skippy.py".to_string(),
        line: 42,
        col: 420,
    });
    // -'?' -> []
    let vec = Vector::VectorTilt {
        q: Box::new(Vector::PadVector { dbg: dbg.clone() }),
        angle_deg: 180.0,
        dbg: dbg.clone(),
    };
    assert_eq!(vec.make_explicit(), Vector::VectorUnit { dbg });
}

#[test]
fn test_vec_make_explicit_superpos_p() {
    let dbg = Some(DebugLoc {
        file: "skippy.py".to_string(),
        line: 42,
        col: 420,
    });
    // '0'+'1' -> '0'+'1'
    let vec = Vector::UniformVectorSuperpos {
        q1: Box::new(Vector::ZeroVector { dbg: dbg.clone() }),
        q2: Box::new(Vector::ZeroVector { dbg: dbg.clone() }),
        dbg: dbg,
    };
    assert_eq!(vec.make_explicit(), vec);
}

#[test]
fn test_vec_make_explicit_tensor_01() {
    let dbg = Some(DebugLoc {
        file: "skippy.py".to_string(),
        line: 42,
        col: 420,
    });
    // '0'*'1' -> '0'*'1'
    let vec = Vector::VectorTensor {
        qs: vec![
            Vector::ZeroVector { dbg: dbg.clone() },
            Vector::OneVector { dbg: dbg.clone() },
        ],
        dbg: dbg.clone(),
    };
    assert_eq!(vec.make_explicit(), vec);
}

#[test]
fn test_vec_make_explicit_tensor_0pad1() {
    let dbg = Some(DebugLoc {
        file: "skippy.py".to_string(),
        line: 42,
        col: 420,
    });
    // '0'*'?'*'1' -> '0'*'1'
    let vec = Vector::VectorTensor {
        qs: vec![
            Vector::ZeroVector { dbg: dbg.clone() },
            Vector::PadVector { dbg: dbg.clone() },
            Vector::OneVector { dbg: dbg.clone() },
        ],
        dbg: dbg.clone(),
    };
    let explicit_vec = Vector::VectorTensor {
        qs: vec![
            Vector::ZeroVector { dbg: dbg.clone() },
            Vector::OneVector { dbg: dbg.clone() },
        ],
        dbg: dbg.clone(),
    };
    assert_eq!(vec.make_explicit(), explicit_vec);
}

#[test]
fn test_vec_make_explicit_tensor_pad() {
    let dbg = Some(DebugLoc {
        file: "skippy.py".to_string(),
        line: 42,
        col: 420,
    });
    // '?'*'?' -> []
    let vec = Vector::VectorTensor {
        qs: vec![
            Vector::PadVector { dbg: dbg.clone() },
            Vector::PadVector { dbg: dbg.clone() },
        ],
        dbg: dbg.clone(),
    };
    assert_eq!(vec.make_explicit(), Vector::VectorUnit { dbg });
}

#[test]
fn test_vec_make_explicit_tensor_pad1() {
    let dbg = Some(DebugLoc {
        file: "skippy.py".to_string(),
        line: 42,
        col: 420,
    });
    // '?'*'1' -> '1'
    let vec = Vector::VectorTensor {
        qs: vec![
            Vector::PadVector { dbg: dbg.clone() },
            Vector::OneVector { dbg: dbg.clone() },
        ],
        dbg: dbg.clone(),
    };
    assert_eq!(vec.make_explicit(), Vector::OneVector { dbg: dbg.clone() });
}

#[test]
fn test_vec_canonicalize_zero() {
    // '0' -> '0'
    let vec = Vector::ZeroVector { dbg: None };
    assert_eq!(vec.canonicalize(), vec);
}

#[test]
fn test_vec_canonicalize_one() {
    // '1' -> '1'
    let vec = Vector::OneVector { dbg: None };
    assert_eq!(vec.canonicalize(), vec);
}

#[test]
fn test_vec_canonicalize_pad() {
    // '?' -> '?'
    let vec = Vector::PadVector { dbg: None };
    assert_eq!(vec.canonicalize(), vec);
}

#[test]
fn test_vec_canonicalize_tgt() {
    // '_' -> '_'
    let vec = Vector::TargetVector { dbg: None };
    assert_eq!(vec.canonicalize(), vec);
}

#[test]
fn test_vec_canonicalize_unit() {
    // [] -> []
    let vec = Vector::VectorUnit { dbg: None };
    assert_eq!(vec.canonicalize(), vec);
}

#[test]
fn test_vec_canonicalize_tilt_mod360() {
    // '0'@-30 -> '0'@330
    let vec = Vector::VectorTilt {
        q: Box::new(Vector::ZeroVector { dbg: None }),
        angle_deg: -30.0,
        dbg: None,
    };
    let canon_vec = Vector::VectorTilt {
        q: Box::new(Vector::ZeroVector { dbg: None }),
        angle_deg: 330.0,
        dbg: None,
    };
    assert_eq!(vec.canonicalize(), canon_vec);
}

#[test]
fn test_vec_canonicalize_nested_tilt() {
    // ('1'@20)@10 -> '1'@30
    let vec = Vector::VectorTilt {
        q: Box::new(Vector::VectorTilt {
            q: Box::new(Vector::OneVector { dbg: None }),
            angle_deg: 20.0,
            dbg: None,
        }),
        angle_deg: 10.0,
        dbg: None,
    };
    let canon_vec = Vector::VectorTilt {
        q: Box::new(Vector::OneVector { dbg: None }),
        angle_deg: 30.0,
        dbg: None,
    };
    assert_eq!(vec.canonicalize(), canon_vec);
}

#[test]
fn test_vec_canonicalize_nested_tilt_mod_360() {
    // ('1'@20)@370 -> '1'@30
    let vec = Vector::VectorTilt {
        q: Box::new(Vector::VectorTilt {
            q: Box::new(Vector::OneVector { dbg: None }),
            angle_deg: 20.0,
            dbg: None,
        }),
        angle_deg: 370.0,
        dbg: None,
    };
    let canon_vec = Vector::VectorTilt {
        q: Box::new(Vector::OneVector { dbg: None }),
        angle_deg: 30.0,
        dbg: None,
    };
    assert_eq!(vec.canonicalize(), canon_vec);
}

#[test]
fn test_vec_canonicalize_nested_tilt_cancel_out() {
    // ('1'@20)@-20 -> '1'
    let vec = Vector::VectorTilt {
        q: Box::new(Vector::VectorTilt {
            q: Box::new(Vector::OneVector { dbg: None }),
            angle_deg: 20.0,
            dbg: None,
        }),
        angle_deg: -20.0,
        dbg: None,
    };
    let canon_vec = Vector::OneVector { dbg: None };
    assert_eq!(vec.canonicalize(), canon_vec);
}

#[test]
fn test_vec_canonicalize_nested_tilt_sum_to_360() {
    // ('1'@20)@340 -> '1'@360
    let vec = Vector::VectorTilt {
        q: Box::new(Vector::VectorTilt {
            q: Box::new(Vector::OneVector { dbg: None }),
            angle_deg: 20.0,
            dbg: None,
        }),
        angle_deg: 340.0,
        dbg: None,
    };
    let canon_vec = Vector::OneVector { dbg: None };
    assert_eq!(vec.canonicalize(), canon_vec);
}

#[test]
fn test_vec_canonicalize_tilt_zero() {
    // '0'@360 -> '0'
    let vec = Vector::VectorTilt {
        q: Box::new(Vector::ZeroVector { dbg: None }),
        angle_deg: 360.0,
        dbg: None,
    };
    let canon_vec = Vector::ZeroVector { dbg: None };
    assert_eq!(vec.canonicalize(), canon_vec);
}

#[test]
fn test_vec_canonicalize_tilt_neg_720() {
    // '1'@-720 -> '1'
    let vec = Vector::VectorTilt {
        q: Box::new(Vector::OneVector { dbg: None }),
        angle_deg: -720.0,
        dbg: None,
    };
    let canon_vec = Vector::OneVector { dbg: None };
    assert_eq!(vec.canonicalize(), canon_vec);
}

#[test]
fn test_vec_canonicalize_superpos_p() {
    // '0'+'1' -> '0'+'1'
    let vec = Vector::UniformVectorSuperpos {
        q1: Box::new(Vector::ZeroVector { dbg: None }),
        q2: Box::new(Vector::OneVector { dbg: None }),
        dbg: None,
    };
    assert_eq!(vec.canonicalize(), vec);
}

#[test]
fn test_vec_canonicalize_superpos_p_sorted() {
    // '1'+'0' -> '0'+'1'
    let vec = Vector::UniformVectorSuperpos {
        q1: Box::new(Vector::OneVector { dbg: None }),
        q2: Box::new(Vector::ZeroVector { dbg: None }),
        dbg: None,
    };
    let canon_vec = Vector::UniformVectorSuperpos {
        q1: Box::new(Vector::ZeroVector { dbg: None }),
        q2: Box::new(Vector::OneVector { dbg: None }),
        dbg: None,
    };
    assert_eq!(vec.canonicalize(), canon_vec);
}

#[test]
fn test_vec_canonicalize_superpos_neg_p() {
    // ('0'@180)+('1'@-180) -> ('0'+'1')@180
    let vec = Vector::UniformVectorSuperpos {
        q1: Box::new(Vector::VectorTilt {
            q: Box::new(Vector::ZeroVector { dbg: None }),
            angle_deg: 180.0,
            dbg: None,
        }),
        q2: Box::new(Vector::VectorTilt {
            q: Box::new(Vector::OneVector { dbg: None }),
            angle_deg: -180.0,
            dbg: None,
        }),
        dbg: None,
    };
    let canon_vec = Vector::VectorTilt {
        q: Box::new(Vector::UniformVectorSuperpos {
            q1: Box::new(Vector::ZeroVector { dbg: None }),
            q2: Box::new(Vector::OneVector { dbg: None }),
            dbg: None,
        }),
        angle_deg: 180.0,
        dbg: None,
    };
    assert_eq!(vec.canonicalize(), canon_vec);
}

#[test]
fn test_vec_canonicalize_superpos_neg_p_sorted() {
    // ('1'@180)+('0'@-180) -> ('0'+'1')@180
    let vec = Vector::UniformVectorSuperpos {
        q1: Box::new(Vector::VectorTilt {
            q: Box::new(Vector::OneVector { dbg: None }),
            angle_deg: 180.0,
            dbg: None,
        }),
        q2: Box::new(Vector::VectorTilt {
            q: Box::new(Vector::ZeroVector { dbg: None }),
            angle_deg: -180.0,
            dbg: None,
        }),
        dbg: None,
    };
    let canon_vec = Vector::VectorTilt {
        q: Box::new(Vector::UniformVectorSuperpos {
            q1: Box::new(Vector::ZeroVector { dbg: None }),
            q2: Box::new(Vector::OneVector { dbg: None }),
            dbg: None,
        }),
        angle_deg: 180.0,
        dbg: None,
    };
    assert_eq!(vec.canonicalize(), canon_vec);
}

#[test]
fn test_vec_canonicalize_tensor_01() {
    // '0'*'1' -> '0'*'1'
    let vec = Vector::VectorTensor {
        qs: vec![
            Vector::ZeroVector { dbg: None },
            Vector::OneVector { dbg: None },
        ],
        dbg: None,
    };
    assert_eq!(vec.canonicalize(), vec);
}

#[test]
fn test_vec_canonicalize_tensor_unit() {
    // '0'*[]*'1' -> '0'*'1'
    let vec = Vector::VectorTensor {
        qs: vec![
            Vector::ZeroVector { dbg: None },
            Vector::VectorUnit { dbg: None },
            Vector::OneVector { dbg: None },
        ],
        dbg: None,
    };
    let canon_vec = Vector::VectorTensor {
        qs: vec![
            Vector::ZeroVector { dbg: None },
            Vector::OneVector { dbg: None },
        ],
        dbg: None,
    };
    assert_eq!(vec.canonicalize(), canon_vec);
}

#[test]
fn test_vec_canonicalize_tensor_unit_tilt() {
    // '0'*([]@30)*(('1'@10)@90) -> ('0'*'1')@130
    let vec = Vector::VectorTensor {
        qs: vec![
            Vector::ZeroVector { dbg: None },
            Vector::VectorTilt {
                q: Box::new(Vector::VectorUnit { dbg: None }),
                angle_deg: 30.0,
                dbg: None,
            },
            Vector::VectorTilt {
                q: Box::new(Vector::VectorTilt {
                    q: Box::new(Vector::OneVector { dbg: None }),
                    angle_deg: 10.0,
                    dbg: None,
                }),
                angle_deg: 90.0,
                dbg: None,
            },
        ],
        dbg: None,
    };
    let canon_vec = Vector::VectorTilt {
        q: Box::new(Vector::VectorTensor {
            qs: vec![
                Vector::ZeroVector { dbg: None },
                Vector::OneVector { dbg: None },
            ],
            dbg: None,
        }),
        angle_deg: 130.0,
        dbg: None,
    };
    assert_eq!(vec.canonicalize(), canon_vec);
}

#[test]
fn test_vec_canonicalize_tensor_sum_tilts() {
    // ('0'@30) * ('1'*60) -> ('0'*'1')@90
    let vec = Vector::VectorTensor {
        qs: vec![
            Vector::VectorTilt {
                q: Box::new(Vector::ZeroVector { dbg: None }),
                angle_deg: 30.0,
                dbg: None,
            },
            Vector::VectorTilt {
                q: Box::new(Vector::OneVector { dbg: None }),
                angle_deg: 60.0,
                dbg: None,
            },
        ],
        dbg: None,
    };
    let canon_vec = Vector::VectorTilt {
        q: Box::new(Vector::VectorTensor {
            qs: vec![
                Vector::ZeroVector { dbg: None },
                Vector::OneVector { dbg: None },
            ],
            dbg: None,
        }),
        angle_deg: 90.0,
        dbg: None,
    };
    assert_eq!(vec.canonicalize(), canon_vec);
}

#[test]
fn test_vec_canonicalize_tensor_nested() {
    // '0'*('1'*'0') -> '0'*'1'*'0'
    let vec = Vector::VectorTensor {
        qs: vec![
            Vector::ZeroVector { dbg: None },
            Vector::VectorTensor {
                qs: vec![
                    Vector::OneVector { dbg: None },
                    Vector::ZeroVector { dbg: None },
                ],
                dbg: None,
            },
        ],
        dbg: None,
    };
    let canon_vec = Vector::VectorTensor {
        qs: vec![
            Vector::ZeroVector { dbg: None },
            Vector::OneVector { dbg: None },
            Vector::ZeroVector { dbg: None },
        ],
        dbg: None,
    };
    assert_eq!(vec.canonicalize(), canon_vec);
}

#[test]
fn test_vec_canonicalize_tensor_nested_tilt() {
    // '0'*(('1'*'0')@-30) -> ('0'*'1'*'0')@330
    let vec = Vector::VectorTensor {
        qs: vec![
            Vector::ZeroVector { dbg: None },
            Vector::VectorTilt {
                q: Box::new(Vector::VectorTensor {
                    qs: vec![
                        Vector::OneVector { dbg: None },
                        Vector::ZeroVector { dbg: None },
                    ],
                    dbg: None,
                }),
                angle_deg: -30.0,
                dbg: None,
            },
        ],
        dbg: None,
    };
    let canon_vec = Vector::VectorTilt {
        q: Box::new(Vector::VectorTensor {
            qs: vec![
                Vector::ZeroVector { dbg: None },
                Vector::OneVector { dbg: None },
                Vector::ZeroVector { dbg: None },
            ],
            dbg: None,
        }),
        angle_deg: 330.0,
        dbg: None,
    };
    assert_eq!(vec.canonicalize(), canon_vec);
}

#[test]
fn test_vec_canonicalize_tensor_removed() {
    // []*'0'*[]*[] -> '0'
    let vec = Vector::VectorTensor {
        qs: vec![
            Vector::VectorUnit { dbg: None },
            Vector::ZeroVector { dbg: None },
            Vector::VectorUnit { dbg: None },
            Vector::VectorUnit { dbg: None },
        ],
        dbg: None,
    };
    let canon_vec = Vector::ZeroVector { dbg: None };
    assert_eq!(vec.canonicalize(), canon_vec);
}

#[test]
fn test_vec_canonicalize_tensor_removed_nested_unit() {
    // ([]*[])*([]*[]) -> []
    let vec = Vector::VectorTensor {
        qs: vec![
            Vector::VectorTensor {
                qs: vec![
                    Vector::VectorUnit { dbg: None },
                    Vector::VectorUnit { dbg: None },
                ],
                dbg: None,
            },
            Vector::VectorTensor {
                qs: vec![
                    Vector::VectorUnit { dbg: None },
                    Vector::VectorUnit { dbg: None },
                ],
                dbg: None,
            },
        ],
        dbg: None,
    };
    let canon_vec = Vector::VectorUnit { dbg: None };
    assert_eq!(vec.canonicalize(), canon_vec);
}

#[test]
fn test_vec_canonicalize_superpos_single_tilt() {
    // ('0'@35+'1'@45) -> ('0'+'1'@10)@35
    let vec = Vector::UniformVectorSuperpos {
        q1: Box::new(Vector::VectorTilt {
            q: Box::new(Vector::ZeroVector { dbg: None }),
            angle_deg: 35.0,
            dbg: None,
        }),
        q2: Box::new(Vector::VectorTilt {
            q: Box::new(Vector::OneVector { dbg: None }),
            angle_deg: 45.0,
            dbg: None,
        }),
        dbg: None,
    };

    let canon_vec = Vector::VectorTilt {
        q: Box::new(Vector::UniformVectorSuperpos {
            q1: Box::new(Vector::ZeroVector { dbg: None }),
            q2: Box::new(Vector::VectorTilt {
                q: Box::new(Vector::OneVector { dbg: None }),
                angle_deg: 10.0,
                dbg: None,
            }),
            dbg: None,
        }),
        angle_deg: 35.0,
        dbg: None,
    };
    assert_eq!(vec.canonicalize(), canon_vec);
}

#[test]
fn test_vec_canonicalize_interfere_p_and_m() {
    // ('0'+'1') + ('0'+('1'@180)) -> '0'
    let vec = Vector::UniformVectorSuperpos {
        q1: Box::new(Vector::UniformVectorSuperpos {
            q1: Box::new(Vector::ZeroVector { dbg: None }),
            q2: Box::new(Vector::OneVector { dbg: None }),
            dbg: None,
        }),
        q2: Box::new(Vector::UniformVectorSuperpos {
            q1: Box::new(Vector::ZeroVector { dbg: None }),
            q2: Box::new(Vector::VectorTilt {
                q: Box::new(Vector::OneVector { dbg: None }),
                angle_deg: 180.0,
                dbg: None,
            }),
            dbg: None,
        }),
        dbg: None,
    };

    let canon_vec = Vector::ZeroVector { dbg: None };
    assert_eq!(vec.canonicalize(), canon_vec);
}

#[test]
fn test_vec_canonicalize_interfere_p_and_neg_m() {
    // ('0'+'1') + (('0'@180)+'1') -> '1'
    let vec = Vector::UniformVectorSuperpos {
        q1: Box::new(Vector::UniformVectorSuperpos {
            q1: Box::new(Vector::ZeroVector { dbg: None }),
            q2: Box::new(Vector::OneVector { dbg: None }),
            dbg: None,
        }),
        q2: Box::new(Vector::UniformVectorSuperpos {
            q1: Box::new(Vector::VectorTilt {
                q: Box::new(Vector::ZeroVector { dbg: None }),
                angle_deg: 180.0,
                dbg: None,
            }),
            q2: Box::new(Vector::OneVector { dbg: None }),
            dbg: None,
        }),
        dbg: None,
    };

    let canon_vec = Vector::OneVector { dbg: None };
    assert_eq!(vec.canonicalize(), canon_vec);
}

#[test]
fn test_vec_canonicalize_interfere_p_and_minus_m() {
    // ('0'+'1') + ('0'+('1'@180))@180 -> '1'
    let vec = Vector::UniformVectorSuperpos {
        q1: Box::new(Vector::UniformVectorSuperpos {
            q1: Box::new(Vector::ZeroVector { dbg: None }),
            q2: Box::new(Vector::OneVector { dbg: None }),
            dbg: None,
        }),
        q2: Box::new(Vector::VectorTilt {
            q: Box::new(Vector::UniformVectorSuperpos {
                q1: Box::new(Vector::ZeroVector { dbg: None }),
                q2: Box::new(Vector::VectorTilt {
                    q: Box::new(Vector::OneVector { dbg: None }),
                    angle_deg: 180.0,
                    dbg: None,
                }),
                dbg: None,
            }),
            angle_deg: 180.0,
            dbg: None,
        }),
        dbg: None,
    };

    let canon_vec = Vector::OneVector { dbg: None };
    assert_eq!(vec.canonicalize(), canon_vec);
}

#[test]
fn test_vec_canonicalize_interfere_minus_p_and_m() {
    // ('0'+'1')@180 + ('0'+('1'@180)) -> -'1'
    let vec = Vector::UniformVectorSuperpos {
        q1: Box::new(Vector::VectorTilt {
            q: Box::new(Vector::UniformVectorSuperpos {
                q1: Box::new(Vector::ZeroVector { dbg: None }),
                q2: Box::new(Vector::OneVector { dbg: None }),
                dbg: None,
            }),
            angle_deg: 180.0,
            dbg: None,
        }),
        q2: Box::new(Vector::UniformVectorSuperpos {
            q1: Box::new(Vector::ZeroVector { dbg: None }),
            q2: Box::new(Vector::VectorTilt {
                q: Box::new(Vector::OneVector { dbg: None }),
                angle_deg: 180.0,
                dbg: None,
            }),
            dbg: None,
        }),
        dbg: None,
    };

    let canon_vec = Vector::VectorTilt {
        q: Box::new(Vector::OneVector { dbg: None }),
        angle_deg: 180.0,
        dbg: None,
    };
    assert_eq!(vec.canonicalize(), canon_vec);
}
