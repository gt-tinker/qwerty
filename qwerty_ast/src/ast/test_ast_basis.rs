// Unit tests for typechecking bases

use super::*;
use crate::dbg::DebugLoc;

#[test]
fn test_basis_get_dbg_basis_literal() {
    let dbg = DebugLoc {
        file: "skippy.py".to_string(),
        line: 42,
        col: 3,
    };
    // {'0'}
    let basis = Basis::BasisLiteral {
        vecs: vec![Vector::ZeroVector { dbg: None }],
        dbg: Some(dbg.clone()),
    };
    assert_eq!(basis.get_dbg(), Some(dbg));
}

#[test]
fn test_basis_get_dbg_empty_basis_literal() {
    let dbg = DebugLoc {
        file: "skippy.py".to_string(),
        line: 42,
        col: 3,
    };
    // []
    let basis = Basis::EmptyBasisLiteral {
        dbg: Some(dbg.clone()),
    };
    assert_eq!(basis.get_dbg(), Some(dbg));
}

#[test]
fn test_basis_get_dbg_basis_tensor() {
    let dbg = DebugLoc {
        file: "skippy.py".to_string(),
        line: 42,
        col: 3,
    };
    // {'0'} + {'1'}
    let basis = Basis::BasisTensor {
        bases: vec![
            Basis::BasisLiteral {
                vecs: vec![Vector::ZeroVector { dbg: None }],
                dbg: None,
            },
            Basis::BasisLiteral {
                vecs: vec![Vector::OneVector { dbg: None }],
                dbg: None,
            },
        ],
        dbg: Some(dbg.clone()),
    };
    assert_eq!(basis.get_dbg(), Some(dbg));
}

#[test]
fn test_basis_get_dim_basis_literal_empty() {
    // |{}| = 0
    let basis = Basis::BasisLiteral {
        vecs: vec![],
        dbg: None,
    };
    assert_eq!(basis.get_dim(), None);
}

#[test]
fn test_basis_get_dim_basis_literal_std() {
    // |{'0','1'}| = 1
    let basis = Basis::BasisLiteral {
        vecs: vec![
            Vector::ZeroVector { dbg: None },
            Vector::OneVector { dbg: None },
        ],
        dbg: None,
    };
    assert_eq!(basis.get_dim(), Some(1));
}

#[test]
fn test_basis_get_dim_basis_literal_mismatch() {
    // |{'0','1'*'1'}| = 1
    let basis = Basis::BasisLiteral {
        vecs: vec![
            Vector::ZeroVector { dbg: None },
            Vector::VectorTensor {
                qs: vec![
                    Vector::OneVector { dbg: None },
                    Vector::OneVector { dbg: None },
                ],
                dbg: None,
            },
        ],
        dbg: None,
    };
    assert_eq!(basis.get_dim(), None);
}

#[test]
fn test_basis_get_dim_empty_basis_literal() {
    // |[]| = 0
    let basis = Basis::EmptyBasisLiteral { dbg: None };
    assert_eq!(basis.get_dim(), Some(0));
}

#[test]
fn test_basis_get_dim_basis_tensor_empty() {
    // | * | is undefined
    //  ^^^
    // empty tensor
    let basis = Basis::BasisTensor {
        bases: vec![],
        dbg: None,
    };
    assert_eq!(basis.get_dim(), None);
}

#[test]
fn test_basis_get_dim_basis_tensor() {
    // |[] * {'0'}| is undefined
    //  ^^^
    // empty tensor
    let basis = Basis::BasisTensor {
        bases: vec![
            Basis::EmptyBasisLiteral { dbg: None },
            Basis::BasisLiteral {
                vecs: vec![Vector::ZeroVector { dbg: None }],
                dbg: None,
            },
        ],
        dbg: None,
    };
    assert_eq!(basis.get_dim(), Some(1));
}

#[test]
fn test_basis_get_atom_indices_basis_literal_empty() {
    // Ξ'?'[{}] is undefined
    // Ξ'_'[{}] is undefined
    let basis = Basis::BasisLiteral {
        vecs: vec![],
        dbg: None,
    };
    assert_eq!(basis.get_atom_indices(VectorAtomKind::PadAtom), None);
    assert_eq!(basis.get_atom_indices(VectorAtomKind::TargetAtom), None);
}

#[test]
fn test_basis_get_atom_indices_basis_literal_std() {
    // Ξ'?'[{'0','1'}] = empty list
    // Ξ'_'[{'0','1'}] = empty list
    let basis = Basis::BasisLiteral {
        vecs: vec![
            Vector::ZeroVector { dbg: None },
            Vector::OneVector { dbg: None },
        ],
        dbg: None,
    };
    assert_eq!(
        basis.get_atom_indices(VectorAtomKind::PadAtom),
        Some(vec![])
    );
    assert_eq!(
        basis.get_atom_indices(VectorAtomKind::TargetAtom),
        Some(vec![])
    );
}

#[test]
fn test_basis_get_atom_indices_basis_literal_std_pad() {
    // Ξ'?'[{'0'*'?','1'*'?'}] = 1
    // Ξ'_'[{'0'*'?','1'*'?'}] = empty list
    let basis = Basis::BasisLiteral {
        vecs: vec![
            Vector::VectorTensor {
                qs: vec![
                    Vector::ZeroVector { dbg: None },
                    Vector::PadVector { dbg: None },
                ],
                dbg: None,
            },
            Vector::VectorTensor {
                qs: vec![
                    Vector::OneVector { dbg: None },
                    Vector::PadVector { dbg: None },
                ],
                dbg: None,
            },
        ],
        dbg: None,
    };
    assert_eq!(
        basis.get_atom_indices(VectorAtomKind::PadAtom),
        Some(vec![1])
    );
    assert_eq!(
        basis.get_atom_indices(VectorAtomKind::TargetAtom),
        Some(vec![])
    );
}

#[test]
fn test_basis_get_atom_indices_basis_literal_mismatch() {
    // Ξ'?'[{'0','1'*'?'}] is undefined
    // Ξ'_'[{'0','1'*'?'}] = empty list
    let basis = Basis::BasisLiteral {
        vecs: vec![
            Vector::ZeroVector { dbg: None },
            Vector::VectorTensor {
                qs: vec![
                    Vector::OneVector { dbg: None },
                    Vector::PadVector { dbg: None },
                ],
                dbg: None,
            },
        ],
        dbg: None,
    };
    assert_eq!(basis.get_atom_indices(VectorAtomKind::PadAtom), None);
    assert_eq!(
        basis.get_atom_indices(VectorAtomKind::TargetAtom),
        Some(vec![])
    );
}

#[test]
fn test_basis_get_atom_indices_empty_basis_literal() {
    // Ξ'?'[[]] = empty list
    // Ξ'_'[[]] = empty list
    let basis = Basis::EmptyBasisLiteral { dbg: None };
    assert_eq!(
        basis.get_atom_indices(VectorAtomKind::PadAtom),
        Some(vec![])
    );
    assert_eq!(
        basis.get_atom_indices(VectorAtomKind::TargetAtom),
        Some(vec![])
    );
}

#[test]
fn test_basis_get_atom_indices_basis_tensor_empty() {
    // Ξ'?'[ * ] is undefined
    // Ξ'_'[ * ] is undefined
    //      ^^^
    //  empty tensor
    let basis = Basis::BasisTensor {
        bases: vec![],
        dbg: None,
    };
    assert_eq!(basis.get_atom_indices(VectorAtomKind::PadAtom), None);
    assert_eq!(basis.get_atom_indices(VectorAtomKind::TargetAtom), None);
}

#[test]
fn test_basis_get_atom_indices_basis_tensor_singleton() {
    // Ξ'?'[{'1'} * ] is undefined
    // Ξ'_'[{'1'} * ] is undefined
    //      ^^^^^^
    //  singleton tensor
    let basis = Basis::BasisTensor {
        bases: vec![Basis::BasisLiteral {
            vecs: vec![Vector::OneVector { dbg: None }],
            dbg: None,
        }],
        dbg: None,
    };
    assert_eq!(basis.get_atom_indices(VectorAtomKind::PadAtom), None);
    assert_eq!(basis.get_atom_indices(VectorAtomKind::TargetAtom), None);
}

#[test]
fn test_basis_get_atom_indices_basis_tensor_pad() {
    // Ξ'?'[[] * {'?'}] = 0
    // Ξ'_'[[] * {'?'}] = empty
    let basis = Basis::BasisTensor {
        bases: vec![
            Basis::EmptyBasisLiteral { dbg: None },
            Basis::BasisLiteral {
                vecs: vec![Vector::PadVector { dbg: None }],
                dbg: None,
            },
        ],
        dbg: None,
    };
    assert_eq!(
        basis.get_atom_indices(VectorAtomKind::PadAtom),
        Some(vec![0])
    );
    assert_eq!(
        basis.get_atom_indices(VectorAtomKind::TargetAtom),
        Some(vec![])
    );
}

#[test]
fn test_basis_make_explicit_pad() {
    let dbg = Some(DebugLoc {
        file: "skippy.py".to_string(),
        line: 42,
        col: 420,
    });
    // {'?'} -> []
    let basis = Basis::BasisLiteral {
        vecs: vec![Vector::PadVector { dbg: dbg.clone() }],
        dbg: dbg.clone(),
    };
    assert_eq!(basis.make_explicit(), Basis::EmptyBasisLiteral { dbg });
}

#[test]
fn test_basis_make_explicit_tgt() {
    let dbg = Some(DebugLoc {
        file: "skippy.py".to_string(),
        line: 42,
        col: 420,
    });
    // {'_'} -> []
    let basis = Basis::BasisLiteral {
        vecs: vec![Vector::PadVector { dbg: dbg.clone() }],
        dbg: dbg.clone(),
    };
    assert_eq!(basis.make_explicit(), Basis::EmptyBasisLiteral { dbg });
}

#[test]
fn test_basis_make_explicit_std() {
    let dbg = Some(DebugLoc {
        file: "skippy.py".to_string(),
        line: 42,
        col: 420,
    });
    // {'0','1'} -> {'0','1'}
    let basis = Basis::BasisLiteral {
        vecs: vec![
            Vector::ZeroVector { dbg: dbg.clone() },
            Vector::OneVector { dbg: dbg.clone() },
        ],
        dbg: dbg,
    };
    assert_eq!(basis.make_explicit(), basis);
}

#[test]
fn test_basis_make_explicit_std_pad() {
    let dbg = Some(DebugLoc {
        file: "skippy.py".to_string(),
        line: 42,
        col: 420,
    });
    // {'?'*'0','?'*'1'} -> {'0','1'}
    let basis = Basis::BasisLiteral {
        vecs: vec![
            Vector::VectorTensor {
                qs: vec![
                    Vector::PadVector { dbg: dbg.clone() },
                    Vector::ZeroVector { dbg: dbg.clone() },
                ],
                dbg: dbg.clone(),
            },
            Vector::VectorTensor {
                qs: vec![
                    Vector::PadVector { dbg: dbg.clone() },
                    Vector::OneVector { dbg: dbg.clone() },
                ],
                dbg: dbg.clone(),
            },
        ],
        dbg: dbg.clone(),
    };
    let explicit_basis = Basis::BasisLiteral {
        vecs: vec![
            Vector::ZeroVector { dbg: dbg.clone() },
            Vector::OneVector { dbg: dbg.clone() },
        ],
        dbg: dbg,
    };
    assert_eq!(basis.make_explicit(), explicit_basis);
}

#[test]
fn test_basis_make_explicit_unit() {
    let dbg = Some(DebugLoc {
        file: "skippy.py".to_string(),
        line: 42,
        col: 420,
    });
    // [] -> []
    let basis = Basis::EmptyBasisLiteral { dbg };
    assert_eq!(basis.make_explicit(), basis);
}

#[test]
fn test_basis_make_explicit_tensor_pad1() {
    let dbg = Some(DebugLoc {
        file: "skippy.py".to_string(),
        line: 42,
        col: 420,
    });
    // {'?'} * {'1'} -> {'1'}
    let basis = Basis::BasisTensor {
        bases: vec![
            Basis::BasisLiteral {
                vecs: vec![Vector::PadVector { dbg: dbg.clone() }],
                dbg: dbg.clone(),
            },
            Basis::BasisLiteral {
                vecs: vec![Vector::OneVector { dbg: dbg.clone() }],
                dbg: dbg.clone(),
            },
        ],
        dbg: dbg.clone(),
    };
    let explicit_basis = Basis::BasisLiteral {
        vecs: vec![Vector::OneVector { dbg: dbg.clone() }],
        dbg: dbg,
    };
    assert_eq!(basis.make_explicit(), explicit_basis);
}

#[test]
fn test_basis_make_explicit_tensor_pad_pad() {
    let dbg = Some(DebugLoc {
        file: "skippy.py".to_string(),
        line: 42,
        col: 420,
    });
    // {'?'} * {'?'} -> []
    let basis = Basis::BasisTensor {
        bases: vec![
            Basis::BasisLiteral {
                vecs: vec![Vector::PadVector { dbg: dbg.clone() }],
                dbg: dbg.clone(),
            },
            Basis::BasisLiteral {
                vecs: vec![Vector::PadVector { dbg: dbg.clone() }],
                dbg: dbg.clone(),
            },
        ],
        dbg: dbg.clone(),
    };
    assert_eq!(basis.make_explicit(), Basis::EmptyBasisLiteral { dbg });
}

#[test]
fn test_basis_make_explicit_tensor_01() {
    let dbg = Some(DebugLoc {
        file: "skippy.py".to_string(),
        line: 42,
        col: 420,
    });
    // {'0'} * {'1'} -> {'0'} * {'1'}
    let basis = Basis::BasisTensor {
        bases: vec![
            Basis::BasisLiteral {
                vecs: vec![Vector::ZeroVector { dbg: dbg.clone() }],
                dbg: dbg.clone(),
            },
            Basis::BasisLiteral {
                vecs: vec![Vector::OneVector { dbg: dbg.clone() }],
                dbg: dbg.clone(),
            },
        ],
        dbg: dbg,
    };
    assert_eq!(basis.make_explicit(), basis);
}
