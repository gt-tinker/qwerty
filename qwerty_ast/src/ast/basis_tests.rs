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
        vecs: vec![
            Vector::ZeroVector { dbg: None },
        ],
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
                vecs: vec![
                    Vector::ZeroVector { dbg: None },
                ],
                dbg: None,
            },
            Basis::BasisLiteral {
                vecs: vec![
                    Vector::OneVector { dbg: None },
                ],
                dbg: None,
            },
        ],
        dbg: Some(dbg.clone()),
    };
    assert_eq!(basis.get_dbg(), Some(dbg));
}
