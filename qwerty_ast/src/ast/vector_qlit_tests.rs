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
