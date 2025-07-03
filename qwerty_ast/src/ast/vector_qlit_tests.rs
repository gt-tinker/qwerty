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
