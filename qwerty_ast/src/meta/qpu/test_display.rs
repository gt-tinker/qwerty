use super::{FloatExpr, MetaVector};

#[test]
fn test_meta_vec_to_string_zero() {
    let vec = MetaVector::ZeroVector { dbg: None };
    assert_eq!(vec.to_string(), "'0'");
}

#[test]
fn test_meta_vec_to_string_one() {
    let vec = MetaVector::OneVector { dbg: None };
    assert_eq!(vec.to_string(), "'1'");
}

#[test]
fn test_meta_vec_to_string_pad() {
    let vec = MetaVector::PadVector { dbg: None };
    assert_eq!(vec.to_string(), "'?'");
}

#[test]
fn test_meta_vec_to_string_tgt() {
    let vec = MetaVector::TargetVector { dbg: None };
    assert_eq!(vec.to_string(), "'_'");
}

#[test]
fn test_meta_vec_to_string_tilt_180() {
    let vec = MetaVector::VectorTilt {
        q: Box::new(MetaVector::OneVector { dbg: None }),
        angle_deg: FloatExpr::FloatConst {
            val: 180.0,
            dbg: None,
        },
        dbg: None,
    };
    assert_eq!(vec.to_string(), "('1')@(180)");
}

#[test]
fn test_meta_vec_to_string_tilt_non_180() {
    let vec = MetaVector::VectorTilt {
        q: Box::new(MetaVector::OneVector { dbg: None }),
        angle_deg: FloatExpr::FloatConst {
            val: 1.23456,
            dbg: None,
        },
        dbg: None,
    };
    assert_eq!(vec.to_string(), "('1')@(1.23456)");
}

#[test]
fn test_meta_vec_to_string_superpos() {
    let vec = MetaVector::UniformVectorSuperpos {
        q1: Box::new(MetaVector::ZeroVector { dbg: None }),
        q2: Box::new(MetaVector::OneVector { dbg: None }),
        dbg: None,
    };
    assert_eq!(vec.to_string(), "('0') + ('1')");
}

#[test]
fn test_meta_vec_to_string_tensor_01() {
    let vec = MetaVector::VectorBiTensor {
        left: Box::new(MetaVector::ZeroVector { dbg: None }),
        right: Box::new(MetaVector::OneVector { dbg: None }),
        dbg: None,
    };
    assert_eq!(vec.to_string(), "('0') * ('1')");
}

#[test]
fn test_meta_vec_to_string_tensor_01pad() {
    let vec = MetaVector::VectorBiTensor {
        left: Box::new(MetaVector::ZeroVector { dbg: None }),
        right: Box::new(MetaVector::VectorBiTensor {
            left: Box::new(MetaVector::OneVector { dbg: None }),
            right: Box::new(MetaVector::PadVector { dbg: None }),
            dbg: None,
        }),
        dbg: None,
    };
    assert_eq!(vec.to_string(), "('0') * (('1') * ('?'))");
}

#[test]
fn test_meta_vec_to_string_vector_unit() {
    let vec = MetaVector::VectorUnit { dbg: None };
    assert_eq!(vec.to_string(), "''");
}
