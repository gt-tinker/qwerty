//! Unit tests pinning the canonical `Vector` shapes that the MLIR
//! basis-vector-tree transcription relies on. `ast_vec_to_tree`
//! (qwerty_ast_to_mlir) transcribes canonicalized vectors structurally, and
//! the C++ `tryFlatten` recognizes Pauli letters from these exact shapes
//! (e.g. 'm' = '0' + '1'@180). If canonicalization changes these forms, the
//! tree pipeline changes behavior -- these tests make that loud.

use super::*;
use qpu::Vector;

fn zero() -> Vector {
    Vector::ZeroVector { dbg: None }
}

fn one() -> Vector {
    Vector::OneVector { dbg: None }
}

fn tilt(q: Vector, angle_deg: f64) -> Vector {
    Vector::VectorTilt {
        q: Box::new(q),
        angle_deg,
        dbg: None,
    }
}

fn superpos(q1: Vector, q2: Vector) -> Vector {
    Vector::UniformVectorSuperpos {
        q1: Box::new(q1),
        q2: Box::new(q2),
        dbg: None,
    }
}

/// The canonical 'p' shape ('0' + '1') is a fixed point of canonicalization.
#[test]
fn test_canonicalize_p_shape_fixed_point() {
    let p = superpos(zero(), one());
    let canon = p.clone().canonicalize();
    assert!(
        canon.approx_equal(&p),
        "'0' + '1' should canonicalize to itself, got {canon:?}"
    );
}

/// The canonical 'm' shape ('0' + '1'@180) is a fixed point of
/// canonicalization. tryFlatten classifies X-MINUS from exactly this shape.
#[test]
fn test_canonicalize_m_shape_fixed_point() {
    let m = superpos(zero(), tilt(one(), 180.0));
    let canon = m.clone().canonicalize();
    assert!(
        canon.approx_equal(&m),
        "'0' + '1'@180 should canonicalize to itself, got {canon:?}"
    );
}

/// The canonical 'i'/'j' shapes (Y letters) are fixed points too.
#[test]
fn test_canonicalize_y_shapes_fixed_point() {
    for angle_deg in [90.0, 270.0] {
        let v = superpos(zero(), tilt(one(), angle_deg));
        let canon = v.clone().canonicalize();
        assert!(
            canon.approx_equal(&v),
            "'0' + '1'@{angle_deg} should canonicalize to itself, got {canon:?}"
        );
    }
}

/// Canonicalization is idempotent on the shapes the tree transcription
/// feeds on (tensors of letters, tilted letters).
#[test]
fn test_canonicalize_idempotent_on_tree_shapes() {
    let shapes = vec![
        Vector::VectorTensor {
            qs: vec![zero(), superpos(zero(), one())],
            dbg: None,
        },
        tilt(one(), 45.0),
        superpos(one(), tilt(zero(), 180.0)),
    ];
    for v in shapes {
        let once = v.clone().canonicalize();
        let twice = once.clone().canonicalize();
        assert!(
            twice.approx_equal(&once),
            "canonicalize not idempotent for {v:?}: {once:?} vs {twice:?}"
        );
    }
}
