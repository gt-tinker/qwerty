use super::NotImplementedError;
use crate::ast::qpu::{Basis, BasisTranslation, Vector};
use ndarray::{Array2, array};
use num_complex::Complex64;

/// Helper function to convert a Basis into a vector of dense column vectors (Array2<Complex64>).
/// This function recursively processes the Basis structure, handling different types of Basis representations,
/// including BasisLiteral, EmptyBasisLiteral, BasisTensor, and ApplyBasisGenerator. The resulting column
/// vectors are returned as a vector of Array2<Complex64>.
pub fn basis_to_dense_vectors(
    basis: &Basis,
) -> Result<Vec<Array2<Complex64>>, NotImplementedError> {
    match basis {
        Basis::BasisLiteral { vecs, .. } => {
            let mut result = Vec::new();
            for vec in vecs {
                result.push(ast_vec_to_col_vec(vec.clone())?);
            }
            Ok(result)
        }
        Basis::EmptyBasisLiteral { .. } => Ok(vec![]),
        Basis::BasisTensor { bases, .. } => {
            if bases.is_empty() {
                return Ok(vec![]);
            }
            let mut result = basis_to_dense_vectors(&bases[0])?;
            for b in &bases[1..] {
                let next_vectors = basis_to_dense_vectors(b)?;
                let mut combined = Vec::new();
                for v1 in &result {
                    for v2 in &next_vectors {
                        combined.push(ndarray::linalg::kron(&v1.view(), &v2.view()));
                    }
                }
                result = combined;
            }
            Ok(result)
        }
        Basis::ApplyBasisGenerator { .. } => Err(NotImplementedError(
            "Basis generators are not implemented in REPL basis translation".to_string(),
        )),
    }
}

/// Input: b1 >> b2, where
///     b1 == {bv1, bv2, ..., bvn} and b2 == {bv1', bv2', ..., bvn'}
/// Output: 2^m x 2^m unitary that achieves b1>>b2
///     where m is the number of qubits for any bvi or any bvi'
///
/// P_W = |bv1⟩⟨bv1| + |bv2⟩⟨bv2| + ... + |bvn⟩⟨bvn|
/// P_U = I - P_W
/// M = |bv1'⟩⟨bv1| + |bv2'⟩⟨bv2| + ... + |bvn'⟩⟨bvn|
/// return M + P_U
pub fn basis_translation_unitary(
    ast: BasisTranslation,
) -> Result<Array2<Complex64>, NotImplementedError> {
    let bin_vecs = basis_to_dense_vectors(&ast.bin)?;
    let bout_vecs = basis_to_dense_vectors(&ast.bout)?;

    if bin_vecs.is_empty() || bout_vecs.is_empty() {
        return Err(NotImplementedError(
            "Basis vectors can't be empty".to_string(),
        ));
    }

    if bin_vecs.len() != bout_vecs.len() {
        return Err(NotImplementedError(
            "Input and output bases must have the same number of vectors".to_string(),
        ));
    }

    // 1. Get dimension from the first element
    let dim = bin_vecs[0].nrows();

    // 2. initialize our M and P_W matrices to zero matrices of size dim x dim
    let mut m_matrix = Array2::<Complex64>::zeros((dim, dim));
    let mut p_w_matrix = Array2::<Complex64>::zeros((dim, dim));

    // 3. Time for the steps 1-2 from the paper (/docs/basis-translation.md), computing M and P_W in a single pass
    for (bv_in, bv_out) in bin_vecs.iter().zip(bout_vecs.iter()) {
        // to get a row vector from the column vector,
        // need to take the conjugate transpose (take transpose,
        // then take complex conjugate of every entry, or equivalently vice versa)
        let row_in = bv_in.t().mapv(|x| x.conj());

        // cross-paired outer product : |bv'_i⟩⟨bv_i|, store in M
        m_matrix += &bv_out.dot(&row_in);

        // self-paired outer product: |bv_i⟩⟨bv_i|, store in P_W
        p_w_matrix += &bv_in.dot(&row_in);
    }

    // 4. Compute P_U = I - P_W
    let identity_matrix = Array2::<Complex64>::eye(dim);
    let p_u_matrix = identity_matrix - p_w_matrix;

    // 5. Step 3 from paper: U_total = M + P_U
    let u_total_matrix = m_matrix + p_u_matrix;

    Ok(u_total_matrix)
}

/// reference /docs/basis-translation.md to understand why we need this
/// Basically we're trying to implement Figure 12 from QWERTY spec overleaf
pub fn ast_vec_to_col_vec(input: Vector) -> Result<Array2<Complex64>, NotImplementedError> {
    let c1 = Complex64::ONE;
    let c0 = Complex64::ZERO;
    match input {
        Vector::ZeroVector { dbg: _ } => Ok(array![[c1], [c0]]),

        Vector::VectorTilt {
            dbg: _,
            q,
            angle_deg,
        } => {
            let q_vec = ast_vec_to_col_vec(*q)?;

            let angle_rad = angle_deg * std::f64::consts::PI / 180.0;

            Ok(Complex64::cis(angle_rad) * q_vec)
        }

        Vector::OneVector { dbg: _ } => {
            //ket 1
            Ok(array![[c0], [c1]])
        }

        Vector::PadVector { dbg: _ } => {
            //Ok so having a pad vector is a bit wierd, since we need it to preserve quantum state,
            // we need to input KET 0 not pure 2x1 zero vector
            Ok(array![[Complex64::ONE]])
        }

        Vector::TargetVector { dbg: _ } => Ok(array![[Complex64::ONE]]),

        Vector::UniformVectorSuperpos { dbg: _, q1, q2 } => {
            let q1_vec = ast_vec_to_col_vec(*q1)?;
            let q2_vec = ast_vec_to_col_vec(*q2)?;

            // Enforce Fig. 13 rule: dimensions must match exactly
            if q1_vec.shape() == q2_vec.shape() {
                // Combine states and normalize by 1 / sqrt(2)
                let factor = Complex64::new(std::f64::consts::FRAC_1_SQRT_2, 0.0);
                Ok((q1_vec + q2_vec) * factor)
            } else {
                Err(NotImplementedError(
                    "Superposition components must have identical dimensions".to_string(),
                ))
            }
        }

        Vector::VectorUnit { dbg: _ } => Ok(array![[c1]]),

        Vector::VectorTensor { dbg: _, qs } => {
            if qs.is_empty() {
                return Ok(array![[c1]]); // Fallback to |> if empty
            }

            let mut result = ast_vec_to_col_vec(qs[0].clone())?;

            for q in &qs[1..] {
                let next_vec = ast_vec_to_col_vec(q.clone())?;
                result = ndarray::linalg::kron(&result.view(), &next_vec.view());
            }
            Ok(result)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::qpu::BasisTranslation;
    use ndarray::Array2;
    use num_complex::Complex64;

    const EPS: f64 = 1e-10;

    /// Check two matrices are element-wise approximately equal within `EPS`.
    fn approx_eq(a: &Array2<Complex64>, b: &Array2<Complex64>) -> bool {
        if a.shape() != b.shape() {
            return false;
        }
        a.iter().zip(b.iter()).all(|(x, y)| (x - y).norm() < EPS)
    }

    // helpers for building basis vectors

    fn zero() -> Vector {
        Vector::ZeroVector { dbg: None }
    }
    fn one() -> Vector {
        Vector::OneVector { dbg: None }
    }

    /// '0' + '1'  (uniform superposition, i.e. |+⟩)
    fn plus() -> Vector {
        Vector::UniformVectorSuperpos {
            q1: Box::new(zero()),
            q2: Box::new(one()),
            dbg: None,
        }
    }

    /// '0' - '1'  (|−⟩  =  |0⟩ + e^{iπ}|1⟩)
    fn minus() -> Vector {
        Vector::UniformVectorSuperpos {
            q1: Box::new(zero()),
            q2: Box::new(Vector::VectorTilt {
                q: Box::new(one()),
                angle_deg: 180.0,
                dbg: None,
            }),
            dbg: None,
        }
    }

    /// Tensor two single-qubit vectors into a two-qubit ket, e.g. '01' = |0⟩⊗|1⟩.
    fn tensor2(v1: Vector, v2: Vector) -> Vector {
        Vector::VectorTensor {
            qs: vec![v1, v2],
            dbg: None,
        }
    }

    fn btrans(bin_vecs: Vec<Vector>, bout_vecs: Vec<Vector>) -> BasisTranslation {
        BasisTranslation {
            bin: Basis::BasisLiteral {
                vecs: bin_vecs,
                dbg: None,
            },
            bout: Basis::BasisLiteral {
                vecs: bout_vecs,
                dbg: None,
            },
            dbg: None,
        }
    }

    #[test]
    fn test_basis_translation_unitary() {
        // {'0'} >> {'0'}
        let ast = BasisTranslation {
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
        let unitary = basis_translation_unitary(ast).unwrap();
        let identity_matrix = Array2::<Complex64>::eye(2);
        assert_eq!(unitary, identity_matrix);
    }

    // test 1: {'0','1'} >> {'0'+'1', '0'-'1'}

    #[test]
    fn test_std_to_hadamard_basis() {
        // {'0','1'} >> {'0'+'1', '0'-'1'} should produce (1/√2)[[1,1],[1,-1]]
        let ast = btrans(vec![zero(), one()], vec![plus(), minus()]);
        let u = basis_translation_unitary(ast).unwrap();

        let s = std::f64::consts::FRAC_1_SQRT_2;
        let expected: Array2<Complex64> = ndarray::array![
            [Complex64::new(s, 0.0), Complex64::new(s, 0.0)],
            [Complex64::new(s, 0.0), Complex64::new(-s, 0.0)],
        ];
        assert!(
            approx_eq(&u, &expected),
            "test 1 failed:\n{u:?}\n≠\n{expected:?}"
        );
    }

    // test 2: {'0'+'1', '0'-'1'} >> {'0','1'}

    #[test]
    fn test_hadamard_basis_to_std() {
        // {'0'+'1', '0'-'1'} >> {'0','1'}
        let ast = btrans(vec![plus(), minus()], vec![zero(), one()]);
        let u = basis_translation_unitary(ast).unwrap();

        let s = std::f64::consts::FRAC_1_SQRT_2;
        let expected: Array2<Complex64> = ndarray::array![
            [Complex64::new(s, 0.0), Complex64::new(s, 0.0)],
            [Complex64::new(s, 0.0), Complex64::new(-s, 0.0)],
        ];
        assert!(
            approx_eq(&u, &expected),
            "test 2 failed:\n{u:?}\n≠\n{expected:?}"
        );
    }

    // test 3: {'0','1'} >> {'1','0'}

    #[test]
    fn test_std_to_swapped_std() {
        // {'0','1'} >> {'1', '0'} should produce [[0,1],[1,0]]
        let ast = btrans(vec![zero(), one()], vec![one(), zero()]);
        let u = basis_translation_unitary(ast).unwrap();

        let c0 = Complex64::ZERO;
        let c1 = Complex64::ONE;
        let expected: Array2<Complex64> = ndarray::array![[c0, c1], [c1, c0]];
        assert!(
            approx_eq(&u, &expected),
            "test 3 failed:\n{u:?}\n≠\n{expected:?}"
        );
    }

    // test 4: {'1','0'} >> {'0','1'}

    #[test]
    fn test_swapped_std_to_std() {
        // {'1', '0'} >> {'0','1'} should also produce [[0,1],[1,0]]
        let ast = btrans(vec![one(), zero()], vec![zero(), one()]);
        let u = basis_translation_unitary(ast).unwrap();

        let c0 = Complex64::ZERO;
        let c1 = Complex64::ONE;
        let expected: Array2<Complex64> = ndarray::array![[c0, c1], [c1, c0]];
        assert!(
            approx_eq(&u, &expected),
            "test 4 failed:\n{u:?}\n≠\n{expected:?}"
        );
    }

    // test 5: {'01','10'} >> {'10','01'}

    #[test]
    fn test_2q_swap_01_10_to_10_01() {
        // {'01','10'} >> {'10','01'}
        let ast = btrans(
            vec![tensor2(zero(), one()), tensor2(one(), zero())],
            vec![tensor2(one(), zero()), tensor2(zero(), one())],
        );
        let u = basis_translation_unitary(ast).unwrap();

        let (o, i) = (Complex64::ZERO, Complex64::ONE);
        let expected: Array2<Complex64> =
            ndarray::array![[i, o, o, o], [o, o, i, o], [o, i, o, o], [o, o, o, i],];
        assert!(
            approx_eq(&u, &expected),
            "test 5 failed:\n{u:?}\n≠\n{expected:?}"
        );
    }

    // test 6: {'10','01'} >> {'01','10'}

    #[test]
    fn test_2q_swap_10_01_to_01_10() {
        // {'10', '01'} >> {'01', '10'}
        let ast = btrans(
            vec![tensor2(one(), zero()), tensor2(zero(), one())],
            vec![tensor2(zero(), one()), tensor2(one(), zero())],
        );
        let u = basis_translation_unitary(ast).unwrap();

        let (o, i) = (Complex64::ZERO, Complex64::ONE);
        let expected: Array2<Complex64> =
            ndarray::array![[i, o, o, o], [o, o, i, o], [o, i, o, o], [o, o, o, i],];
        assert!(
            approx_eq(&u, &expected),
            "test 6 failed:\n{u:?}\n≠\n{expected:?}"
        );
    }

    // test 7: {'10','01','11'} >> {'01','10','11'}

    #[test]
    fn test_2q_swap_with_11_unchanged() {
        // {'10', '01', '11'} >> {'01', '10', '11'}
        let ast = btrans(
            vec![
                tensor2(one(), zero()),
                tensor2(zero(), one()),
                tensor2(one(), one()),
            ],
            vec![
                tensor2(zero(), one()),
                tensor2(one(), zero()),
                tensor2(one(), one()),
            ],
        );
        let u = basis_translation_unitary(ast).unwrap();

        let (o, i) = (Complex64::ZERO, Complex64::ONE);
        let expected: Array2<Complex64> =
            ndarray::array![[i, o, o, o], [o, o, i, o], [o, i, o, o], [o, o, o, i],];
        assert!(
            approx_eq(&u, &expected),
            "test 7 failed:\n{u:?}\n≠\n{expected:?}"
        );
    }

    // test 8: {'10','11'} >> {'11','10'}

    #[test]
    fn test_2q_swap_10_11_to_11_10() {
        // {'10','11'} >> {'11','10'}
        let ast = btrans(
            vec![tensor2(one(), zero()), tensor2(one(), one())],
            vec![tensor2(one(), one()), tensor2(one(), zero())],
        );
        let u = basis_translation_unitary(ast).unwrap();

        let (o, i) = (Complex64::ZERO, Complex64::ONE);
        let expected: Array2<Complex64> =
            ndarray::array![[i, o, o, o], [o, i, o, o], [o, o, o, i], [o, o, i, o],];
        assert!(
            approx_eq(&u, &expected),
            "test 8 failed:\n{u:?}\n≠\n{expected:?}"
        );
    }

    // test 10: {'11','10'} >> {'10','11'}

    #[test]
    fn test_2q_swap_11_10_to_10_11() {
        // {'11', '10'} >> {'10', '11'}
        let ast = btrans(
            vec![tensor2(one(), one()), tensor2(one(), zero())],
            vec![tensor2(one(), zero()), tensor2(one(), one())],
        );
        let u = basis_translation_unitary(ast).unwrap();

        let (o, i) = (Complex64::ZERO, Complex64::ONE);
        let expected: Array2<Complex64> =
            ndarray::array![[i, o, o, o], [o, i, o, o], [o, o, o, i], [o, o, i, o],];
        assert!(
            approx_eq(&u, &expected),
            "test 10 failed:\n{u:?}\n≠\n{expected:?}"
        );
    }

    // test 11: {'00','01','10','11'} >> {'00','01','11','10'}

    #[test]
    fn test_2q_full_basis_swap_10_11() {
        // {'00','01','10','11'} >> {'00','01','11','10'}
        let ast = btrans(
            vec![
                tensor2(zero(), zero()),
                tensor2(zero(), one()),
                tensor2(one(), zero()),
                tensor2(one(), one()),
            ],
            vec![
                tensor2(zero(), zero()),
                tensor2(zero(), one()),
                tensor2(one(), one()),
                tensor2(one(), zero()),
            ],
        );
        let u = basis_translation_unitary(ast).unwrap();

        let (o, i) = (Complex64::ZERO, Complex64::ONE);
        let expected: Array2<Complex64> =
            ndarray::array![[i, o, o, o], [o, i, o, o], [o, o, o, i], [o, o, i, o],];
        assert!(
            approx_eq(&u, &expected),
            "test 11 failed:\n{u:?}\n≠\n{expected:?}"
        );
    }
}
