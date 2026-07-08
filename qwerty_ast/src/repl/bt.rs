/*
Input: b1 >> b2, where
       b1 == {bv1, bv2, ..., bvn} and b2 == {bv1', bv2', ..., bvn'}
Output: 2^m x 2^m unitary that achieves b1>>b2
    where m is the number of qubits for any bvi or any bvi'

P_W = |bv1⟩⟨bv1| + |bv2⟩⟨bv2| + ... + |bvn⟩⟨bvn|
P_U = I - P_W
M = |bv1'⟩⟨bv1| + |bv2'⟩⟨bv2| + ... + |bvn'⟩⟨bvn|
return M + P_U
*/

use crate::ast::qpu::{BasisTranslation, Basis, Vector};
use ndarray::{Array2, array};
use num_complex::Complex64;
use super::NotImplementedError;

/*
Helper function to convert a Basis into a vector of dense column vectors (Array2<Complex64>).
This function recursively processes the Basis structure, handling different types of Basis representations,
including BasisLiteral, EmptyBasisLiteral, BasisTensor, and ApplyBasisGenerator. The resulting column
vectors are returned as a vector of Array2<Complex64>.
*/
pub fn basis_to_dense_vectors(basis: &Basis) -> Result<Vec<Array2<Complex64>>, NotImplementedError> {
    match basis {
        Basis::BasisLiteral { vecs, .. } => {
            let mut result = Vec::new();
            for vec in vecs {
                result.push(ast_vec_to_col_vec(vec.clone())?);
            }
            Ok(result)
        }
        Basis::EmptyBasisLiteral { .. } => {
            Ok(vec![])
        }
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
        Basis::ApplyBasisGenerator { .. } => {
            Err(NotImplementedError("Basis generators are not implemented in REPL basis translation".to_string()))
        }
    }
}

pub fn basis_tranlation_unitary(ast: BasisTranslation) -> Result<Array2<Complex64>, NotImplementedError> 
{
    let bin_vecs = basis_to_dense_vectors(&ast.bin)?;
    let bout_vecs = basis_to_dense_vectors(&ast.bout)?;

    if bin_vecs.is_empty() || bout_vecs.is_empty() 
    {
        return Err(NotImplementedError("Basis vectors can't be empty".to_string()));
    }

    if bin_vecs.len() != bout_vecs.len() 
    {
        return Err(NotImplementedError("Input and output bases must have the same number of vectors".to_string()));
    }

    //1. Get dimension from the first element 
    let dim = bin_vecs[0].nrows();

    //2. initialize our M and P_W matrices to zero matrices of size dim x dim
    let mut m_matrix = Array2::<Complex64>::zeros((dim, dim));
    let mut p_w_matrix = Array2::<Complex64>::zeros((dim, dim));

    //3. Time for the steps 1-2 from the paper, computing M and P_W in a single pass
    for (bv_in, bv_out) in bin_vecs.iter().zip(bout_vecs.iter()) 
    {
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


//reference basisis translation paper to understand why we need this
pub fn ast_vec_to_col_vec(input: Vector) -> Result<Array2<Complex64>, NotImplementedError> 
{
    //I just realized since we're returning type of Array2<Complex64> we need
    //to use array! macro with Complex64 types
    let c1 = Complex64::new(1.0, 0.0);
    let c0 = Complex64::new(0.0, 0.0);
    match input 
    {
        /*implmeent for all these
            /// The first standard basis vector, |0⟩. Example syntax:
            /// ```text
            /// '0'
            /// ```
            ZeroVector { dbg: Option<DebugLoc> },

            /// The second standard basis vector, |1⟩. Example syntax:
            /// ```text
            /// '1'
            /// ```
            OneVector { dbg: Option<DebugLoc> },

            /// The pad atom. Example syntax:
            /// ```text
            /// '?'
            /// ```
            PadVector { dbg: Option<DebugLoc> },

            /// The target atom. Example syntax:
            /// ```text
            /// '_'
            /// ```
            TargetVector { dbg: Option<DebugLoc> },

            /// Tilts a vector. Example syntax:
            /// ```text
            /// '1' @ 180
            /// ```
            VectorTilt {
                q: Box<Vector>,
                angle_deg: f64,
                dbg: Option<DebugLoc>,
            },

            /// A uniform vector superposition. Example syntax:
            /// ```text
            /// '0' + '1'
            /// ```
            UniformVectorSuperpos {
                q1: Box<Vector>,
                q2: Box<Vector>,
                dbg: Option<DebugLoc>,
            },

            /// A tensor product. Example syntax:
            /// ```text
            /// '0' * '1'
            /// ```
            VectorTensor {
                qs: Vec<Vector>,
                dbg: Option<DebugLoc>,
            },

            /// An empty vector. Example syntax:
            /// ```text
            /// ''
            /// ```
            VectorUnit { dbg: Option<DebugLoc> },
        */

        /*
        Basically we're trying to implement Figure 12 from QWERTY spec overleaf
        */

        Vector::ZeroVector { dbg:_ } => 
        {
            //TODO: create matrix like in numpy [[1], [0]]
            // https://docs.rs/ndarray/latest/ndarray/macro.array.html
            // macro included in this library is useful for creating a column vector
            Ok(array![[c1], [c0]])
        }

        Vector::VectorTilt { dbg:_, q, angle_deg } => 
        {
            let q_vec = ast_vec_to_col_vec(*q)?;

            let angle_rad = angle_deg * std::f64::consts::PI / 180.0;

            Ok(Complex64::cis(angle_rad) * q_vec)
        }

        Vector::OneVector { dbg:_ } => 
        {
            //ket 1
            Ok(array![[c0], [c1]])
        }

        //TODO: Get the following code reviewed by austin
        Vector::PadVector { dbg:_ } => 
        {
            //Ok so having a pad vector is a bit wierd, since we need it to preserve quantum state, 
            // we need to input KET 0 not pure 2x1 zero vector
            Ok(array![[c1], [c0]])
        }

        Vector::TargetVector { dbg:_ } => 
        {
            //since we need to preserve state, ket 0 will be used
            Ok(array![[c1], [c0]])
        }

        Vector::UniformVectorSuperpos { dbg: _, q1, q2 } => 
        {
            let q1_vec = ast_vec_to_col_vec(*q1)?;
            let q2_vec = ast_vec_to_col_vec(*q2)?;

            // Enforce Fig. 13 rule: dimensions must match exactly
            if q1_vec.shape() != q2_vec.shape() {
                return Err(NotImplementedError(
                    "Superposition components must have identical dimensions".to_string(),
                ));
            }

            // Combine states and normalize by 1 / sqrt(2)
            let factor = Complex64::new(2.0_f64.sqrt().recip(), 0.0);
            Ok((q1_vec + q2_vec) * factor)
        }

        Vector::VectorUnit { dbg:_ } => 
        {
            //the unit vector is just 1x1 matrix containing 1 I think
            Ok(array![[c1]])
        }

        Vector::VectorTensor { dbg: _, qs } => 
        {
            if qs.is_empty() {
                return Ok(array![[c1]]); // Fallback to identity if empty
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

//unit tests
#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::qpu::BasisTranslation;
    use num_complex::Complex64;


    #[test]
    fn test_basis_translation_unitary() {
        let ast = BasisTranslation {
            bin: Basis::BasisLiteral {
                vecs: vec![Vector::ZeroVector { dbg: None }],
                dbg: None,
            },
            bout: Basis::BasisLiteral {
                vecs: vec![Vector::ZeroVector { dbg: None }], //zero to zero vector (identity)
                dbg: None,
            },
            dbg: None,
        };
        let unitary = basis_tranlation_unitary(ast).unwrap();
        // Check that the unitary is correct (this is just a placeholder check)
        assert_eq!(unitary.shape(), &[2, 2]);

        //TODO: check contents (shape)
        /*
        [1,0]
        [0,1]
        */

        let identity_matrix = Array2::<Complex64>::eye(2);
        assert_eq!(unitary, identity_matrix);
    }
}
