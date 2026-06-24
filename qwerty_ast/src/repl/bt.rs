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

use crate::ast::qpu::BasisTranslation;
use ndarray::Array2;

pub fn basis_tranlation_unitary(BasisTranslation ast) -> Result<Array2<Complex64>, NotImplementedError> 
{

}

//unit tests
#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::qpu::{BasisTranslation, QLit};
    use dashu::integer::UBig;
    use num_complex::Complex64;

    #[test]
    fn test_basis_translation_unitary() {
        let ast = qpu::Expr::BasisTranslation(BasisTranslation {
            bin: Basis::BasisLiteral {
                vecs: vec![Vector::ZeroVector { dbg: None }],
                dbg: None,
            },
            bout: Basis::BasisLiteral {
                vecs: vec![Vector::ZeroVector { dbg: None }], //zero to zero vector (identity)
                dbg: None,
            },
            dbg: None,
        });
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
