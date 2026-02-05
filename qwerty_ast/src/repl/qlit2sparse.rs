use std::f64::consts::PI;
use std::f64::consts::FRAC_1_SQRT_2;
use dashu::integer::UBig;
use crate::ast::qpu::QLit;
use num_complex::Complex64;

impl QLit {

    pub fn to_sparse(&self) -> SparseVec {
        match &self {
            QLit::ZeroQubit { .. } => SparseVec::newZero(),
            QLit::OneQubit { .. } => SparseVec::newOne(),
            QLit::QubitTilt { q, angle_deg, .. } => {
                let z = Complex64::cis(*angle_deg * PI / 180.0);
                q.to_sparse().scale(z)
            }
            QLit::UniformSuperpos { q1, q2, .. } => {
                q1.to_sparse().add(q2.to_sparse()).scale(Complex64::new(FRAC_1_SQRT_2, 0.0))
            }
            QLit::QubitTensor { qs, .. } => {
                qs.iter()
                    .map(|x| x.to_sparse())
                    .reduce(|v1, v2| v1.kron(v2))
                    .unwrap_or_else(SparseVec::newZero)
            }
            QLit::QubitUnit { .. } => { SparseVec::newEmpty() }
        }
    }
}

// Invariant: Vec is sorted by UBig
#[derive(Debug, Clone)]
struct SparseVec {
    v: Vec<(UBig, Complex64)>,
    num_qbits: usize
}
impl SparseVec {
    pub fn newEmpty() -> Self {
        SparseVec { v: vec![], num_qbits: 0 }
    }

    pub fn newZero() -> Self {
        SparseVec { v: vec![(UBig::ZERO, Complex64::ONE)], num_qbits: 1 }
    }

    pub fn newOne() -> Self {
        SparseVec { v: vec![(UBig::ONE, Complex64::ONE)], num_qbits: 1 }
    }

    pub fn scale(self, z: Complex64) -> Self {
        SparseVec { v: self.v.into_iter().map(|(u, v)| (u, v * z)).collect(), num_qbits: self.num_qbits }
    }

    pub fn add(self, q: SparseVec) -> Self {
        // // Invariant: vector is sorted by UBig
        let mut result = Vec::new();
        let mut i = 0;
        let mut j = 0;

        while i < self.v.len() && j < q.v.len() {
            match self.v[i].0.cmp(&q.v[j].0) {
                std::cmp::Ordering::Less => {
                    result.push(self.v[i].clone());
                    i += 1;
                }
                std::cmp::Ordering::Greater => {
                    result.push(q.v[j].clone());
                    j += 1;
                }
                std::cmp::Ordering::Equal => {
                    let sum = self.v[i].1 + q.v[j].1;
                    if sum.norm_sqr() > f64::EPSILON {
                        result.push((self.v[i].0.clone(), sum));
                    }
                    i += 1;
                    j += 1;
                }
            }
        }

        while i < self.v.len() {
            result.push(self.v[i].clone());
            i += 1;
        }
        while j < q.v.len() {
            result.push(q.v[j].clone());
            j += 1;
        }

        SparseVec { v: result, num_qbits: self.num_qbits.max(q.num_qbits) }
    }

    pub fn kron(self, vec: SparseVec) -> Self {
        let v = self.v.into_iter()
            .flat_map(
                move |(x, z)|
                    vec.v.clone().into_iter().map(
                        move |(x2, z2)| (x.clone() << vec.num_qbits | x2, z * z2)))
            .collect();
        SparseVec { v, num_qbits: self.num_qbits + vec.num_qbits }
    }
}
