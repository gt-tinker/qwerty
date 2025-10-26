use crate::{
    ctx::{BoundVals, Ctx, MLIR_CTX},
    lower::prog_stmt::{Lowerable, ast_stmt_to_mlir},
    lower::ty::{ast_func_mlir_ty, ast_ty_to_mlir_tys, dbg_to_loc},
};
use dashu::{base::BitTest, integer::UBig};
use melior::{
    dialect::{arith, qcirc, qwerty, scf},
    ir::{
        self, Block, BlockLike, Location, Operation, OperationLike, Region, RegionLike, Type,
        TypeLike, Value, ValueLike,
        attribute::{
            FlatSymbolRefAttribute, FloatAttribute, IntegerAttribute, StringAttribute,
            TypeAttribute,
        },
        operation::OperationResult,
        symbol_table::Visibility,
        r#type::{FunctionType, IntegerType},
    },
};
use qwerty_ast::{
    ast::{
        self, BitLiteral, FunctionDef, RegKind, Variable, angle_is_approx_zero,
        angles_are_approx_equal,
        qpu::{
            self, Adjoint, Basis, BasisGenerator, BasisTranslation, Conditional, Discard,
            EmbedClassical, EmbedKind, Ensemble, Measure, NonUniformSuperpos, Pipe, Predicated,
            QLit, Tensor, Vector, VectorAtomKind,
        },
    },
    typecheck::{ComputeKind, FuncsAvailable},
};

impl Lowerable for qpu::Expr {
    fn lower_to_mlir(
        &self,
        ctx: &mut Ctx,
        block: &Block<'static>,
    ) -> (ast::Type, ComputeKind, Vec<Value<'static, 'static>>) {
        ast_qpu_expr_to_mlir(self, ctx, block)
    }

    fn build_return(
        vals: &[Value<'static, 'static>],
        loc: Location<'static>,
    ) -> Operation<'static> {
        qwerty::r#return(vals, loc)
    }

    fn bit_register_ty(dim: usize) -> ir::Type<'static> {
        qwerty::BitBundleType::new(&MLIR_CTX, dim.try_into().unwrap()).into()
    }

    fn build_bit_register_unpack(
        reg: Value<'static, 'static>,
        loc: Location<'static>,
    ) -> Operation<'static> {
        qwerty::bitunpack(reg, loc)
    }
}

/// Converts degrees (used in the AST) to radians (used in MLIR)
fn deg_to_rad(angle_deg: f64) -> f64 {
    angle_deg / 360.0 * 2.0 * std::f64::consts::PI
}

/// Creates a wrapper quantum.calc op for the operation(s) of your choosing.
fn mlir_wrap_calc<F>(
    root_block: &Block<'static>,
    loc: Location<'static>,
    f: F,
) -> Value<'static, 'static>
where
    F: FnOnce(&Block<'static>) -> Value<'static, 'static>,
{
    let calc_block = Block::new(&[]);
    let val_to_yield = f(&calc_block);
    calc_block.append_operation(qcirc::calc_yield(&[val_to_yield], loc));

    let calc_region = Region::new();
    calc_region.append_block(calc_block);

    root_block
        .append_operation(qcirc::calc(calc_region, &[val_to_yield.r#type()], loc))
        .result(0)
        .unwrap()
        .into()
}

/// Returns an f64 mlir::Value defined inside a quantum.calc op.
fn mlir_f64_const(
    theta: f64,
    loc: Location<'static>,
    root_block: &Block<'static>,
) -> Value<'static, 'static> {
    mlir_wrap_calc(root_block, loc, |block| {
        block
            .append_operation(arith::constant(
                &MLIR_CTX,
                FloatAttribute::new(&MLIR_CTX, Type::float64(&MLIR_CTX), theta).into(),
                loc,
            ))
            .result(0)
            .unwrap()
            .into()
    })
}

/// Creates a wrapper qwerty.lambda op for some operations of your choosing.
fn mlir_wrap_lambda<F>(
    captures: &[Value<'static, 'static>],
    in_tys: &[ir::Type<'static>],
    out_tys: &[ir::Type<'static>],
    is_rev: bool,
    loc: Location<'static>,
    block: &Block<'static>,
    f: F,
) -> Value<'static, 'static>
where
    F: FnOnce(&Block<'static>) -> Vec<Value<'static, 'static>>,
{
    let lambda_ty = qwerty::FunctionType::new(
        &MLIR_CTX,
        FunctionType::new(&MLIR_CTX, in_tys, out_tys),
        is_rev,
    );

    let lambda_block_args = captures
        .iter()
        .map(|cap_val| cap_val.r#type())
        .chain(in_tys.iter().copied())
        .map(|arg_ty| (arg_ty, loc))
        .collect::<Vec<_>>();
    let lambda_block = Block::new(&lambda_block_args);
    let vals_to_yield = f(&lambda_block);
    lambda_block.append_operation(qwerty::r#return(&vals_to_yield, loc));

    let lambda_region = Region::new();
    lambda_region.append_block(lambda_block);

    block
        .append_operation(qwerty::lambda(captures, lambda_ty, lambda_region, loc))
        .result(0)
        .unwrap()
        .into()
}

/// Determines the primitive basis, eigenstate, and phase for a basis vector.
/// The basis vector should be explicit (not contain any `'?'` or `'_'` atoms).
/// Intended to be used only by `ast_vec_to_mlir()`, since it canonicalizes the
/// vector, removes any outer tilt node, and calls [`Vector::make_explicit`].
fn ast_vec_to_mlir_helper(vec: &Vector) -> (qwerty::PrimitiveBasis, qwerty::Eigenstate, f64) {
    match vec {
        Vector::ZeroVector { .. } => (qwerty::PrimitiveBasis::Z, qwerty::Eigenstate::Plus, 0.0),

        Vector::OneVector { .. } => (qwerty::PrimitiveBasis::Z, qwerty::Eigenstate::Minus, 0.0),

        Vector::UniformVectorSuperpos { q1, q2, .. } => match (&**q1, &**q2) {
            // '0' + '1' ==> 'p'
            (Vector::ZeroVector { .. }, Vector::OneVector { .. }) => {
                (qwerty::PrimitiveBasis::X, qwerty::Eigenstate::Plus, 0.0)
            }

            // '0' + '1'@180 ==> 'm'
            (Vector::ZeroVector { .. }, Vector::VectorTilt { q, angle_deg, .. })
                if angles_are_approx_equal(*angle_deg, 180.0)
                    && matches!(**q, Vector::OneVector { .. }) =>
            {
                (qwerty::PrimitiveBasis::X, qwerty::Eigenstate::Minus, 0.0)
            }

            // '1' + '0'@180 ==> -'m'
            (Vector::OneVector { .. }, Vector::VectorTilt { q, angle_deg, .. })
                if angles_are_approx_equal(*angle_deg, 180.0)
                    && matches!(**q, Vector::ZeroVector { .. }) =>
            {
                (
                    qwerty::PrimitiveBasis::X,
                    qwerty::Eigenstate::Minus,
                    std::f64::consts::PI,
                )
            }

            // '0'@180 + '1'@180 ==> -'p'
            (
                Vector::VectorTilt {
                    q: q1,
                    angle_deg: angle_deg1,
                    ..
                },
                Vector::VectorTilt {
                    q: q2,
                    angle_deg: angle_deg2,
                    ..
                },
            ) if angles_are_approx_equal(*angle_deg1, 180.0)
                && angles_are_approx_equal(*angle_deg2, 180.0)
                && matches!(**q1, Vector::ZeroVector { .. })
                && matches!(**q2, Vector::OneVector { .. }) =>
            {
                (
                    qwerty::PrimitiveBasis::X,
                    qwerty::Eigenstate::Plus,
                    std::f64::consts::PI,
                )
            }

            // '0' + '1'@90 ==> 'i'
            (Vector::ZeroVector { .. }, Vector::VectorTilt { q, angle_deg, .. })
                if angles_are_approx_equal(*angle_deg, 90.0)
                    && matches!(**q, Vector::OneVector { .. }) =>
            {
                (qwerty::PrimitiveBasis::Y, qwerty::Eigenstate::Plus, 0.0)
            }

            // '0' + '1'@270 ==> 'j'
            (Vector::ZeroVector { .. }, Vector::VectorTilt { q, angle_deg, .. })
                if angles_are_approx_equal(*angle_deg, 270.0)
                    && matches!(**q, Vector::OneVector { .. }) =>
            {
                (qwerty::PrimitiveBasis::Y, qwerty::Eigenstate::Minus, 0.0)
            }

            _ => todo!("nontrivial superposition {vec}"),
        },

        Vector::PadVector { .. } | Vector::TargetVector { .. } => {
            unreachable!("'?' and '_' atoms should be removed earlier")
        }

        Vector::VectorTilt { .. } => {
            unreachable!("Outer tilt should be removed by ast_vec_to_mlir()")
        }

        // Should be removed by canonicalize()
        Vector::VectorTensor { .. } | Vector::VectorUnit { .. } => {
            unreachable!("should have been removed by Vector::canonicalize()")
        }
    }
}

/// Returns a sequence of qwerty::BasisVectorAttrs for an AST Vector node.
fn ast_vec_to_mlir(vec: &Vector) -> (Vec<qwerty::BasisVectorAttribute<'static>>, f64) {
    let canon_vec = vec.canonicalize();
    let mut vec_attrs = vec![];
    let mut eigenbits = UBig::ZERO;
    let mut prim_basis = None;
    let mut dim = 0;

    let (root_phase, root_vec) = if let Vector::VectorTilt { q, angle_deg, .. } = canon_vec {
        (deg_to_rad(angle_deg), *q)
    } else {
        (0.0, canon_vec)
    };
    let mut phase = root_phase;

    let vecs = root_vec.to_vec();

    for v in &vecs {
        let (v_prim_basis, v_eigenstate, v_phase) = ast_vec_to_mlir_helper(v);
        if let Some(pb) = prim_basis {
            if pb != v_prim_basis {
                // Set has_phase=true only on the last BasisVectorAttr if we
                // end up having a nonzero phase
                let has_phase = false;
                vec_attrs.push(qwerty::BasisVectorAttribute::new(
                    &MLIR_CTX, pb, eigenbits, dim, has_phase,
                ));
                eigenbits = UBig::ZERO;
                prim_basis = Some(v_prim_basis);
                dim = 0;
            }
        } else {
            prim_basis = Some(v_prim_basis);
        }

        eigenbits <<= 1usize;
        if v_eigenstate == qwerty::Eigenstate::Minus {
            eigenbits |= 1usize;
        }
        phase += v_phase;
        // For now, the dimension of everything understood by the helper is 1.
        dim += 1;
    }

    if let Some(pb) = prim_basis {
        let has_phase = !angle_is_approx_zero(phase);
        vec_attrs.push(qwerty::BasisVectorAttribute::new(
            &MLIR_CTX, pb, eigenbits, dim, has_phase,
        ));
    }

    (vec_attrs, phase)
}

/// Holds the ingredients to generate some basis-oriented MLIR op. The indices
/// are useful for having e.g. pad indices bypass a qbtrans op.
struct MlirBasis {
    basis_attr: qwerty::BasisAttribute<'static>,
    phases: Vec<f64>,
    explicit_indices: Vec<usize>,
    pad_indices: Vec<usize>,
    tgt_indices: Vec<usize>,
}

/// Determines if a basis literal containing vectors `vecs` represents the Bell
/// basis. This is a hard-coded hack that should be removed in the future.
fn is_bell_basis(vecs: &[Vector]) -> bool {
    if let [
        Vector::UniformVectorSuperpos {
            q1: q11, q2: q12, ..
        },
        Vector::UniformVectorSuperpos {
            q1: q21, q2: q22, ..
        },
        Vector::UniformVectorSuperpos {
            q1: q31, q2: q32, ..
        },
        Vector::UniformVectorSuperpos {
            q1: q41, q2: q42, ..
        },
    ] = vecs
    {
        let q11_is_00 = if let Vector::VectorTensor { qs: vecs11, .. } = &**q11 {
            matches!(
                &vecs11[..],
                [Vector::ZeroVector { .. }, Vector::ZeroVector { .. }]
            )
        } else {
            false
        };
        let q12_is_11 = if let Vector::VectorTensor { qs: vecs12, .. } = &**q12 {
            matches!(
                &vecs12[..],
                [Vector::OneVector { .. }, Vector::OneVector { .. }]
            )
        } else {
            false
        };

        let q21_is_neg_11 = if let Vector::VectorTilt {
            q: q21q,
            angle_deg: q21_angle_deg,
            ..
        } = &**q21
        {
            if angles_are_approx_equal(*q21_angle_deg, 180.0) {
                if let Vector::VectorTensor { qs: vecs22, .. } = &**q21q {
                    matches!(
                        &vecs22[..],
                        [Vector::OneVector { .. }, Vector::OneVector { .. }]
                    )
                } else {
                    false
                }
            } else {
                false
            }
        } else {
            false
        };
        let q22_is_00 = if let Vector::VectorTensor { qs: vecs21, .. } = &**q22 {
            matches!(
                &vecs21[..],
                [Vector::ZeroVector { .. }, Vector::ZeroVector { .. }]
            )
        } else {
            false
        };

        let q31_is_01 = if let Vector::VectorTensor { qs: vecs32, .. } = &**q31 {
            matches!(
                &vecs32[..],
                [Vector::ZeroVector { .. }, Vector::OneVector { .. }]
            )
        } else {
            false
        };
        let q32_is_10 = if let Vector::VectorTensor { qs: vecs31, .. } = &**q32 {
            matches!(
                &vecs31[..],
                [Vector::OneVector { .. }, Vector::ZeroVector { .. }]
            )
        } else {
            false
        };

        let q41_is_neg_10 = if let Vector::VectorTilt {
            q: q41q,
            angle_deg: q41_angle_deg,
            ..
        } = &**q41
        {
            if angles_are_approx_equal(*q41_angle_deg, 180.0) {
                if let Vector::VectorTensor { qs: vecs42, .. } = &**q41q {
                    matches!(
                        &vecs42[..],
                        [Vector::OneVector { .. }, Vector::ZeroVector { .. }]
                    )
                } else {
                    false
                }
            } else {
                false
            }
        } else {
            false
        };
        let q42_is_01 = if let Vector::VectorTensor { qs: vecs41, .. } = &**q42 {
            matches!(
                &vecs41[..],
                [Vector::ZeroVector { .. }, Vector::OneVector { .. }]
            )
        } else {
            false
        };

        q11_is_00
            && q12_is_11
            && q21_is_neg_11
            && q22_is_00
            && q31_is_01
            && q32_is_10
            && q41_is_neg_10
            && q42_is_01
    } else {
        false
    }
}

/// If the Vec of Bases is a vector of only basis literals {0, 1},
/// or only {p, m}, or only {i, j}, then we want to find this!
fn try_basis_as_primitive(basis_elems: &Vec<Basis>) -> Option<qwerty::PrimitiveBasis> {
    if basis_elems.is_empty() {
        return None;
    }

    // Define the canonical primitive bases
    let std = (
        Vector::ZeroVector { dbg: None },
        Vector::OneVector { dbg: None },
    );

    let p = Vector::UniformVectorSuperpos {
        q1: Box::new(Vector::ZeroVector { dbg: None }),
        q2: Box::new(Vector::OneVector { dbg: None }),
        dbg: None,
    };
    let m = Vector::UniformVectorSuperpos {
        q1: Box::new(Vector::ZeroVector { dbg: None }),
        q2: Box::new(Vector::VectorTilt {
            q: Box::new(Vector::OneVector { dbg: None }),
            angle_deg: 180.0,
            dbg: None,
        }),
        dbg: None,
    };
    let pm = (p, m);

    let i = Vector::UniformVectorSuperpos {
        q1: Box::new(Vector::ZeroVector { dbg: None }),
        q2: Box::new(Vector::VectorTilt {
            q: Box::new(Vector::OneVector { dbg: None }),
            angle_deg: 90.0,
            dbg: None,
        }),
        dbg: None,
    };
    let j = Vector::UniformVectorSuperpos {
        q1: Box::new(Vector::ZeroVector { dbg: None }),
        q2: Box::new(Vector::VectorTilt {
            q: Box::new(Vector::OneVector { dbg: None }),
            angle_deg: 270.0,
            dbg: None,
        }),
        dbg: None,
    };
    let ij = (i, j);

    // helper for classifying
    let classify = |v1: &Vector, v2: &Vector| -> Option<qwerty::PrimitiveBasis> {
        if v1.approx_equal(&std.0) && v2.approx_equal(&std.1) {
            Some(qwerty::PrimitiveBasis::Z)
        } else if v1.approx_equal(&pm.0) && v2.approx_equal(&pm.1) {
            Some(qwerty::PrimitiveBasis::X)
        } else if v1.approx_equal(&ij.0) && v2.approx_equal(&ij.1) {
            Some(qwerty::PrimitiveBasis::Y)
        } else {
            None
        }
    };

    let candidate_basis = match &basis_elems[0] {
        Basis::BasisLiteral { vecs, .. } if vecs.len() == 2 => classify(&vecs[0], &vecs[1]),
        _ => None,
    };

    let Some(candidate) = candidate_basis else {
        return None;
    };

    // verify all other elements match the same primitive basis pattern
    for elem in basis_elems.iter().skip(1) {
        match elem {
            Basis::BasisLiteral { vecs, .. } if vecs.len() == 2 => {
                if classify(&vecs[0], &vecs[1]) != Some(candidate) {
                    return None;
                }
            }
            _ => return None,
        }
    }

    Some(candidate)
}

/// Converts a Basis AST node into a qwerty::BasisAttribute and a separate list
/// of phases which correspond one-to-one with any vectors that have
/// hasPhase==true.
fn ast_basis_to_mlir(basis: &Basis) -> MlirBasis {
    let basis_elements = basis.make_explicit().canonicalize().to_vec();

    let prim_basis = try_basis_as_primitive(&basis_elements);
    let (elems, phases) = if let Some(prim_basis) = prim_basis {
        let std =
            qwerty::BuiltinBasisAttribute::new(&MLIR_CTX, prim_basis, basis_elements.len() as u64);
        let basis_elem = qwerty::BasisElemAttribute::from_std(&MLIR_CTX, std);
        let elems = vec![basis_elem];
        let phases = vec![];
        (elems, phases)
    } else {
        let (elems, phases): (Vec<_>, Vec<_>) = basis_elements
            .iter()
            .map(|elem| {
                match elem {
                    Basis::BasisLiteral { vecs, .. } if is_bell_basis(vecs) => {
                        let std = qwerty::BuiltinBasisAttribute::new(&MLIR_CTX, qwerty::PrimitiveBasis::Bell, 2);
                        (qwerty::BasisElemAttribute::from_std(&MLIR_CTX, std), vec![])
                    }

                    Basis::BasisLiteral { vecs, .. } => {
                        let (vec_attrs, phases): (Vec<_>, Vec<_>) = vecs.iter().map(|vec| {
                            let (vec_attrs, phase) = ast_vec_to_mlir(vec);
                            // TODO: Fix this
                            assert_eq!(vec_attrs.len(), 1, "vectors must be nonempty, and mixing primitive bases in a vector is not currently supported");
                            let vec_attr = vec_attrs[0];
                            let phase_opt = if vec_attr.has_phase() {
                                Some(phase)
                            } else {
                                None
                            };
                            (vec_attr, phase_opt)
                        }).unzip();
                        let veclist = qwerty::BasisVectorListAttribute::new(&MLIR_CTX, &vec_attrs);
                        (qwerty::BasisElemAttribute::from_veclist(&MLIR_CTX, veclist), phases)
                    },

                    Basis::ApplyBasisGenerator { basis, generator, .. } => {
                        match generator {
                            BasisGenerator::Revolve {v1, v2, ..} => {
                                let MlirBasis { basis_attr: foo_basis, phases: foo_phases, .. } = ast_basis_to_mlir(&basis);
                                let (mut bv1_vec, bv1_phase) = ast_vec_to_mlir(v1);
                                let bv1 = bv1_vec.pop().expect("In revolve, basis vectors must be a single vector");
                                let (mut bv2_vec, bv2_phase) = ast_vec_to_mlir(v2);
                                let bv2 = bv2_vec.pop().expect("In revolve, basis vectors must be a single vector");
                                let revolve = qwerty::ApplyRevolveGeneratorAttribute::new(&MLIR_CTX, foo_basis, bv1, bv2);

                                // Because we care about phases
                                let mut phase_vec = foo_phases.iter().map(|phase| Some(*phase)).collect::<Vec<Option<f64>>>();
                                if !angle_is_approx_zero(bv1_phase) { phase_vec.push(Some(bv1_phase)) }
                                if !angle_is_approx_zero(bv2_phase) { phase_vec.push(Some(bv2_phase)) }
                                (qwerty::BasisElemAttribute::from_revolve(&MLIR_CTX, revolve), phase_vec)
                            },
                        }
                    }

                    Basis::EmptyBasisLiteral { .. } | Basis::BasisTensor { .. } => unreachable!("EmptyBasisLiteral and BasisTensor should have been canonicalized away"),
                }
            })
            .unzip();
        (elems, phases)
    };

    let basis_attr = qwerty::BasisAttribute::new(&MLIR_CTX, &elems);
    let phases = phases.iter().flatten().filter_map(|opt| *opt).collect();

    // What follows is a simple linear-time algorithm to find the explicit and
    // implicit indices.

    let basis_dim = basis
        .get_dim()
        .expect("basis to have a well-defined dimension");
    let pad_indices = basis
        .get_atom_indices(VectorAtomKind::PadAtom)
        .expect("basis to have matching pad atoms");
    let tgt_indices = basis
        .get_atom_indices(VectorAtomKind::TargetAtom)
        .expect("basis to have matching target atoms");

    let mut pad_index_queue = pad_indices.to_vec();
    pad_index_queue.reverse();
    let mut tgt_index_queue = tgt_indices.to_vec();
    tgt_index_queue.reverse();

    let mut explicit_indices = vec![];
    for i in 0..basis_dim {
        if pad_index_queue.pop_if(|j| i == *j).is_none()
            && tgt_index_queue.pop_if(|j| i == *j).is_none()
        {
            explicit_indices.push(i);
        }
    }

    MlirBasis {
        basis_attr,
        phases,
        explicit_indices,
        pad_indices,
        tgt_indices,
    }
}

/// Converts a QLit to an mlir::Value. Returns None to handle the edge case []
/// (QubitUnit). That is, None does not indicate an error.
fn ast_qlit_to_mlir(qlit: &QLit, block: &Block<'static>) -> Option<Value<'static, 'static>> {
    let canon_qlit = qlit.canonicalize();

    match &canon_qlit {
        QLit::QubitUnit { .. } => None,

        QLit::ZeroQubit { dbg } => {
            let loc = dbg_to_loc(dbg.clone());
            let dim = IntegerAttribute::new(IntegerType::new(&MLIR_CTX, 64).into(), 1);
            Some(
                block
                    .append_operation(qwerty::qbprep(
                        &MLIR_CTX,
                        qwerty::PrimitiveBasis::Z,
                        qwerty::Eigenstate::Plus,
                        dim,
                        loc,
                    ))
                    .result(0)
                    .unwrap()
                    .into(),
            )
        }

        QLit::OneQubit { dbg } => {
            let loc = dbg_to_loc(dbg.clone());
            let dim = IntegerAttribute::new(IntegerType::new(&MLIR_CTX, 64).into(), 1);
            Some(
                block
                    .append_operation(qwerty::qbprep(
                        &MLIR_CTX,
                        qwerty::PrimitiveBasis::Z,
                        qwerty::Eigenstate::Minus,
                        dim,
                        loc,
                    ))
                    .result(0)
                    .unwrap()
                    .into(),
            )
        }

        QLit::QubitTilt { q, angle_deg, dbg } => ast_qlit_to_mlir(&**q, block).map(|q_val| {
            let loc = dbg_to_loc(dbg.clone());
            let theta_val = mlir_f64_const(deg_to_rad(*angle_deg), loc, block);
            block
                .append_operation(qwerty::qbphase(theta_val, q_val, loc))
                .result(0)
                .unwrap()
                .into()
        }),

        QLit::QubitTensor { qs, dbg } => {
            let loc = dbg_to_loc(dbg.clone());
            let qubits: Vec<_> = qs
                .iter()
                .filter_map(|q| ast_qlit_to_mlir(q, block))
                .flat_map(|q| {
                    block
                        .append_operation(qwerty::qbunpack(q, loc))
                        .results()
                        .map(OperationResult::into)
                        .collect::<Vec<_>>()
                })
                .collect();
            if qubits.is_empty() {
                None
            } else {
                Some(
                    block
                        .append_operation(qwerty::qbpack(&qubits, loc))
                        .result(0)
                        .unwrap()
                        .into(),
                )
            }
        }

        QLit::UniformSuperpos { q1, q2, dbg } => {
            let loc = dbg_to_loc(dbg.clone());
            let (vecs1, phase1) = ast_vec_to_mlir(&q1.convert_to_basis_vector());
            let (vecs2, phase2) = ast_vec_to_mlir(&q2.convert_to_basis_vector());
            if vecs1.is_empty() || vecs2.is_empty() {
                None
            } else {
                let uniform_prob = FloatAttribute::new(&MLIR_CTX, Type::float64(&MLIR_CTX), 0.5);
                let elem1 = qwerty::SuperposElemAttribute::new(
                    &MLIR_CTX,
                    uniform_prob,
                    FloatAttribute::new(&MLIR_CTX, Type::float64(&MLIR_CTX), phase1),
                    &vecs1,
                );
                let elem2 = qwerty::SuperposElemAttribute::new(
                    &MLIR_CTX,
                    uniform_prob,
                    FloatAttribute::new(&MLIR_CTX, Type::float64(&MLIR_CTX), phase2),
                    &vecs2,
                );
                let sup = qwerty::SuperposAttribute::new(&MLIR_CTX, &[elem1, elem2]);
                Some(
                    block
                        .append_operation(qwerty::superpos(&MLIR_CTX, sup, loc))
                        .result(0)
                        .unwrap()
                        .into(),
                )
            }
        }
    }
}

/// Perform a tensor product of many MLIR function values `funcs` with
/// respective AST types `func_tys`. The result of the tensor product (per
/// AST typechecking) has the type `in_ty -> out_ty` (if is_rev==false`) or
/// `in_ty rev-> out_ty` (if `is_rev==true`).
fn synth_function_tensor_product(
    in_ty: &ast::Type,
    out_ty: &ast::Type,
    is_rev: bool,
    funcs: &[Value<'static, 'static>],
    func_tys: &[ast::Type],
    block: &Block<'static>,
    loc: Location<'static>,
) -> Value<'static, 'static> {
    let arg_tys = ast_ty_to_mlir_tys::<qpu::Expr>(in_ty);
    assert!(arg_tys.len() <= 1);
    let res_tys = ast_ty_to_mlir_tys::<qpu::Expr>(out_ty);
    assert!(res_tys.len() <= 1);

    mlir_wrap_lambda(
        &funcs,
        &arg_tys,
        &res_tys,
        is_rev,
        loc,
        block,
        |lambda_block| {
            assert_eq!(lambda_block.argument_count(), funcs.len() + arg_tys.len());
            let unpacked = match in_ty {
                ast::Type::RegType {
                    elem_ty: RegKind::Bit,
                    dim,
                } if *dim > 0 => {
                    let bitbundle_in = lambda_block.argument(funcs.len()).unwrap().into();
                    lambda_block
                        .append_operation(qwerty::bitunpack(bitbundle_in, loc))
                        .results()
                        .map(OperationResult::into)
                        .collect::<Vec<_>>()
                }
                ast::Type::RegType {
                    elem_ty: RegKind::Qubit,
                    dim,
                } if *dim > 0 => {
                    let qbundle_in = lambda_block.argument(funcs.len()).unwrap().into();
                    lambda_block
                        .append_operation(qwerty::qbunpack(qbundle_in, loc))
                        .results()
                        .map(OperationResult::into)
                        .collect::<Vec<_>>()
                }
                ast::Type::RegType {
                    elem_ty: RegKind::Basis,
                    ..
                } => panic!("cannot take tensor product of function returning bases"),
                ast::Type::UnitType | ast::Type::RegType { .. } => vec![], // dim == 0
                // TODO: tensor args & results elementwise
                ast::Type::TupleType { .. } => {
                    panic!("cannot take tensor product of function taking tuples")
                }
                ast::Type::FuncType { .. } | ast::Type::RevFuncType { .. } => {
                    panic!("cannot take tensor product of function taking functions")
                }
            };
            let mut unpacked_iter = unpacked.into_iter();
            let after_func_unpacked: Vec<_> = (0..funcs.len())
                .map(|idx| lambda_block.argument(idx).unwrap().into())
                .zip(func_tys.iter().filter_map(|func_ty| match func_ty {
                    ast::Type::FuncType { in_ty, .. }
                    | ast::Type::RevFuncType { in_out_ty: in_ty } => Some(match &**in_ty {
                        ast::Type::RegType { dim, .. } => *dim,
                        ast::Type::UnitType => 0,
                        ast::Type::FuncType { .. } | ast::Type::RevFuncType { .. } => {
                            panic!("cannot take tensor product of functions taking functions")
                        }
                        ast::Type::TupleType { .. } => {
                            panic!("cannot take tensor product of functions taking tuples")
                        }
                    }),

                    ast::Type::RegType { .. } => {
                        panic!("cannot tensor product registers and get a function")
                    }
                    ast::Type::TupleType { .. } => {
                        panic!("cannot tensor product tuples and get a function")
                    }
                    ast::Type::UnitType => None,
                }))
                .filter_map(|(func_val, in_dim): (Value<'_, '_>, _)| {
                    let func_inputs = if in_dim == 0 {
                        vec![]
                    } else {
                        let elems: Vec<_> = unpacked_iter.by_ref().take(in_dim).collect();
                        vec![match in_ty {
                            ast::Type::RegType {
                                elem_ty: RegKind::Bit,
                                dim,
                            } if *dim > 0 => lambda_block
                                .append_operation(qwerty::bitpack(&elems, loc))
                                .result(0)
                                .unwrap()
                                .into(),
                            ast::Type::RegType {
                                elem_ty: RegKind::Qubit,
                                dim,
                            } if *dim > 0 => lambda_block
                                .append_operation(qwerty::qbpack(&elems, loc))
                                .result(0)
                                .unwrap()
                                .into(),
                            ast::Type::RegType {
                                elem_ty: RegKind::Basis,
                                ..
                            } => panic!("cannot take tensor product of function taking bases"),
                            ast::Type::UnitType | ast::Type::RegType { .. } => {
                                panic!("input type being unit should mean that in_dim==0")
                            }
                            ast::Type::FuncType { .. } | ast::Type::RevFuncType { .. } => {
                                panic!("cannot take tensor product of function taking functions")
                            }
                            ast::Type::TupleType { .. } => {
                                panic!("cannot take tensor product of function taking tuples")
                            }
                        }]
                    };
                    let calli = lambda_block.append_operation(qwerty::call_indirect(
                        func_val,
                        &func_inputs,
                        loc,
                    ));
                    if calli.result_count() == 0 {
                        None
                    } else {
                        assert_eq!(calli.result_count(), 1);
                        Some(calli.result(0).unwrap().into())
                    }
                })
                .flat_map(|result_bundle: Value<'_, '_>| match out_ty {
                    ast::Type::RegType {
                        elem_ty: RegKind::Bit,
                        dim,
                    } if *dim > 0 => lambda_block
                        .append_operation(qwerty::bitunpack(result_bundle, loc))
                        .results()
                        .map(OperationResult::into)
                        .collect::<Vec<_>>(),

                    ast::Type::RegType {
                        elem_ty: RegKind::Qubit,
                        dim,
                    } if *dim > 0 => lambda_block
                        .append_operation(qwerty::qbunpack(result_bundle, loc))
                        .results()
                        .map(OperationResult::into)
                        .collect::<Vec<_>>(),

                    ast::Type::RegType {
                        elem_ty: RegKind::Basis,
                        ..
                    } => panic!("cannot take tensor product of function returning bases"),

                    ast::Type::UnitType | ast::Type::RegType { .. } => {
                        panic!("output type being unit should mean that this code does not run")
                    }

                    ast::Type::FuncType { .. } | ast::Type::RevFuncType { .. } => {
                        panic!("cannot take tensor product of function taking functions")
                    }

                    ast::Type::TupleType { .. } => {
                        panic!("cannot take tensor product of function taking functions")
                    }
                })
                .collect();

            match out_ty {
                ast::Type::RegType {
                    elem_ty: RegKind::Bit,
                    dim,
                } if *dim > 0 => vec![
                    lambda_block
                        .append_operation(qwerty::bitpack(&after_func_unpacked, loc))
                        .result(0)
                        .unwrap()
                        .into(),
                ],
                ast::Type::RegType {
                    elem_ty: RegKind::Qubit,
                    dim,
                } if *dim > 0 => vec![
                    lambda_block
                        .append_operation(qwerty::qbpack(&after_func_unpacked, loc))
                        .result(0)
                        .unwrap()
                        .into(),
                ],
                ast::Type::RegType {
                    elem_ty: RegKind::Basis,
                    ..
                } => panic!("cannot take tensor product of function returning bases"),
                ast::Type::UnitType | ast::Type::RegType { .. } => vec![],
                ast::Type::FuncType { .. } | ast::Type::RevFuncType { .. } => {
                    panic!("cannot take tensor product of function returning functions")
                }
                ast::Type::TupleType { .. } => {
                    panic!("cannot take tensor product of function returning tuples")
                }
            }
        },
    )
}

/// Converts the list of pairs in [`qpu::Expr::NonUniformSuperpos`] and
/// [`qpu::Expr::Ensemble`] nodes into a list of types for
/// [`qpu::NonUniformSuperpos::calc_type`] or [`qpu::Ensemble::calc_type`] and
/// a [`qwerty::SuperposAttribute`].
fn ast_qpu_qlit_pairs_to_mlir(
    pairs: &[(f64, QLit)],
) -> (
    Vec<(ast::Type, ComputeKind)>,
    qwerty::SuperposAttribute<'static>,
) {
    let (term_tys, elems): (Vec<_>, Vec<_>) = pairs
        .iter()
        .map(|(prob, qlit)| {
            let bv = qlit.convert_to_basis_vector();
            let (vec_attrs, phase) = ast_vec_to_mlir(&bv);

            let prob_attr = FloatAttribute::new(&MLIR_CTX, ir::Type::float64(&MLIR_CTX), *prob);
            let phase_attr = FloatAttribute::new(&MLIR_CTX, ir::Type::float64(&MLIR_CTX), phase);
            let elem_attr =
                qwerty::SuperposElemAttribute::new(&MLIR_CTX, prob_attr, phase_attr, &vec_attrs);
            let (qlit_ty, qlit_compute_kind) =
                qlit.typecheck().expect("QLit to pass type checking");
            ((qlit_ty, qlit_compute_kind), elem_attr)
        })
        .unzip();
    let superpos_attr = qwerty::SuperposAttribute::new(&MLIR_CTX, &elems);
    (term_tys, superpos_attr)
}

/// Converts a `@qpu` AST Expr node to mlir::Values by appending ops to the
/// provided block.
fn ast_qpu_expr_to_mlir(
    expr: &qpu::Expr,
    ctx: &mut Ctx,
    block: &Block<'static>,
) -> (ast::Type, ComputeKind, Vec<Value<'static, 'static>>) {
    match expr {
        qpu::Expr::Variable(var @ Variable { name, dbg }) => {
            let (ty, compute_kind) = var
                .calc_type(&mut ctx.type_env)
                .expect("Variable to pass typechecking");
            let bound_vals = ctx
                .bindings
                .get(name)
                .expect(&format!("Variable {} to be bound", name));

            let mlir_vals = match bound_vals {
                BoundVals::Materialized(vals) => vals.clone(),
                BoundVals::UnmaterializedFunction(func_ty) => {
                    let loc = dbg_to_loc(dbg.clone());
                    // We use root_block in case block is inside e.g. an scf.if.
                    let vals = vec![
                        ctx.root_block
                            .insert_operation(
                                0,
                                qwerty::func_const(
                                    &MLIR_CTX,
                                    FlatSymbolRefAttribute::new(&MLIR_CTX, name),
                                    &[],
                                    *func_ty,
                                    loc,
                                ),
                            )
                            .result(0)
                            .unwrap()
                            .into(),
                    ];
                    ctx.bindings
                        .insert(name.to_string(), BoundVals::Materialized(vals.clone()));
                    vals
                }
            };

            (ty, compute_kind, mlir_vals)
        }

        qpu::Expr::UnitLiteral(unit) => {
            let (ty, compute_kind) = unit.typecheck().expect("Unit literal to pass typechecking");
            (ty, compute_kind, vec![])
        }

        qpu::Expr::Pipe(pipe @ Pipe { lhs, rhs, dbg }) => {
            let loc = dbg_to_loc(dbg.clone());
            let (lhs_ty, lhs_compute_kind, lhs_vals) = ast_qpu_expr_to_mlir(&**lhs, ctx, block);
            let (rhs_ty, rhs_compute_kind, rhs_vals) = ast_qpu_expr_to_mlir(&**rhs, ctx, block);
            assert_eq!(
                rhs_vals.len(),
                1,
                "Every function value should have 1 MLIR value"
            );

            let (ty, compute_kind) = pipe
                .calc_type(&(lhs_ty, lhs_compute_kind), &(rhs_ty, rhs_compute_kind))
                .expect("Pipe to pass typechecking");

            let callee_val = rhs_vals[0];
            let calli_results = block
                .append_operation(qwerty::call_indirect(callee_val, &lhs_vals, loc))
                .results()
                .map(OperationResult::into)
                .collect();
            (ty, compute_kind, calli_results)
        }

        qpu::Expr::Measure(meas @ Measure { basis, dbg }) => {
            let basis_ty = basis
                .typecheck()
                .expect("Measurement basis to pass typechecking");
            let (ty, compute_kind) = meas
                .calc_type(&basis_ty)
                .expect("Measurement to pass typechecking");

            let loc = dbg_to_loc(dbg.clone());
            let dim = match basis_ty {
                ast::Type::RegType {
                    elem_ty: RegKind::Basis,
                    dim,
                } => dim,
                _ => panic!("basis should have basis type"),
            };

            let lambda_in_tys =
                &[qwerty::QBundleType::new(&MLIR_CTX, dim.try_into().unwrap()).into()];
            let lambda_out_tys =
                &[qwerty::BitBundleType::new(&MLIR_CTX, dim.try_into().unwrap()).into()];
            let is_rev = false;
            let lambda = vec![mlir_wrap_lambda(
                &[],
                lambda_in_tys,
                lambda_out_tys,
                is_rev,
                loc,
                block,
                |lambda_block| {
                    assert_eq!(lambda_block.argument_count(), 1);
                    let qbundle_in = lambda_block.argument(0).unwrap().into();

                    let MlirBasis {
                        basis_attr,
                        phases: _,
                        explicit_indices,
                        pad_indices,
                        tgt_indices,
                    } = ast_basis_to_mlir(basis);
                    assert!(pad_indices.is_empty());
                    assert!(tgt_indices.is_empty());
                    assert_eq!(explicit_indices.len(), dim);

                    lambda_block
                        .append_operation(qwerty::qbmeas(&MLIR_CTX, basis_attr, qbundle_in, loc))
                        .results()
                        .map(OperationResult::into)
                        .collect()
                },
            )];
            (ty, compute_kind, lambda)
        }

        qpu::Expr::Discard(discard @ Discard { dbg }) => {
            let (ty, compute_kind) = discard.typecheck().expect("Discard to pass typechecking");

            let loc = dbg_to_loc(dbg.clone());
            let lambda_in_tys = &[qwerty::QBundleType::new(&MLIR_CTX, 1).into()];
            let lambda_out_tys = &[];
            let is_rev = false;
            let lambda = vec![mlir_wrap_lambda(
                &[],
                lambda_in_tys,
                lambda_out_tys,
                is_rev,
                loc,
                block,
                |lambda_block| {
                    assert_eq!(lambda_block.argument_count(), 1);
                    let qbundle_in = lambda_block.argument(0).unwrap().into();
                    lambda_block.append_operation(qwerty::qbdiscard(qbundle_in, loc));
                    vec![]
                },
            )];
            (ty, compute_kind, lambda)
        }

        qpu::Expr::Tensor(tensor @ Tensor { vals, dbg }) => {
            let loc = dbg_to_loc(dbg.clone());

            let (val_results, val_vals_2d): (Vec<_>, Vec<_>) = vals
                .iter()
                .map(|val| {
                    let (ty, compute_kind, vals) = ast_qpu_expr_to_mlir(val, ctx, block);
                    ((ty, compute_kind), vals)
                })
                .unzip();
            let (ty, compute_kind) = tensor
                .calc_type(&val_results)
                .expect("Tensor to pass typechecking");
            let val_vals: Vec<_> = val_vals_2d.into_iter().flatten().collect();
            let val_tys: Vec<_> = val_results
                .into_iter()
                .map(|(val_ty, _val_compute_kind)| val_ty)
                .collect();

            let mlir_vals = match &ty {
                ast::Type::FuncType { in_ty, out_ty } => vec![synth_function_tensor_product(
                    &**in_ty, &**out_ty, false, &val_vals, &val_tys, block, loc,
                )],
                ast::Type::RevFuncType { in_out_ty } => vec![synth_function_tensor_product(
                    &**in_out_ty,
                    &**in_out_ty,
                    true,
                    &val_vals,
                    &val_tys,
                    block,
                    loc,
                )],
                ast::Type::RegType {
                    elem_ty: RegKind::Qubit,
                    dim,
                } => {
                    if *dim == 0 {
                        vec![]
                    } else {
                        let unpacked_vals = val_vals
                            .into_iter()
                            .flat_map(|val| {
                                assert!(val.r#type().is_qwerty_q_bundle());
                                block
                                    .append_operation(qwerty::qbunpack(val, loc))
                                    .results()
                                    .map(OperationResult::into)
                                    .collect::<Vec<_>>()
                            })
                            .collect::<Vec<Value<'static, 'static>>>();
                        block
                            .append_operation(qwerty::qbpack(&unpacked_vals, loc))
                            .results()
                            .map(OperationResult::into)
                            .collect()
                    }
                }
                ast::Type::RegType {
                    elem_ty: RegKind::Bit,
                    dim,
                } => {
                    if *dim == 0 {
                        vec![]
                    } else {
                        let unpacked_vals = val_vals
                            .into_iter()
                            .flat_map(|val| {
                                assert!(val.r#type().is_qwerty_bit_bundle());
                                block
                                    .append_operation(qwerty::bitunpack(val, loc))
                                    .results()
                                    .map(OperationResult::into)
                                    .collect::<Vec<_>>()
                            })
                            .collect::<Vec<Value<'static, 'static>>>();
                        block
                            .append_operation(qwerty::bitpack(&unpacked_vals, loc))
                            .results()
                            .map(OperationResult::into)
                            .collect()
                    }
                }
                ast::Type::RegType {
                    elem_ty: RegKind::Basis,
                    ..
                } => panic!("basis is not an expression"),
                ast::Type::TupleType { .. } => panic!("cannot tensor tuples together"),
                ast::Type::UnitType => vec![],
            };
            (ty, compute_kind, mlir_vals)
        }

        qpu::Expr::BasisTranslation(btrans @ BasisTranslation { bin, bout, dbg }) => {
            let bin_ty = bin.typecheck().expect("Input basis to pass typechecking");
            let bout_ty = bout.typecheck().expect("Output basis to pass typechecking");
            let (ty, compute_kind) = btrans
                .calc_type(&bin_ty, &bout_ty)
                .expect("Basis translation to pass typechecking");

            let loc = dbg_to_loc(dbg.clone());
            let dim = match bin_ty {
                ast::Type::RegType {
                    elem_ty: RegKind::Basis,
                    dim,
                } => dim,
                _ => panic!("input basis should have basis type"),
            };

            let lambda_in_out_tys =
                &[qwerty::QBundleType::new(&MLIR_CTX, dim.try_into().unwrap()).into()];
            let is_rev = true;
            let lambda = vec![mlir_wrap_lambda(
                &[],
                lambda_in_out_tys,
                lambda_in_out_tys,
                is_rev,
                loc,
                block,
                |lambda_block| {
                    assert_eq!(lambda_block.argument_count(), 1);
                    let qbundle_in = lambda_block.argument(0).unwrap().into();

                    let MlirBasis {
                        basis_attr: bin_attr,
                        phases: bin_phases,
                        explicit_indices: bin_explicit_indices,
                        pad_indices: bin_pad_indices,
                        tgt_indices: bin_tgt_indices,
                    } = ast_basis_to_mlir(bin);
                    let MlirBasis {
                        basis_attr: bout_attr,
                        phases: bout_phases,
                        explicit_indices: bout_explicit_indices,
                        pad_indices: bout_pad_indices,
                        tgt_indices: bout_tgt_indices,
                    } = ast_basis_to_mlir(bout);

                    assert_eq!(bin_explicit_indices, bout_explicit_indices);
                    assert_eq!(bin_pad_indices, bout_pad_indices);
                    assert!(bin_tgt_indices.is_empty());
                    assert!(bout_tgt_indices.is_empty());

                    let phase_vals: Vec<_> = bin_phases
                        .into_iter()
                        .chain(bout_phases.into_iter())
                        .map(|phase| mlir_f64_const(phase, loc, lambda_block))
                        .collect();

                    let (bypass_qubits, qbtrans_in) = if bin_explicit_indices.len() == dim {
                        // Easy path: there are no pad atoms '?'
                        (vec![], Some(qbundle_in))
                    } else {
                        // Trickier path: we need to unpack and shuffle qubits
                        let unpacked: Vec<_> = lambda_block
                            .append_operation(qwerty::qbunpack(qbundle_in, loc))
                            .results()
                            .map(OperationResult::into)
                            .collect();
                        let explicit_qubits: Vec<_> =
                            bin_explicit_indices.iter().map(|i| unpacked[*i]).collect();
                        let pad_qubits = bin_pad_indices.iter().map(|i| unpacked[*i]).collect();
                        let explicit_qbundle: Option<Value<'static, 'static>> =
                            if explicit_qubits.is_empty() {
                                None
                            } else {
                                Some(
                                    lambda_block
                                        .append_operation(qwerty::qbpack(&explicit_qubits, loc))
                                        .result(0)
                                        .unwrap()
                                        .into(),
                                )
                            };
                        (pad_qubits, explicit_qbundle)
                    };

                    let qbtrans_out: Option<Value<'static, 'static>> =
                        if let Some(qbundle) = qbtrans_in {
                            Some(
                                lambda_block
                                    .append_operation(qwerty::qbtrans(
                                        &MLIR_CTX,
                                        bin_attr,
                                        bout_attr,
                                        &phase_vals,
                                        qbundle,
                                        loc,
                                    ))
                                    .result(0)
                                    .unwrap()
                                    .into(),
                            )
                        } else {
                            None
                        };

                    let ret = if bypass_qubits.is_empty() {
                        // Easy path: just return the result of the basis translation
                        qbtrans_out.expect(
                            "There should be an explicit qubit if there are no implicit qubits",
                        )
                    } else {
                        let to_pack = if let Some(qbundle) = qbtrans_out {
                            let mut qbtrans_out_queue: Vec<_> = lambda_block
                                .append_operation(qwerty::qbunpack(qbundle, loc))
                                .results()
                                .map(OperationResult::into)
                                .zip(bin_explicit_indices.into_iter())
                                .collect();
                            qbtrans_out_queue.reverse();
                            let mut bypass_queue: Vec<_> = bypass_qubits
                                .into_iter()
                                .zip(bin_pad_indices.into_iter())
                                .collect();
                            bypass_queue.reverse();

                            let mut repack_ready = vec![];
                            for i in 0..dim {
                                if let Some((qubit, _j)) =
                                    qbtrans_out_queue.pop_if(|(_qubit, j)| i == *j)
                                {
                                    repack_ready.push(qubit);
                                } else if let Some((qubit, _j)) =
                                    bypass_queue.pop_if(|(_qubit, j)| i == *j)
                                {
                                    repack_ready.push(qubit);
                                } else {
                                    unreachable!(
                                        "qubit neither went through basis translation nor bypassed it"
                                    );
                                }
                            }
                            repack_ready
                        } else {
                            bypass_qubits
                        };
                        lambda_block
                            .append_operation(qwerty::qbpack(&to_pack, loc))
                            .result(0)
                            .unwrap()
                            .into()
                    };

                    vec![ret]
                },
            )];
            (ty, compute_kind, lambda)
        }

        qpu::Expr::Predicated(
            predicated @ Predicated {
                then_func,
                else_func,
                pred,
                dbg,
            },
        ) => {
            let loc = dbg_to_loc(dbg.clone());

            let (then_ty, then_compute_kind, then_vals) =
                ast_qpu_expr_to_mlir(then_func, ctx, block);
            let (else_ty, else_compute_kind, else_vals) =
                ast_qpu_expr_to_mlir(else_func, ctx, block);
            let pred_ty = pred
                .typecheck()
                .expect("Predicate basis to pass typechecking");
            let (ty, compute_kind) = predicated
                .calc_type(
                    &(then_ty, then_compute_kind),
                    &(else_ty, else_compute_kind),
                    &pred_ty,
                )
                .expect("Predication to pass typechecking");

            assert_eq!(then_vals.len(), 1);
            assert_eq!(else_vals.len(), 1);
            let then_func_val = then_vals[0];
            let else_func_val = else_vals[0];

            let dim = match pred_ty {
                ast::Type::RegType {
                    elem_ty: RegKind::Basis,
                    dim,
                } => dim,
                _ => panic!("input basis should have basis type"),
            };

            let lambda_captures = &[then_func_val, else_func_val];
            let lambda_in_out_tys =
                &[qwerty::QBundleType::new(&MLIR_CTX, dim.try_into().unwrap()).into()];
            let is_rev = true;
            let lambda = vec![mlir_wrap_lambda(
                lambda_captures,
                lambda_in_out_tys,
                lambda_in_out_tys,
                is_rev,
                loc,
                block,
                |lambda_block| {
                    assert_eq!(lambda_block.argument_count(), 3);
                    let then_func = lambda_block.argument(0).unwrap().into();
                    let else_func = lambda_block.argument(1).unwrap().into();
                    let qbundle_in = lambda_block.argument(2).unwrap().into();

                    let MlirBasis {
                        basis_attr: pred_basis_attr,
                        phases: _,
                        explicit_indices: pred_explicit_indices,
                        pad_indices: pred_pad_indices,
                        tgt_indices: pred_tgt_indices,
                    } = ast_basis_to_mlir(pred);

                    assert!(!pred_explicit_indices.is_empty());

                    let unpacked: Vec<_> = lambda_block
                        .append_operation(qwerty::qbunpack(qbundle_in, loc))
                        .results()
                        .map(OperationResult::into)
                        .collect();
                    let bypass_qubits: Vec<_> =
                        pred_pad_indices.iter().map(|i| unpacked[*i]).collect();
                    let shuffled_qubits: Vec<_> = pred_explicit_indices
                        .iter()
                        .map(|i| unpacked[*i])
                        .chain(pred_tgt_indices.iter().map(|i| unpacked[*i]))
                        .collect();

                    let shuffled_qbundle: Value<'static, 'static> = lambda_block
                        .append_operation(qwerty::qbpack(&shuffled_qubits, loc))
                        .result(0)
                        .unwrap()
                        .into();

                    let pred_then_func = lambda_block
                        .append_operation(qwerty::func_pred(
                            &MLIR_CTX,
                            pred_basis_attr,
                            then_func,
                            loc,
                        ))
                        .result(0)
                        .unwrap()
                        .into();
                    let then_calli = lambda_block
                        .append_operation(qwerty::call_indirect(
                            pred_then_func,
                            &[shuffled_qbundle],
                            loc,
                        ))
                        .result(0)
                        .unwrap()
                        .into();

                    let adj_else_func = lambda_block
                        .append_operation(qwerty::func_adj(else_func, loc))
                        .result(0)
                        .unwrap()
                        .into();
                    let pred_adj_else_func = lambda_block
                        .append_operation(qwerty::func_pred(
                            &MLIR_CTX,
                            pred_basis_attr,
                            adj_else_func,
                            loc,
                        ))
                        .result(0)
                        .unwrap()
                        .into();
                    let else_calli = lambda_block
                        .append_operation(qwerty::call_indirect(
                            pred_adj_else_func,
                            &[then_calli],
                            loc,
                        ))
                        .result(0)
                        .unwrap()
                        .into();

                    let qbunpack_shuffled =
                        lambda_block.append_operation(qwerty::qbunpack(else_calli, loc));
                    let mut unpacked_shuffled_qubits =
                        qbunpack_shuffled.results().map(OperationResult::into);

                    let mut explicit_queue: Vec<_> = unpacked_shuffled_qubits
                        .by_ref()
                        .take(pred_explicit_indices.len())
                        .zip(pred_explicit_indices.into_iter())
                        .collect();
                    let tgt_fwd_qubits: Vec<_> = unpacked_shuffled_qubits.collect();

                    // Still need to call the unadjointed (forward) version of else
                    let tgt_fwd_qbundle = lambda_block
                        .append_operation(qwerty::qbpack(&tgt_fwd_qubits, loc))
                        .result(0)
                        .unwrap()
                        .into();
                    let fwd_qbundle_out = lambda_block
                        .append_operation(qwerty::call_indirect(else_func, &[tgt_fwd_qbundle], loc))
                        .result(0)
                        .unwrap()
                        .into();

                    explicit_queue.reverse();
                    let mut tgt_queue: Vec<_> = lambda_block
                        .append_operation(qwerty::qbunpack(fwd_qbundle_out, loc))
                        .results()
                        .map(OperationResult::into)
                        .zip(pred_tgt_indices.into_iter())
                        .collect();
                    tgt_queue.reverse();
                    let mut bypass_queue: Vec<_> = bypass_qubits
                        .into_iter()
                        .zip(pred_pad_indices.into_iter())
                        .collect();
                    bypass_queue.reverse();

                    let mut repack_ready = vec![];
                    for i in 0..dim {
                        if let Some((qubit, _j)) = explicit_queue.pop_if(|(_qubit, j)| i == *j) {
                            repack_ready.push(qubit);
                        } else if let Some((qubit, _j)) = tgt_queue.pop_if(|(_qubit, j)| i == *j) {
                            repack_ready.push(qubit);
                        } else if let Some((qubit, _j)) = bypass_queue.pop_if(|(_qubit, j)| i == *j)
                        {
                            repack_ready.push(qubit);
                        } else {
                            unreachable!(
                                "qubit was neither part of the explicit basis, a padding, nor a target"
                            );
                        }
                    }

                    let qbundle_out = lambda_block
                        .append_operation(qwerty::qbpack(&repack_ready, loc))
                        .result(0)
                        .unwrap()
                        .into();
                    vec![qbundle_out]
                },
            )];

            (ty, compute_kind, lambda)
        }

        qpu::Expr::Conditional(
            conditional @ Conditional {
                then_expr,
                else_expr,
                cond,
                dbg,
            },
        ) => {
            let loc = dbg_to_loc(dbg.clone());

            let (cond_ty, cond_compute_kind, cond_vals) = ast_qpu_expr_to_mlir(cond, ctx, block);
            assert_eq!(cond_vals.len(), 1);
            let cond_bitbundle = cond_vals[0];

            let cond_i1 = block
                .append_operation(qwerty::bitunpack(cond_bitbundle, loc))
                .result(0)
                .unwrap()
                .into();

            let then_block_args = &[];
            let then_block = Block::new(then_block_args);

            let mut conditional_ctx = conditional.linearity_check_before_then(&ctx.type_env);
            let (then_ty, then_compute_kind, then_vals) =
                ast_qpu_expr_to_mlir(then_expr, ctx, &then_block);
            then_block.append_operation(scf::r#yield(&then_vals, loc));
            conditional
                .linearity_check_after_then_before_else(&mut ctx.type_env, &mut conditional_ctx);

            let then_region = Region::new();
            then_region.append_block(then_block);

            let else_block_args = &[];
            let else_block = Block::new(else_block_args);
            let (else_ty, else_compute_kind, else_vals) =
                ast_qpu_expr_to_mlir(else_expr, ctx, &else_block);
            else_block.append_operation(scf::r#yield(&else_vals, loc));
            conditional
                .linearity_check_after_else(&ctx.type_env, &conditional_ctx)
                .expect("conditionals to have no linearity issues");

            let else_region = Region::new();
            else_region.append_block(else_block);

            let (ty, compute_kind) = conditional
                .calc_type(
                    &(then_ty, then_compute_kind),
                    &(else_ty, else_compute_kind),
                    &(cond_ty, cond_compute_kind),
                )
                .expect("Conditional to pass typechecking");
            let result_mlir_tys = ast_ty_to_mlir_tys::<qpu::Expr>(&ty);

            let mlir_vals: Vec<_> = block
                .append_operation(scf::r#if(
                    cond_i1,
                    &result_mlir_tys,
                    then_region,
                    else_region,
                    loc,
                ))
                .results()
                .map(OperationResult::into)
                .collect();

            (ty, compute_kind, mlir_vals)
        }

        qpu::Expr::QLit(qlit) => {
            let (ty, compute_kind) = qlit
                .typecheck()
                .expect("Qubit literal to pass type checking");
            let vals = ast_qlit_to_mlir(qlit, block).into_iter().collect();
            (ty, compute_kind, vals)
        }

        qpu::Expr::BitLiteral(blit @ BitLiteral { val, n_bits, dbg }) => {
            let loc = dbg_to_loc(dbg.clone());
            let mlir_vals = vec![mlir_wrap_calc(block, loc, |calc_block| {
                let bit_vals: Vec<_> = (0..*n_bits)
                    .rev()
                    .map(|idx| {
                        let bit = val.bit(idx) as i64;
                        calc_block
                            .append_operation(arith::constant(
                                &MLIR_CTX,
                                IntegerAttribute::new(IntegerType::new(&MLIR_CTX, 1).into(), bit)
                                    .into(),
                                loc,
                            ))
                            .result(0)
                            .unwrap()
                            .into()
                    })
                    .collect();

                calc_block
                    .append_operation(qwerty::bitpack(&bit_vals, loc))
                    .result(0)
                    .unwrap()
                    .into()
            })];

            let (ty, compute_kind) = blit.calc_type().expect("bit literal to pass typechecking");
            (ty, compute_kind, mlir_vals)
        }

        qpu::Expr::Adjoint(adj @ Adjoint { func, dbg }) => {
            let loc = dbg_to_loc(dbg.clone());

            let (func_ty, func_compute_kind, func_vals) = ast_qpu_expr_to_mlir(&**func, ctx, block);
            assert_eq!(
                func_vals.len(),
                1,
                "Every function value should have 1 MLIR value"
            );
            let func_val = func_vals[0];

            let (ty, compute_kind) = adj
                .calc_type(&(func_ty, func_compute_kind))
                .expect("Adjoint to pass typechecking");

            let adj_vals = vec![
                block
                    .append_operation(qwerty::func_adj(func_val, loc))
                    .result(0)
                    .unwrap()
                    .into(),
            ];
            (ty, compute_kind, adj_vals)
        }

        qpu::Expr::NonUniformSuperpos(superpos @ NonUniformSuperpos { pairs, dbg }) => {
            let loc = dbg_to_loc(dbg.clone());

            let (term_tys, superpos_attr) = ast_qpu_qlit_pairs_to_mlir(pairs);
            let (ty, compute_kind) = superpos
                .calc_type(&term_tys)
                .expect("NonUniformSuperpos to pass type checking");

            let superpos_vals = vec![
                block
                    .append_operation(qwerty::superpos(&MLIR_CTX, superpos_attr, loc))
                    .result(0)
                    .unwrap()
                    .into(),
            ];
            (ty, compute_kind, superpos_vals)
        }

        qpu::Expr::Ensemble(ensemble @ Ensemble { pairs, dbg }) => {
            let loc = dbg_to_loc(dbg.clone());

            let (term_tys, superpos_attr) = ast_qpu_qlit_pairs_to_mlir(pairs);
            let (ty, compute_kind) = ensemble
                .calc_type(&term_tys)
                .expect("NonUniformSuperpos to pass type checking");

            let superpos_vals = vec![
                block
                    .append_operation(qwerty::ensemble(&MLIR_CTX, superpos_attr, loc))
                    .result(0)
                    .unwrap()
                    .into(),
            ];
            (ty, compute_kind, superpos_vals)
        }

        qpu::Expr::EmbedClassical(
            embed @ EmbedClassical {
                func_name,
                embed_kind,
                dbg,
            },
        ) => {
            let loc = dbg_to_loc(dbg.clone());
            let (ty, compute_kind) = embed
                .calc_type(&ctx.type_env)
                .expect("EmbedClassical to pass type checking");
            let mlir_tys = ast_ty_to_mlir_tys::<qpu::Expr>(&ty);
            assert_eq!(mlir_tys.len(), 1, "function value must have one MLIR value");
            let mlir_ty = mlir_tys[0]
                .try_into()
                .expect("function value does not have function type");

            let embed_op_func = match embed_kind {
                EmbedKind::Sign => qwerty::embed_sign,
                EmbedKind::Xor => qwerty::embed_xor,
                EmbedKind::InPlace => qwerty::embed_inplace,
            };
            let circuit_sym_attr = FlatSymbolRefAttribute::new(&MLIR_CTX, func_name);
            let embed_vals = vec![
                block
                    .append_operation(embed_op_func(&MLIR_CTX, circuit_sym_attr, mlir_ty, loc))
                    .result(0)
                    .unwrap()
                    .into(),
            ];
            (ty, compute_kind, embed_vals)
        }

        qpu::Expr::QubitRef(_) => panic!("QubitRef should never be lowered to MLIR"),
    }
}

/// Converts an AST `@qpu` `FunctionDef` node into a `qwerty::func` op.
pub fn ast_qpu_func_def_to_mlir(
    func_def: &FunctionDef<qpu::Expr>,
    funcs_available: &FuncsAvailable,
    mlir_func_tys: &[(String, qwerty::FunctionType<'static>)],
) -> (Operation<'static>, qwerty::FunctionType<'static>) {
    let sym_name = StringAttribute::new(&MLIR_CTX, &func_def.name);
    let func_ty = ast_func_mlir_ty(func_def);
    let func_ty_attr = TypeAttribute::new(func_ty);
    let func_loc = dbg_to_loc(func_def.dbg.clone());

    let qwerty_func_ty: qwerty::FunctionType = func_ty.try_into().unwrap();
    let block_args: Vec<_> = qwerty_func_ty
        .get_function_type()
        .inputs()
        .iter()
        .map(|ty| (*ty, func_loc))
        .collect();
    let func_block = Block::new(&block_args);

    let type_env = func_def
        .new_type_env(&funcs_available)
        .expect("valid type env");
    let mut ctx = Ctx::new(&func_block, type_env);

    // Bind function arguments
    assert_eq!(func_def.args.len(), func_block.argument_count());
    for (arg_name, arg_val) in func_def
        .args
        .iter()
        .map(|(_ty, name)| name)
        .zip(func_block.arguments())
    {
        let old_binding = ctx.bindings.insert(
            arg_name.to_string(),
            BoundVals::Materialized(vec![arg_val.into()]),
        );
        assert!(old_binding.is_none());
    }

    // Bind other function names
    for (func_name, mlir_func_ty) in mlir_func_tys {
        let old_binding = ctx.bindings.insert(
            func_name.to_string(),
            BoundVals::UnmaterializedFunction(*mlir_func_ty),
        );
        assert!(old_binding.is_none());
    }

    for stmt in &func_def.body {
        let compute_kind = ast_stmt_to_mlir(
            stmt,
            &mut ctx,
            &func_block,
            func_def.get_expected_ret_type(),
        );
        func_def
            .check_stmt_compute_kind(compute_kind)
            .expect("Statement to have a valid ComputeKind");
    }

    func_def
        .final_linearity_check(&ctx.type_env)
        .expect("No linear typing problems");

    let func_region = Region::new();
    func_region.append_block(func_block);

    let visibility = if func_def.is_classically_callable() {
        Visibility::Public
    } else {
        Visibility::Private
    };

    let func_op = qwerty::func(
        &MLIR_CTX,
        sym_name,
        func_ty_attr,
        visibility,
        func_region,
        func_loc,
    );
    let qwerty_func_ty = func_ty.try_into().unwrap();
    (func_op, qwerty_func_ty)
}
