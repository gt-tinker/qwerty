use dashu::integer::UBig;
use melior::{
    dialect::{arith, qcirc, qwerty, DialectHandle, DialectRegistry},
    execution_engine::SymbolFlags,
    ir::{
        self,
        attribute::{FloatAttribute, IntegerAttribute, StringAttribute, TypeAttribute},
        operation::OperationResult,
        r#type::{FunctionType, IntegerType},
        Block, BlockLike, Location, Module, Operation, OperationLike, Region, RegionLike, Type,
        TypeLike, Value, ValueLike,
    },
    pass::{transform, PassManager},
    utility::register_inliner_extensions,
    Context, Error, ExecutionEngine,
};
use qwerty_ast::{
    ast::{
        self, angle_is_approx_zero, angles_are_approx_equal, Basis, Expr, FunctionDef, Program,
        QLit, RegKind, Stmt, Vector,
    },
    dbg::DebugLoc,
};
use std::{collections::HashMap, sync::LazyLock};

/// Holds the MLIR context in static memory, initializing it on first use.
static MLIR_CTX: LazyLock<Context> = LazyLock::new(|| {
    let ctx = Context::new();
    let registry = DialectRegistry::new();
    let dialects = [
        DialectHandle::arith(),
        DialectHandle::cf(),
        DialectHandle::scf(),
        DialectHandle::func(),
        DialectHandle::math(),
        DialectHandle::llvm(),
        DialectHandle::qcirc(),
        DialectHandle::qwerty(),
    ];

    for dialect in dialects {
        dialect.insert_dialect(&registry);
    }
    register_inliner_extensions(&registry);
    ctx.append_dialect_registry(&registry);

    for dialect in dialects {
        dialect.load_dialect(&ctx);
    }

    ctx
});

/// Converts an AST debug location to an mlir::Location.
fn dbg_to_loc(dbg: Option<DebugLoc>) -> Location<'static> {
    dbg.map_or_else(
        || Location::unknown(&MLIR_CTX),
        |dbg| Location::new(&MLIR_CTX, &dbg.file, dbg.line, dbg.col),
    )
}

/// Converts AST types to mlir::Types. Returns a vec to account for tuples,
/// which will be represented by multiple MLIR values.
fn ast_ty_to_mlir_tys(ty: &ast::Type) -> Vec<ir::Type<'static>> {
    match ty {
        ast::Type::FuncType { in_ty, out_ty } => {
            vec![qwerty::FunctionType::new(
                &MLIR_CTX,
                FunctionType::new(
                    &MLIR_CTX,
                    &ast_ty_to_mlir_tys(&**in_ty),
                    &ast_ty_to_mlir_tys(&**out_ty),
                ),
                /*reversible=*/ false,
            )
            .into()]
        }

        ast::Type::RevFuncType { in_out_ty } => {
            let in_out_mlir_tys = ast_ty_to_mlir_tys(&**in_out_ty);
            vec![qwerty::FunctionType::new(
                &MLIR_CTX,
                FunctionType::new(&MLIR_CTX, &in_out_mlir_tys, &in_out_mlir_tys),
                /*reversible=*/ true,
            )
            .into()]
        }

        ast::Type::RegType {
            elem_ty: RegKind::Bit,
            dim,
        } => {
            vec![qwerty::BitBundleType::new(&MLIR_CTX, *dim).into()]
        }

        ast::Type::RegType {
            elem_ty: RegKind::Qubit,
            dim,
        } => {
            vec![qwerty::QBundleType::new(&MLIR_CTX, *dim).into()]
        }

        ast::Type::RegType {
            elem_ty: RegKind::Basis,
            ..
        } => unreachable!(),

        ast::Type::UnitType => vec![],
        // TODO: Support TupleType once added to qwery_ast::ast::Type (that is
        //       the reason why this returns a Vec)
    }
}

/// Returns the type of a FunctionDef AST node as an mlir::Type.
fn ast_func_mlir_ty(func_def: &FunctionDef) -> ir::Type<'static> {
    let mlir_tys = ast_ty_to_mlir_tys(&func_def.get_type());
    assert_eq!(mlir_tys.len(), 1);
    mlir_tys[0]
}

struct Ctx {
    bindings: HashMap<String, Vec<Value<'static, 'static>>>,
}

impl Ctx {
    fn new() -> Self {
        Self {
            bindings: HashMap::new(),
        }
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
    in_tys: &[ir::Type<'static>],
    out_tys: &[ir::Type<'static>],
    is_rev: bool,
    loc: Location<'static>,
    block: &Block<'static>,
    f: F,
) -> Vec<Value<'static, 'static>>
where
    F: FnOnce(&Block<'static>) -> Vec<Value<'static, 'static>>,
{
    let lambda_ty = qwerty::FunctionType::new(
        &MLIR_CTX,
        FunctionType::new(&MLIR_CTX, in_tys, out_tys),
        is_rev,
    );

    let lambda_block = Block::new(
        &in_tys
            .iter()
            .map(|arg_ty| (*arg_ty, loc))
            .collect::<Vec<_>>(),
    );
    let vals_to_yield = f(&lambda_block);
    lambda_block.append_operation(qwerty::r#return(&vals_to_yield, loc));

    let lambda_region = Region::new();
    lambda_region.append_block(lambda_block);

    let captures = &[];
    block
        .append_operation(qwerty::lambda(captures, lambda_ty, lambda_region, loc))
        .results()
        .map(OperationResult::into)
        .collect()
}

/// Determines the primitive basis, eigenstate, and phase for a basis vector.
/// Will break horribly if not run on a canonicalized vector.
fn ast_vec_to_mlir_helper(vec: &Vector) -> (qwerty::PrimitiveBasis, qwerty::Eigenstate, f64) {
    match vec {
        Vector::ZeroVector { .. } => (qwerty::PrimitiveBasis::Z, qwerty::Eigenstate::Plus, 0.0),

        Vector::OneVector { .. } => (qwerty::PrimitiveBasis::Z, qwerty::Eigenstate::Minus, 0.0),

        Vector::UniformVectorSuperpos { q1, q2, .. } => match (&**q1, &**q2) {
            (Vector::ZeroVector { .. }, Vector::OneVector { .. }) => {
                (qwerty::PrimitiveBasis::X, qwerty::Eigenstate::Plus, 0.0)
            }
            (Vector::ZeroVector { .. }, Vector::VectorTilt { q, angle_deg, .. })
                if angles_are_approx_equal(*angle_deg, 180.0)
                    && matches!(**q, Vector::OneVector { .. }) =>
            {
                (qwerty::PrimitiveBasis::X, qwerty::Eigenstate::Minus, 0.0)
            }
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
            _ => todo!("nontrivial superposition"),
        },

        // TODO: this function should operate on explicit vectors. other
        //       code can handle shuffling for ? and _
        Vector::PadVector { .. } | Vector::TargetVector { .. } => todo!("'?' and '_' lowering"),

        // Should be removed by canonicalize()
        Vector::VectorTilt { .. } | Vector::VectorTensor { .. } | Vector::VectorUnit { .. } => {
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

    let (root_phase, root_vec) = if let Vector::VectorTilt { q, angle_deg, .. } = &canon_vec {
        (deg_to_rad(*angle_deg), &**q)
    } else {
        (0.0, &canon_vec)
    };
    let mut phase = root_phase;

    let vecs = if let Vector::VectorTensor { qs, .. } = root_vec {
        qs.clone()
    } else if let Vector::VectorUnit { .. } = root_vec {
        vec![]
    } else {
        vec![canon_vec]
    };

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

/// Converts a Basis AST node into a qwerty::BasisAttribute and a separate list
/// of phases which correspond one-to-one with any vectors that have
/// hasPhase==true.
fn ast_basis_to_mlir(basis: &Basis) -> (qwerty::BasisAttribute<'static>, Vec<f64>) {
    let basis_elements = basis.canonicalize().to_vec();

    let (elems, phases): (Vec<_>, Vec<_>) = basis_elements
        .iter()
        .map(|elem| {
            match elem {
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

                Basis::EmptyBasisLiteral { .. } | Basis::BasisTensor { .. } => unreachable!("EmptyBasisLiteral and BasisTensor should have been canonicalized away"),
            }
        })
        .unzip();

    let basis_attr = qwerty::BasisAttribute::new(&MLIR_CTX, &elems);
    let phases = phases.iter().flatten().filter_map(|opt| *opt).collect();
    (basis_attr, phases)
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

/// Converts an AST Expr node to mlir::Values by appending ops to the provided
/// block.
fn ast_expr_to_mlir(
    expr: &Expr,
    ctx: &Ctx,
    block: &Block<'static>,
) -> Vec<Value<'static, 'static>> {
    match expr {
        Expr::Pipe { lhs, rhs, dbg } => {
            let loc = dbg_to_loc(dbg.clone());
            let lhs_vals = ast_expr_to_mlir(&**lhs, ctx, block);
            let rhs_vals = ast_expr_to_mlir(&**rhs, ctx, block);
            assert_eq!(rhs_vals.len(), 1);
            let callee_val = rhs_vals[0];
            block
                .append_operation(qwerty::call_indirect(callee_val, &lhs_vals, loc))
                .results()
                .map(OperationResult::into)
                .collect()
        }

        Expr::Measure { basis, dbg } => {
            let loc = dbg_to_loc(dbg.clone());
            let dim: u64 = basis
                .get_dim()
                .expect("type checking should guarantee a valid measurement basis")
                .try_into()
                .unwrap();

            let lambda_in_tys = &[qwerty::QBundleType::new(&MLIR_CTX, dim).into()];
            let lambda_out_tys = &[qwerty::BitBundleType::new(&MLIR_CTX, dim).into()];
            let is_rev = false;
            mlir_wrap_lambda(
                lambda_in_tys,
                lambda_out_tys,
                is_rev,
                loc,
                block,
                |lambda_block| {
                    assert_eq!(lambda_block.argument_count(), 1);
                    let qbundle_in = lambda_block.argument(0).unwrap().into();
                    let (basis_attr, _basis_phases) = ast_basis_to_mlir(basis);
                    lambda_block
                        .append_operation(qwerty::qbmeas(&MLIR_CTX, basis_attr, qbundle_in, loc))
                        .results()
                        .map(OperationResult::into)
                        .collect()
                },
            )
        }

        Expr::Tensor { vals, dbg } => {
            let loc = dbg_to_loc(dbg.clone());
            let unpacked: Vec<_> = vals
                .iter()
                .flat_map(|val| ast_expr_to_mlir(val, ctx, block))
                .flat_map(|val| {
                    if !val.r#type().is_qwerty_q_bundle() {
                        todo!("i can only tensor qubits");
                    }
                    block
                        .append_operation(qwerty::qbunpack(val, loc))
                        .results()
                        .map(OperationResult::into)
                        .collect::<Vec<_>>()
                })
                .collect();

            block
                .append_operation(qwerty::qbpack(&unpacked, loc))
                .results()
                .map(OperationResult::into)
                .collect()
        }

        Expr::BasisTranslation { bin, bout, dbg } => {
            let loc = dbg_to_loc(dbg.clone());
            let dim_in = bin
                .get_dim()
                .expect("type checking should guarantee a valid input basis");
            assert_eq!(
                dim_in,
                bout.get_dim()
                    .expect("type checking should guarantee a valid output basis")
            );
            let dim: u64 = dim_in.try_into().unwrap();

            let lambda_in_out_tys = &[qwerty::QBundleType::new(&MLIR_CTX, dim).into()];
            let is_rev = true;
            mlir_wrap_lambda(
                lambda_in_out_tys,
                lambda_in_out_tys,
                is_rev,
                loc,
                block,
                |lambda_block| {
                    assert_eq!(lambda_block.argument_count(), 1);
                    let qbundle_in = lambda_block.argument(0).unwrap().into();
                    let (bin_attr, bin_phases) = ast_basis_to_mlir(bin);
                    let (bout_attr, bout_phases) = ast_basis_to_mlir(bout);
                    let phase_vals: Vec<_> = bin_phases
                        .into_iter()
                        .chain(bout_phases.into_iter())
                        .map(|phase| mlir_f64_const(phase, loc, lambda_block))
                        .collect();

                    lambda_block
                        .append_operation(qwerty::qbtrans(
                            &MLIR_CTX,
                            bin_attr,
                            bout_attr,
                            &phase_vals,
                            qbundle_in,
                            loc,
                        ))
                        .results()
                        .map(OperationResult::into)
                        .collect()
                },
            )
        }

        Expr::QLit { qlit, .. } => ast_qlit_to_mlir(qlit, block).into_iter().collect(),

        _ => todo!("expression {}", expr),
    }
}

/// Append ops that implement an AST Stmt node to the provided block.
fn ast_stmt_to_mlir(stmt: &Stmt, ctx: &mut Ctx, block: &Block<'static>) {
    match stmt {
        Stmt::Expr { expr, dbg: _ } => {
            ast_expr_to_mlir(expr, ctx, block);
        }

        Stmt::Assign { lhs, rhs, dbg: _ } => {
            let rhs_vals = ast_expr_to_mlir(rhs, ctx, block);
            ctx.bindings.insert(lhs.to_string(), rhs_vals);
        }

        // TODO: for now, unpacking bundles. for tomorrow, unpacking tuples too
        Stmt::UnpackAssign { .. } => todo!("unpack"),

        Stmt::Return { val, dbg } => {
            let vals = ast_expr_to_mlir(val, ctx, block);
            let loc = dbg_to_loc(dbg.clone());
            block.append_operation(qwerty::r#return(&vals, loc));
        }
    }
}

/// Converts an AST FunctionDef node into a qwerty::function op.
fn ast_func_def_to_mlir(func_def: &FunctionDef) -> Operation<'static> {
    let sym_name = StringAttribute::new(&MLIR_CTX, &func_def.name);
    let func_ty = ast_func_mlir_ty(func_def);
    let func_ty_attr = TypeAttribute::new(func_ty);
    let func_attrs = &[];
    let func_loc = dbg_to_loc(func_def.dbg.clone());

    let qwerty_func_ty: qwerty::FunctionType = func_ty.try_into().unwrap();
    let block_args: Vec<_> = qwerty_func_ty
        .get_function_type()
        .inputs()
        .iter()
        .map(|ty| (*ty, func_loc))
        .collect();
    let func_block = Block::new(&block_args);

    let mut ctx = Ctx::new();
    for stmt in &func_def.body {
        ast_stmt_to_mlir(stmt, &mut ctx, &func_block);
    }

    let func_region = Region::new();
    func_region.append_block(func_block);

    qwerty::func(
        &MLIR_CTX,
        sym_name,
        func_ty_attr,
        func_region,
        func_attrs,
        func_loc,
    )
}

/// Converts a Qwerty AST into an mlir::ModuleOp.
fn ast_program_to_mlir(prog: &Program) -> Module {
    let loc = dbg_to_loc(prog.dbg.clone());
    let module = Module::new(loc);
    let module_block = module.body();

    for func_def in &prog.funcs {
        let func_op = ast_func_def_to_mlir(func_def);
        module_block.append_operation(func_op);
    }

    assert!(module.as_operation().verify());

    module
}

struct RunPassesConfig {
    decompose_multi_ctrl: bool,
    to_base_profile: bool,
    dump: bool,
}

fn run_passes(module: &mut Module, cfg: RunPassesConfig) -> Result<(), Error> {
    if cfg.dump {
        eprintln!("Qwerty-dialect IR:");
        module.as_operation().dump();
    }

    {
        let pm = PassManager::new(&MLIR_CTX);

        // Stage 1: Optimize Qwerty dialect

        // Running the canonicalizer may introduce lambdas, so run it once first
        // before the lambda lifter
        pm.add_pass(transform::create_canonicalizer());
        pm.add_pass(qwerty::create_lift_lambdas());
        // Will turn qwerty.call_indirects into qwerty.calls
        pm.add_pass(transform::create_canonicalizer());
        pm.add_pass(transform::create_inliner());
        // It seems the inliner may not run a final round of canonicalization
        // sometimes, so do it ourselves
        pm.add_pass(transform::create_canonicalizer());
        // Remove any leftover symbols
        pm.add_pass(transform::create_symbol_dce());

        // Stage 2: Convert to QCirc dialect

        // -only-pred-ones will introduce some lambdas, so lift and inline them too
        pm.add_pass(qwerty::create_only_pred_ones());
        pm.add_pass(qwerty::create_lift_lambdas());
        // Will turn qwerty.call_indirects into qwerty.calls
        pm.add_pass(transform::create_canonicalizer());
        pm.add_pass(transform::create_inliner());
        pm.add_pass(qwerty::create_qwerty_to_q_circ_conversion());
        // Add canonicalizer pass to prune unused "builtin.unrealized_conversion_cast" ops
        pm.add_pass(transform::create_canonicalizer());
        pm.run(module)?;
    }

    if cfg.dump {
        eprintln!("QCirc-dialect IR:");
        module.as_operation().dump();
    }

    {
        let pm = PassManager::new(&MLIR_CTX);

        // Stage 3: Optimize QCirc dialect

        let func_pm = pm.nested_under("func.func");
        func_pm.add_pass(qcirc::create_peephole_optimization());
        if cfg.decompose_multi_ctrl {
            func_pm.add_pass(qcirc::create_decompose_multi_control());
            func_pm.add_pass(qcirc::create_peephole_optimization());
            func_pm.add_pass(qcirc::create_replace_non_qasm_gates());
        }

        // Stage 4: Convert to QIR
        pm.add_pass(qcirc::create_replace_non_qir_gates());
        if cfg.to_base_profile {
            pm.add_pass(qcirc::create_base_profile_module_prep());
            let func_pm = pm.nested_under("func.func");
            func_pm.add_pass(qcirc::create_base_profile_func_prep());
        }
        pm.add_pass(qcirc::create_q_circ_to_qir_conversion());
        pm.add_pass(transform::create_canonicalizer());

        pm.run(module)?;
    }

    if cfg.dump {
        eprintln!("LLVM-dialect IR:");
        module.as_operation().dump();
    }

    Ok(())
}

pub struct ShotResult {
    pub bits: UBig,
    pub num_bits: usize,
    pub count: usize,
}

macro_rules! qir_symbol {
    ($func_name:ident) => {
        (
            stringify!($func_name),
            qir_backend::$func_name as *mut (),
            SymbolFlags::CALLABLE,
        )
    };
}

pub fn run_ast(prog: &Program, func_name: &str, num_shots: usize) -> Vec<ShotResult> {
    assert_ne!(num_shots, 0);

    let mut module = ast_program_to_mlir(prog);
    let cfg = RunPassesConfig {
        decompose_multi_ctrl: false,
        to_base_profile: false,
        dump: false,
    };
    run_passes(&mut module, cfg).unwrap();

    let exec = ExecutionEngine::new(&module, 3, &[], false);

    unsafe {
        exec.register_symbols(&[
            qir_symbol!(__quantum__rt__initialize),
            qir_symbol!(__quantum__qis__x__body),
            qir_symbol!(__quantum__qis__y__body),
            qir_symbol!(__quantum__qis__z__body),
            qir_symbol!(__quantum__qis__h__body),
            qir_symbol!(__quantum__qis__rx__body),
            qir_symbol!(__quantum__qis__ry__body),
            qir_symbol!(__quantum__qis__rz__body),
            qir_symbol!(__quantum__qis__s__body),
            qir_symbol!(__quantum__qis__s__adj),
            qir_symbol!(__quantum__qis__t__body),
            qir_symbol!(__quantum__qis__t__adj),
            qir_symbol!(__quantum__qis__cx__body),
            qir_symbol!(__quantum__qis__cy__body),
            qir_symbol!(__quantum__qis__cz__body),
            qir_symbol!(__quantum__qis__ccx__body),
            qir_symbol!(__quantum__qis__x__ctl),
            qir_symbol!(__quantum__qis__y__ctl),
            qir_symbol!(__quantum__qis__z__ctl),
            qir_symbol!(__quantum__qis__h__ctl),
            qir_symbol!(__quantum__qis__rx__ctl),
            qir_symbol!(__quantum__qis__ry__ctl),
            qir_symbol!(__quantum__qis__rz__ctl),
            qir_symbol!(__quantum__qis__s__ctl),
            qir_symbol!(__quantum__qis__s__ctladj),
            qir_symbol!(__quantum__qis__t__ctl),
            qir_symbol!(__quantum__qis__t__ctladj),
            qir_symbol!(__quantum__qis__m__body),
            qir_symbol!(__quantum__qis__reset__body),
            qir_symbol!(__quantum__rt__result_get_one),
            qir_symbol!(__quantum__rt__result_equal),
            qir_symbol!(__quantum__rt__qubit_allocate),
            qir_symbol!(__quantum__rt__qubit_release),
            qir_symbol!(__quantum__rt__array_create_1d),
            qir_symbol!(__quantum__rt__array_copy),
            qir_symbol!(__quantum__rt__array_update_reference_count),
            qir_symbol!(__quantum__rt__array_update_alias_count),
            qir_symbol!(__quantum__rt__array_get_element_ptr_1d),
            qir_symbol!(__quantum__rt__array_get_size_1d),
            qir_symbol!(__quantum__rt__tuple_create),
            qir_symbol!(__quantum__rt__tuple_update_reference_count),
            qir_symbol!(__quantum__rt__tuple_update_alias_count),
            qir_symbol!(__quantum__rt__callable_create),
            qir_symbol!(__quantum__rt__callable_copy),
            qir_symbol!(__quantum__rt__callable_invoke),
            qir_symbol!(__quantum__rt__callable_make_adjoint),
            qir_symbol!(__quantum__rt__callable_make_controlled),
            qir_symbol!(__quantum__rt__callable_update_reference_count),
            qir_symbol!(__quantum__rt__callable_update_alias_count),
            qir_symbol!(__quantum__rt__capture_update_reference_count),
            qir_symbol!(__quantum__rt__capture_update_alias_count),
        ]);
        qir_backend::__quantum__rt__initialize(std::ptr::null::<u8>() as *mut _);
    }

    let mut counts: HashMap<UBig, usize> = HashMap::new();
    let mut num_bits_ret = None;

    for _ in 0..num_shots {
        // TODO: check that the function really takes no arguments and returns a bitbundle
        let (bits, num_bits_qir) = unsafe {
            let mut result: *const qir_backend::QirArray = std::ptr::null();
            exec.invoke_packed(func_name, &mut [&raw mut result as *mut ()])
                .unwrap();

            let num_bits = qir_backend::__quantum__rt__array_get_size_1d(result);
            let mut bits_ubig = UBig::ZERO;
            for i in 0..num_bits {
                bits_ubig <<= 1usize;
                if *qir_backend::__quantum__rt__array_get_element_ptr_1d(result, i) != 0 {
                    bits_ubig |= 1usize;
                }
            }
            qir_backend::__quantum__rt__array_update_reference_count(result, -1);
            (bits_ubig, num_bits)
        };

        if let Some(nbits) = num_bits_ret {
            assert_eq!(nbits, num_bits_qir);
        } else {
            num_bits_ret = Some(num_bits_qir);
        }

        counts
            .entry(bits)
            .and_modify(|count| *count += 1)
            .or_insert(1);
    }

    let num_bits: usize = num_bits_ret
        .expect("There should be at least one shot")
        .try_into()
        .unwrap();

    counts
        .iter()
        .map(|(bits, count)| ShotResult {
            bits: bits.clone(),
            num_bits,
            count: *count,
        })
        .collect()
}
