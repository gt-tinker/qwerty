use crate::{
    ast::RegKind,
    dbg::DebugLoc,
    error::{LowerError, LowerErrorKind, TypeErrorKind},
    meta::{
        DimExpr, MetaFunc, MetaFunctionDef, MetaProgram, MetaType, Progress, classical,
        expand::MacroEnv, qpu,
    },
};
use dashu::integer::IBig;
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone)]
pub struct FuncDimVarAssignments(HashMap<String, IBig>);

impl FuncDimVarAssignments {
    pub fn new() -> Self {
        Self(HashMap::new())
    }

    pub fn get(&self, dim_var: &str) -> Option<IBig> {
        self.0.get(dim_var).cloned()
    }
}

pub struct DimVarAssignments(Vec<FuncDimVarAssignments>);

impl DimVarAssignments {
    pub fn empty(program: &MetaProgram) -> Self {
        Self(
            std::iter::repeat(FuncDimVarAssignments::new())
                .take(program.funcs.len())
                .collect(),
        )
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn iter(&self) -> impl Iterator<Item = &FuncDimVarAssignments> {
        self.0.iter()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TypeConstraint(InferType, InferType);

impl TypeConstraint {
    fn new(left: InferType, right: InferType) -> Self {
        // TODO: sort left and right
        Self(left, right)
    }

    fn as_pair(self) -> (InferType, InferType) {
        (self.0, self.1)
    }

    fn substitute_type_var(&mut self, type_var_id: usize, new_type: &InferType) {
        self.0.substitute_type_var(type_var_id, new_type);
        self.1.substitute_type_var(type_var_id, new_type);
    }
}

#[derive(Debug)]
pub struct TypeConstraints(Vec<TypeConstraint>);

impl TypeConstraints {
    fn new() -> Self {
        Self(vec![])
    }

    fn insert(&mut self, constraint: TypeConstraint) {
        self.0.push(constraint)
    }

    fn pop(&mut self) -> Option<TypeConstraint> {
        self.0.pop()
    }

    fn substitute_type_var(&mut self, type_var_id: usize, new_type: &InferType) {
        for ty_constraint in self.0.iter_mut() {
            ty_constraint.substitute_type_var(type_var_id, new_type);
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct DimVarConstraint(DimExpr, DimExpr);

impl DimVarConstraint {
    pub fn new(left: DimExpr, right: DimExpr) -> Self {
        Self(left, right)
    }
}

pub struct DimVarConstraints(Vec<DimVarConstraint>);

impl DimVarConstraints {
    pub fn new() -> Self {
        Self(vec![])
    }

    fn insert(&mut self, constraint: DimVarConstraint) {
        self.0.push(constraint)
    }
}

pub type FuncTypeAssignment = (Vec<Option<MetaType>>, Option<MetaType>);
pub struct TypeAssignments(Vec<FuncTypeAssignment>);

impl TypeAssignments {
    pub fn empty(program: &MetaProgram) -> Self {
        Self(
            program
                .funcs
                .iter()
                .map(|func| {
                    (
                        std::iter::repeat(None).take(func.arg_count()).collect(),
                        None,
                    )
                })
                .collect(),
        )
    }

    pub fn iter(&self) -> impl Iterator<Item = &FuncTypeAssignment> {
        self.0.iter()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum InferType {
    TypeVar {
        id: usize,
    },
    FuncType {
        in_ty: Box<InferType>,
        out_ty: Box<InferType>,
    },
    RegType {
        elem_ty: RegKind,
        dim: DimExpr,
    },
    TupleType {
        tys: Vec<InferType>,
    },
    UnitType,
}

impl InferType {
    fn from_stripped(tv_allocator: &mut TypeVarAllocator, opt_ty: Option<MetaType>) -> Self {
        if let Some(ty) = opt_ty {
            match ty {
                MetaType::FuncType { in_ty, out_ty } => InferType::FuncType {
                    in_ty: Box::new(Self::from_stripped(tv_allocator, Some(*in_ty))),
                    out_ty: Box::new(Self::from_stripped(tv_allocator, Some(*out_ty))),
                },
                MetaType::RevFuncType { in_out_ty } => InferType::FuncType {
                    in_ty: Box::new(Self::from_stripped(
                        tv_allocator,
                        Some((*in_out_ty).clone()),
                    )),
                    out_ty: Box::new(Self::from_stripped(tv_allocator, Some(*in_out_ty))),
                },
                MetaType::RegType { elem_ty, dim } => InferType::RegType { elem_ty, dim },
                MetaType::TupleType { tys } => InferType::TupleType {
                    tys: tys
                        .into_iter()
                        .map(|ty| InferType::from_stripped(tv_allocator, Some(ty)))
                        .collect(),
                },
                MetaType::UnitType => InferType::UnitType,
            }
        } else {
            tv_allocator.alloc_type_var()
        }
    }

    fn strip(self) -> Option<MetaType> {
        match self {
            InferType::TypeVar { .. } => None,
            InferType::FuncType { in_ty, out_ty } => {
                let meta_in_ty = in_ty.strip()?;
                let meta_out_ty = out_ty.strip()?;
                Some(MetaType::FuncType {
                    in_ty: Box::new(meta_in_ty),
                    out_ty: Box::new(meta_out_ty),
                })
            }
            InferType::RegType { elem_ty, dim } => Some(MetaType::RegType { elem_ty, dim }),
            InferType::TupleType { tys } => {
                let meta_tys = tys
                    .into_iter()
                    .map(|ty| ty.strip())
                    .collect::<Option<Vec<_>>>()?;
                Some(MetaType::TupleType { tys: meta_tys })
            }
            InferType::UnitType => Some(MetaType::UnitType),
        }
    }

    fn expand(self) -> Result<Self, LowerError> {
        match self {
            InferType::FuncType { in_ty, out_ty } => Ok(InferType::FuncType {
                in_ty: Box::new(in_ty.expand()?),
                out_ty: Box::new(out_ty.expand()?),
            }),
            InferType::RegType { elem_ty, dim } => {
                let empty_env = MacroEnv::new();
                let (expanded_dim, _dim_progress) = dim.expand(&empty_env)?;
                Ok(InferType::RegType {
                    elem_ty,
                    dim: expanded_dim,
                })
            }
            InferType::TupleType { tys } => {
                let expanded_tys = tys
                    .into_iter()
                    .map(InferType::expand)
                    .collect::<Result<Vec<_>, LowerError>>()?;
                Ok(InferType::TupleType { tys: expanded_tys })
            }
            InferType::TypeVar { .. } | InferType::UnitType => Ok(self),
        }
    }

    fn substitute_type_var(&mut self, type_var_id: usize, new_type: &InferType) {
        match self {
            InferType::TypeVar { id } if *id == type_var_id => {
                *self = new_type.clone();
            }
            InferType::FuncType { in_ty, out_ty } => {
                in_ty.substitute_type_var(type_var_id, new_type);
                out_ty.substitute_type_var(type_var_id, new_type);
            }
            InferType::TupleType { tys } => {
                for ty in tys.iter_mut() {
                    ty.substitute_type_var(type_var_id, new_type);
                }
            }

            // Nothing to do
            InferType::TypeVar { .. } | InferType::RegType { .. } | InferType::UnitType => (),
        }
    }

    fn contains_type_var(&self, type_var_id: usize) -> bool {
        match self {
            InferType::TypeVar { id } => *id == type_var_id,
            InferType::FuncType { in_ty, out_ty } => {
                in_ty.contains_type_var(type_var_id) && out_ty.contains_type_var(type_var_id)
            }
            InferType::TupleType { tys } => tys.iter().any(|ty| ty.contains_type_var(type_var_id)),

            // Can't contain a type var
            InferType::RegType { .. } | InferType::UnitType => false,
        }
    }
}

pub struct TypeVarAllocator {
    next_type_var_id: usize,
}

impl TypeVarAllocator {
    fn new() -> Self {
        Self {
            next_type_var_id: 0,
        }
    }

    fn alloc_type_var(&mut self) -> InferType {
        let ret = InferType::TypeVar {
            id: self.next_type_var_id,
        };
        self.next_type_var_id += 1;
        ret
    }
}

pub struct TypeEnv {
    bindings: HashMap<String, InferType>,
}

impl TypeEnv {
    fn new() -> Self {
        Self {
            bindings: HashMap::new(),
        }
    }

    fn get_type(&self, name: &str) -> Option<InferType> {
        self.bindings.get(name).cloned()
    }
}

impl qpu::MetaVector {
    /// Returns None if the dimension cannot be determined.
    fn get_dim(&self) -> Option<DimExpr> {
        match self {
            qpu::MetaVector::ZeroVector { dbg }
            | qpu::MetaVector::OneVector { dbg }
            | qpu::MetaVector::PadVector { dbg }
            | qpu::MetaVector::TargetVector { dbg } => Some(DimExpr::DimConst {
                val: IBig::ONE,
                dbg: dbg.clone(),
            }),

            qpu::MetaVector::VectorTilt { q, .. } => q.get_dim(),

            // TODO: should we add a dimvar constraint that these dimensions
            //       match? would this even be useful?
            qpu::MetaVector::UniformVectorSuperpos { q1, .. } => q1.get_dim(),

            qpu::MetaVector::VectorBiTensor { left, right, dbg } => {
                let left_dim = left.get_dim();
                let right_dim = right.get_dim();
                left_dim.zip(right_dim).map(|(ldim, rdim)| DimExpr::DimSum {
                    left: Box::new(ldim),
                    right: Box::new(rdim),
                    dbg: dbg.clone(),
                })
            }

            qpu::MetaVector::VectorUnit { dbg } => Some(DimExpr::DimConst {
                val: IBig::ZERO,
                dbg: dbg.clone(),
            }),

            // These are handled by expansion
            qpu::MetaVector::VectorAlias { .. }
            | qpu::MetaVector::VectorSymbol { .. }
            | qpu::MetaVector::VectorBroadcastTensor { .. } => None,
        }
    }
}

impl qpu::MetaBasisGenerator {
    /// Returns None if the dimension cannot be determined.
    fn get_dim(&self) -> Option<DimExpr> {
        match self {
            qpu::MetaBasisGenerator::BasisGeneratorMacro { .. } => None,
            qpu::MetaBasisGenerator::Revolve { v1, .. } => v1.get_dim(),
        }
    }
}

impl qpu::MetaBasis {
    /// Returns None if the dimension cannot be determined.
    fn get_dim(&self) -> Option<DimExpr> {
        match self {
            qpu::MetaBasis::BasisLiteral { vecs, .. } => {
                assert!(!vecs.is_empty(), "basis literals cannot be empty");
                vecs[0].get_dim()
            }

            qpu::MetaBasis::EmptyBasisLiteral { dbg } => Some(DimExpr::DimConst {
                val: IBig::ZERO,
                dbg: dbg.clone(),
            }),

            qpu::MetaBasis::BasisBiTensor { left, right, dbg } => {
                let left_dim = left.get_dim();
                let right_dim = right.get_dim();
                left_dim.zip(right_dim).map(|(ldim, rdim)| DimExpr::DimSum {
                    left: Box::new(ldim),
                    right: Box::new(rdim),
                    dbg: dbg.clone(),
                })
            }

            qpu::MetaBasis::ApplyBasisGenerator {
                basis,
                generator,
                dbg,
            } => {
                let basis_dim = basis.get_dim();
                let generator_dim = generator.get_dim();
                basis_dim
                    .zip(generator_dim)
                    .map(|(bdim, gdim)| DimExpr::DimSum {
                        left: Box::new(bdim),
                        right: Box::new(gdim),
                        dbg: dbg.clone(),
                    })
            }

            // Fingers crossed expansion will get rid of these
            qpu::MetaBasis::BasisAlias { .. }
            | qpu::MetaBasis::BasisAliasRec { .. }
            | qpu::MetaBasis::BasisBroadcastTensor { .. } => None,
        }
    }
}

impl qpu::MetaExpr {
    fn build_type_constraints(
        &self,
        tv_allocator: &mut TypeVarAllocator,
        env: &mut TypeEnv,
        ty_constraints: &mut TypeConstraints,
    ) -> Result<InferType, LowerError> {
        match self {
            qpu::MetaExpr::Instantiate { .. } => todo!("build_type_constraints() for Instantiate"),

            qpu::MetaExpr::Variable { name, dbg } => {
                if let Some(ty) = env.get_type(name) {
                    Ok(ty.clone())
                } else {
                    Err(LowerError {
                        kind: LowerErrorKind::TypeError {
                            kind: TypeErrorKind::UndefinedVariable(name.to_string()),
                        },
                        dbg: dbg.clone(),
                    })
                }
            }

            qpu::MetaExpr::UnitLiteral { .. } => Ok(InferType::UnitType),

            // TODO: implement these
            qpu::MetaExpr::EmbedClassical { .. } | qpu::MetaExpr::Adjoint { .. } => {
                Ok(tv_allocator.alloc_type_var())
            }

            qpu::MetaExpr::Pipe { lhs, rhs, .. } => {
                // lhs_ty   rhs_ty
                //    vvv   vvv
                //    lhs | rhs
                //   \__________/
                //    result_tv
                //
                // Add this constraint:
                // lhs -> result_tv = rhs
                // And return result_tv as the type.
                let lhs_ty = lhs.build_type_constraints(tv_allocator, env, ty_constraints)?;
                let rhs_ty = rhs.build_type_constraints(tv_allocator, env, ty_constraints)?;

                let result_tv = tv_allocator.alloc_type_var();
                let func_ty = InferType::FuncType {
                    in_ty: Box::new(lhs_ty),
                    out_ty: Box::new(result_tv.clone()),
                };
                ty_constraints.insert(TypeConstraint::new(func_ty, rhs_ty));

                Ok(result_tv)
            }

            qpu::MetaExpr::Measure { basis, .. } => {
                if let Some(basis_dim) = basis.get_dim() {
                    let func_ty = InferType::FuncType {
                        in_ty: Box::new(InferType::RegType {
                            elem_ty: RegKind::Qubit,
                            dim: basis_dim.clone(),
                        }),
                        out_ty: Box::new(InferType::RegType {
                            elem_ty: RegKind::Bit,
                            dim: basis_dim,
                        }),
                    };
                    Ok(func_ty)
                } else {
                    Ok(tv_allocator.alloc_type_var())
                }
            }

            // TODO: implement these
            qpu::MetaExpr::Discard { .. }
            | qpu::MetaExpr::BiTensor { .. }
            | qpu::MetaExpr::BasisTranslation { .. }
            | qpu::MetaExpr::Predicated { .. }
            | qpu::MetaExpr::NonUniformSuperpos { .. }
            | qpu::MetaExpr::Conditional { .. } => Ok(tv_allocator.alloc_type_var()),

            qpu::MetaExpr::QLit { vec } => {
                Ok(if let Some(dim) = vec.get_dim() {
                    InferType::RegType {
                        elem_ty: RegKind::Qubit,
                        dim,
                    }
                } else {
                    // If we can't easily find the number of qubits, just
                    // assign a typevar
                    tv_allocator.alloc_type_var()
                })
            }

            // TODO: implement these
            qpu::MetaExpr::BitLiteral { .. } => Ok(tv_allocator.alloc_type_var()),

            // It's expansion's job to deal with these. Let's just allocate a
            // type var and move on.
            qpu::MetaExpr::ExprMacro { .. }
            | qpu::MetaExpr::BasisMacro { .. }
            | qpu::MetaExpr::BroadcastTensor { .. }
            | qpu::MetaExpr::Repeat { .. } => Ok(tv_allocator.alloc_type_var()),
        }
    }
}

pub trait Constrainable {
    // TODO: move this elsewhere
    fn get_dbg(&self) -> Option<DebugLoc>;

    fn build_type_constraints(
        &self,
        tv_allocator: &mut TypeVarAllocator,
        env: &mut TypeEnv,
        ty_constraints: &mut TypeConstraints,
    ) -> Result<Option<InferType>, LowerError>;
}

impl Constrainable for qpu::MetaStmt {
    fn get_dbg(&self) -> Option<DebugLoc> {
        match self {
            qpu::MetaStmt::ExprMacroDef { dbg, .. }
            | qpu::MetaStmt::BasisMacroDef { dbg, .. }
            | qpu::MetaStmt::BasisGeneratorMacroDef { dbg, .. }
            | qpu::MetaStmt::VectorSymbolDef { dbg, .. }
            | qpu::MetaStmt::BasisAliasDef { dbg, .. }
            | qpu::MetaStmt::BasisAliasRecDef { dbg, .. }
            | qpu::MetaStmt::Assign { dbg, .. }
            | qpu::MetaStmt::UnpackAssign { dbg, .. }
            | qpu::MetaStmt::Return { dbg, .. } => dbg.clone(),

            qpu::MetaStmt::Expr { expr } => expr.get_dbg(),
        }
    }

    fn build_type_constraints(
        &self,
        tv_allocator: &mut TypeVarAllocator,
        env: &mut TypeEnv,
        ty_constraints: &mut TypeConstraints,
    ) -> Result<Option<InferType>, LowerError> {
        match self {
            qpu::MetaStmt::Return { val, .. } => {
                let val_ty = val.build_type_constraints(tv_allocator, env, ty_constraints)?;
                Ok(Some(val_ty))
            }

            // TODO: add constraints for these
            qpu::MetaStmt::Expr { .. }
            | qpu::MetaStmt::Assign { .. }
            | qpu::MetaStmt::UnpackAssign { .. } => Ok(None),

            // It's expansion's job to deal with these. Let's not bother
            qpu::MetaStmt::ExprMacroDef { .. }
            | qpu::MetaStmt::BasisMacroDef { .. }
            | qpu::MetaStmt::BasisGeneratorMacroDef { .. }
            | qpu::MetaStmt::VectorSymbolDef { .. }
            | qpu::MetaStmt::BasisAliasDef { .. }
            | qpu::MetaStmt::BasisAliasRecDef { .. } => Ok(None),
        }
    }
}

impl Constrainable for classical::MetaStmt {
    fn get_dbg(&self) -> Option<DebugLoc> {
        match self {
            classical::MetaStmt::Assign { dbg, .. }
            | classical::MetaStmt::UnpackAssign { dbg, .. }
            | classical::MetaStmt::Return { dbg, .. } => dbg.clone(),

            classical::MetaStmt::Expr { expr } => expr.get_dbg(),
        }
    }

    fn build_type_constraints(
        &self,
        tv_allocator: &mut TypeVarAllocator,
        env: &mut TypeEnv,
        ty_constraints: &mut TypeConstraints,
    ) -> Result<Option<InferType>, LowerError> {
        // TODO: actually add constraints
        Ok(None)
    }
}

impl<S: Constrainable> MetaFunctionDef<S> {
    fn build_type_constraints(
        &self,
        tv_allocator: &mut TypeVarAllocator,
        ty_constraints: &mut TypeConstraints,
    ) -> Result<(), LowerError> {
        let MetaFunctionDef { body, ret_type, .. } = self;
        let mut env = TypeEnv::new();

        let provided_ret_ty = InferType::from_stripped(tv_allocator, ret_type.clone());
        let mut ret_ty_constraint = None;

        for stmt in body {
            if let Some(_) = &ret_ty_constraint {
                return Err(LowerError {
                    kind: LowerErrorKind::TypeError {
                        kind: TypeErrorKind::ReturnNotLastStatement,
                    },
                    dbg: stmt.get_dbg(),
                });
            }

            let opt_ret_ty = stmt.build_type_constraints(tv_allocator, &mut env, ty_constraints)?;
            if let Some(ret_ty) = opt_ret_ty {
                ret_ty_constraint = Some(TypeConstraint::new(provided_ret_ty.clone(), ret_ty));
            }
        }

        if let Some(ret_ty) = ret_ty_constraint {
            ty_constraints.insert(ret_ty);
        } else {
            return Err(LowerError {
                kind: LowerErrorKind::TypeError {
                    kind: TypeErrorKind::ReturnNotLastStatement,
                },
                dbg: self.dbg.clone(),
            });
        }

        Ok(())
    }
}

impl<S: Clone> MetaFunctionDef<S> {
    pub fn with_tys(
        &self,
        arg_tys: &[Option<MetaType>],
        ret_ty: &Option<MetaType>,
    ) -> MetaFunctionDef<S> {
        let MetaFunctionDef {
            name,
            args,
            ret_type,
            body,
            is_rev,
            dim_vars,
            dbg,
        } = self;
        assert_eq!(arg_tys.len(), args.len());

        let new_args = args
            .iter()
            .zip(arg_tys.iter())
            .map(|((old_arg_ty, arg_name), new_arg_ty)| {
                (
                    old_arg_ty.clone().or_else(|| new_arg_ty.clone()),
                    arg_name.to_string(),
                )
            })
            .collect();
        let new_ret_type = ret_type.clone().or_else(|| ret_ty.clone());

        Self {
            name: name.to_string(),
            args: new_args,
            ret_type: new_ret_type,
            body: body.clone(),
            is_rev: *is_rev,
            dim_vars: dim_vars.clone(),
            dbg: dbg.clone(),
        }
    }
}

impl MetaFunc {
    pub fn with_tys(&self, arg_tys: &[Option<MetaType>], ret_ty: &Option<MetaType>) -> MetaFunc {
        match self {
            MetaFunc::Qpu(qpu_kernel) => MetaFunc::Qpu(qpu_kernel.with_tys(arg_tys, ret_ty)),
            MetaFunc::Classical(classical_func) => {
                MetaFunc::Classical(classical_func.with_tys(arg_tys, ret_ty))
            }
        }
    }

    fn build_type_constraints(
        &self,
        tv_allocator: &mut TypeVarAllocator,
        ty_constraints: &mut TypeConstraints,
    ) -> Result<(), LowerError> {
        match self {
            MetaFunc::Qpu(qpu_kernel) => {
                qpu_kernel.build_type_constraints(tv_allocator, ty_constraints)
            }
            MetaFunc::Classical(classical_func) => {
                classical_func.build_type_constraints(tv_allocator, ty_constraints)
            }
        }
    }
}

impl MetaProgram {
    fn build_type_constraints(&self) -> Result<TypeConstraints, LowerError> {
        let MetaProgram { funcs, .. } = self;
        let mut tv_allocator = TypeVarAllocator::new();
        let mut ty_constraints = TypeConstraints::new();

        for func in funcs {
            func.build_type_constraints(&mut tv_allocator, &mut ty_constraints)?;
        }

        Ok(ty_constraints)
    }

    fn unify_ty(
        &self,
        ty_constraints: &mut TypeConstraints,
    ) -> Result<(TypeAssignments, DimVarConstraints, Progress), LowerError> {
        let mut type_assign = TypeAssignments::empty(self);
        let mut dv_constraints = DimVarConstraints::new();

        while let Some(ty_constraint) = ty_constraints.pop() {
            match ty_constraint.as_pair() {
                // Already matches
                (InferType::TypeVar { id: left_id }, InferType::TypeVar { id: right_id })
                    if left_id == right_id =>
                {
                    Ok(())
                }

                (InferType::TypeVar { id: left_id }, right)
                    if !right.contains_type_var(left_id) =>
                {
                    ty_constraints.substitute_type_var(left_id, &right);
                    Ok(())
                }

                (left, InferType::TypeVar { id: right_id })
                    if !left.contains_type_var(right_id) =>
                {
                    ty_constraints.substitute_type_var(right_id, &left);
                    Ok(())
                }

                (
                    InferType::RegType {
                        elem_ty: left_elem_ty,
                        dim: left_dim,
                    },
                    InferType::RegType {
                        elem_ty: right_elem_ty,
                        dim: right_dim,
                    },
                ) => {
                    if left_elem_ty == right_elem_ty {
                        dv_constraints.insert(DimVarConstraint::new(left_dim, right_dim));
                        Ok(())
                    } else {
                        Err(LowerError {
                            kind: LowerErrorKind::TypeError {
                                kind: TypeErrorKind::MismatchedTypes {
                                    expected: format!("{}", left_elem_ty),
                                    found: format!("{}", right_elem_ty),
                                },
                            },
                            // TODO: pass a helpful location
                            dbg: None,
                        })
                    }
                }

                _ => todo!("unify_ty()"),
            }?;
        }
        Ok((type_assign, dv_constraints, Progress::Full))
    }

    fn unify_dv(
        &self,
        dv_constraints: &DimVarConstraints,
    ) -> Result<(DimVarAssignments, Progress), LowerError> {
        Ok((DimVarAssignments::empty(self), Progress::Full))
    }

    pub fn find_assignments(
        &self,
    ) -> Result<(TypeAssignments, DimVarAssignments, Progress), LowerError> {
        let mut ty_constraints = self.build_type_constraints()?;
        let (ty_assign, dv_constraints, ty_progress) = self.unify_ty(&mut ty_constraints)?;
        let (dv_assign, dv_progress) = self.unify_dv(&dv_constraints)?;
        let progress = ty_progress.join(dv_progress);
        Ok((ty_assign, dv_assign, progress))
    }

    pub fn infer(&self) -> Result<(MetaProgram, DimVarAssignments, Progress), LowerError> {
        let (ty_assign, dv_assign, progress) = self.find_assignments()?;

        let new_funcs = ty_assign
            .iter()
            .zip(self.funcs.iter())
            .map(|((arg_tys, ret_ty), func)| func.with_tys(arg_tys, ret_ty))
            .collect();

        Ok((
            MetaProgram {
                funcs: new_funcs,
                dbg: self.dbg.clone(),
            },
            dv_assign,
            progress,
        ))
    }
}
