use crate::{
    ast::RegKind,
    dbg::DebugLoc,
    error::{LowerError, LowerErrorKind, TypeErrorKind},
    meta::{DimExpr, MetaFunc, MetaFunctionDef, MetaProgram, MetaType, Progress, classical, qpu},
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
pub struct TypeConstraint(TypeOrTypeVar, TypeOrTypeVar);

impl TypeConstraint {
    fn new(left: TypeOrTypeVar, right: TypeOrTypeVar) -> Self {
        // TODO: sort left and right
        Self(left, right)
    }
}

pub struct TypeConstraints(HashSet<TypeConstraint>);

impl TypeConstraints {
    fn new() -> Self {
        Self(HashSet::new())
    }

    /// Returns `true` if this constraint was not previously in the set of
    /// constraints.
    fn insert(&mut self, constraint: TypeConstraint) -> bool {
        self.0.insert(constraint)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct DimVarConstraint(DimExpr, DimExpr);

pub struct DimVarConstraints(HashSet<DimVarConstraint>);

impl DimVarConstraints {
    pub fn new() -> Self {
        Self(HashSet::new())
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
pub enum TypeOrTypeVar {
    Type(MetaType),
    TypeVar(usize),
}

impl TypeOrTypeVar {
    fn from_stripped(env: &mut TypeEnv, opt_ty: Option<MetaType>) -> Self {
        if let Some(ty) = opt_ty {
            TypeOrTypeVar::Type(ty)
        } else {
            env.alloc_type_var()
        }
    }

    fn strip(&self) -> Option<MetaType> {
        match self {
            TypeOrTypeVar::Type(ty) => Some(ty.clone()),
            TypeOrTypeVar::TypeVar(_) => None,
        }
    }
}

pub struct TypeEnv {
    next_type_var_id: usize,
    bindings: HashMap<String, TypeOrTypeVar>,
}

impl TypeEnv {
    fn new() -> Self {
        Self {
            next_type_var_id: 0,
            bindings: HashMap::new(),
        }
    }

    fn alloc_type_var(&mut self) -> TypeOrTypeVar {
        let ret = TypeOrTypeVar::TypeVar(self.next_type_var_id);
        self.next_type_var_id += 1;
        ret
    }

    fn get_type(&self, name: &str) -> Option<TypeOrTypeVar> {
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

impl qpu::MetaExpr {
    fn build_type_constraints(
        &self,
        env: &mut TypeEnv,
        ty_constraints: &mut TypeConstraints,
    ) -> Result<TypeOrTypeVar, LowerError> {
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

            qpu::MetaExpr::UnitLiteral { .. } => Ok(TypeOrTypeVar::Type(MetaType::UnitType)),

            // TODO: implement these
            qpu::MetaExpr::EmbedClassical { .. } | qpu::MetaExpr::Adjoint { .. } => {
                Ok(env.alloc_type_var())
            }

            qpu::MetaExpr::Pipe { .. } => todo!("pipe"),

            qpu::MetaExpr::Measure { .. } => todo!("measure"),

            // TODO: implement these
            qpu::MetaExpr::Discard { .. }
            | qpu::MetaExpr::BiTensor { .. }
            | qpu::MetaExpr::BasisTranslation { .. }
            | qpu::MetaExpr::Predicated { .. }
            | qpu::MetaExpr::NonUniformSuperpos { .. }
            | qpu::MetaExpr::Conditional { .. } => Ok(env.alloc_type_var()),

            qpu::MetaExpr::QLit { vec } => {
                Ok(if let Some(dim) = vec.get_dim() {
                    TypeOrTypeVar::Type(MetaType::RegType {
                        elem_ty: RegKind::Qubit,
                        dim,
                    })
                } else {
                    // If we can't easily find the number of qubits, just
                    // assign a typevar
                    env.alloc_type_var()
                })
            }

            // TODO: implement these
            qpu::MetaExpr::BitLiteral { .. } => Ok(env.alloc_type_var()),

            // It's expansion's job to deal with these. Let's just allocate a
            // type var and move on.
            qpu::MetaExpr::ExprMacro { .. }
            | qpu::MetaExpr::BasisMacro { .. }
            | qpu::MetaExpr::BroadcastTensor { .. }
            | qpu::MetaExpr::Repeat { .. } => Ok(env.alloc_type_var()),
        }
    }
}

pub trait Constrainable {
    // TODO: move this elsewhere
    fn get_dbg(&self) -> Option<DebugLoc>;

    fn build_type_constraints(
        &self,
        env: &mut TypeEnv,
        ty_constraints: &mut TypeConstraints,
    ) -> Result<Option<TypeOrTypeVar>, LowerError>;
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
        env: &mut TypeEnv,
        ty_constraints: &mut TypeConstraints,
    ) -> Result<Option<TypeOrTypeVar>, LowerError> {
        match self {
            qpu::MetaStmt::Return { val, .. } => {
                let val_ty = val.build_type_constraints(env, ty_constraints)?;
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
        env: &mut TypeEnv,
        ty_constraints: &mut TypeConstraints,
    ) -> Result<Option<TypeOrTypeVar>, LowerError> {
        // TODO: actually add constraints
        Ok(None)
    }
}

impl<S: Constrainable> MetaFunctionDef<S> {
    fn build_type_constraints(
        &self,
        ty_constraints: &mut TypeConstraints,
    ) -> Result<(), LowerError> {
        let MetaFunctionDef { body, ret_type, .. } = self;
        let mut env = TypeEnv::new();

        let provided_ret_ty = TypeOrTypeVar::from_stripped(&mut env, ret_type.clone());
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

            let opt_ret_ty = stmt.build_type_constraints(&mut env, ty_constraints)?;
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
        ty_constraints: &mut TypeConstraints,
    ) -> Result<(), LowerError> {
        match self {
            MetaFunc::Qpu(qpu_kernel) => qpu_kernel.build_type_constraints(ty_constraints),
            MetaFunc::Classical(classical_func) => {
                classical_func.build_type_constraints(ty_constraints)
            }
        }
    }
}

impl MetaProgram {
    fn build_type_constraints(&self) -> Result<TypeConstraints, LowerError> {
        let MetaProgram { funcs, .. } = self;
        let mut ty_constraints = TypeConstraints::new();

        for func in funcs {
            func.build_type_constraints(&mut ty_constraints)?;
        }

        Ok(ty_constraints)
    }

    fn unify_ty(
        &self,
        ty_constraints: &TypeConstraints,
    ) -> Result<(TypeAssignments, DimVarConstraints, Progress), LowerError> {
        Ok((
            TypeAssignments::empty(self),
            DimVarConstraints::new(),
            Progress::Full,
        ))
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
        let ty_constraints = self.build_type_constraints()?;
        let (ty_assign, dv_constraints, ty_progress) = self.unify_ty(&ty_constraints)?;
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
