use crate::{
    error::LowerError,
    meta::{DimExpr, MetaFunc, MetaFunctionDef, MetaProgram, MetaType, Progress},
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

pub struct TypeConstraints(HashSet<TypeConstraint>);

impl TypeConstraints {
    fn new() -> Self {
        Self(HashSet::new())
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
enum TypeOrTypeVar {
    Type(MetaType),
    TypeVar(String),
}

impl TypeOrTypeVar {
    fn strip(&self) -> Option<MetaType> {
        match self {
            TypeOrTypeVar::Type(ty) => Some(ty.clone()),
            TypeOrTypeVar::TypeVar(_) => None,
        }
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
}

impl MetaProgram {
    fn build_type_constraints(&self) -> Result<TypeConstraints, LowerError> {
        // TODO: do something
        Ok(TypeConstraints::new())
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

    //pub fn infer(&self) -> Result<(MetaProgram, Progress), LowerError> {
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

    //pub fn instantiate(&self) -> Result<(MetaProgram, Progress), LowerError> {
    //    // TODO: implement me
    //    Ok((self.clone(), Progress::Full))
    //}

    //pub fn infer_and_instantiate(&self) -> Result<(MetaProgram, Progress), LowerError> {
    //    let (inferred_program, infer_progress) = self.infer()?;
    //    let (instantiated_program, instantiate_progress) = self.instantiate()?;
    //    (instantiaged_program, infer_progress.join(instantiate_progress))
    //}
}
