use crate::{
    ast::{RegKind, qpu::EmbedKind},
    dbg::DebugLoc,
    error::{LowerError, LowerErrorKind, TypeErrorKind},
    meta::{
        DimExpr, MetaFunc, MetaFunctionDef, MetaProgram, MetaType, Progress, classical,
        expand::MacroEnv, qpu,
    },
};
use dashu::integer::IBig;
use std::collections::HashMap;
use std::fmt;

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

#[derive(Debug, Clone)]
pub struct FuncTypeAssignment(Vec<InferType>, InferType);

impl FuncTypeAssignment {
    pub fn new(arg_tys: Vec<InferType>, ret_ty: InferType) -> Self {
        Self(arg_tys, ret_ty)
    }

    fn substitute_type_var(&mut self, type_var_id: usize, new_type: &InferType) {
        for arg_ty in self.0.iter_mut() {
            arg_ty.substitute_type_var(type_var_id, new_type);
        }
        self.1.substitute_type_var(type_var_id, new_type);
    }

    fn strip(self) -> (Vec<Option<MetaType>>, Option<MetaType>) {
        (
            self.0.into_iter().map(InferType::strip).collect(),
            self.1.strip(),
        )
    }

    // TODO: don't duplicate with FunctionDef::get_type() in ast.rs
    fn get_func_type(&self) -> InferType {
        let in_ty = if self.0.is_empty() {
            InferType::UnitType
        } else if self.0.len() == 1 {
            self.0[0].clone()
        } else {
            InferType::TupleType {
                tys: self.0.clone(),
            }
        };

        InferType::FuncType {
            in_ty: Box::new(in_ty),
            out_ty: Box::new(self.1.clone()),
        }
    }

    /// Returns `None` if any arg is not `bit[N]` (or the return value isn't)
    fn get_in_out_dims(&self) -> (Option<DimExpr>, Option<DimExpr>) {
        let get_dim = |ty: &InferType| {
            if let InferType::RegType {
                elem_ty: RegKind::Bit,
                dim,
            } = ty
            {
                Some(dim.clone())
            } else {
                None
            }
        };

        let in_dim = if self.0.is_empty() {
            // TODO: use a useful debug location
            Some(DimExpr::DimConst {
                val: IBig::ZERO,
                dbg: None,
            })
        } else {
            get_dim(&self.0[0]).and_then(|first_dim| {
                self.0.iter().skip(1).try_fold(first_dim, |acc, ty| {
                    let this_dim = get_dim(ty)?;
                    // TODO: use a useful debug location
                    Some(DimExpr::DimSum {
                        left: Box::new(acc),
                        right: Box::new(this_dim),
                        dbg: None,
                    })
                })
            })
        };

        let out_dim = get_dim(&self.1);
        (in_dim, out_dim)
    }
}

pub struct TypeAssignments(Vec<FuncTypeAssignment>);

impl TypeAssignments {
    pub fn new(func_ty_assigns: Vec<FuncTypeAssignment>) -> Self {
        Self(func_ty_assigns)
    }

    pub fn into_iter_stripped(
        self,
    ) -> impl Iterator<Item = (Vec<Option<MetaType>>, Option<MetaType>)> {
        self.0.into_iter().map(FuncTypeAssignment::strip)
    }

    fn substitute_type_var(&mut self, type_var_id: usize, new_type: &InferType) {
        for func_ty_assign in self.0.iter_mut() {
            func_ty_assign.substitute_type_var(type_var_id, new_type);
        }
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

// TODO: Don't duplicate this with ast.rs and meta/type_dim.rs
impl fmt::Display for InferType {
    /// Returns a representation of a type that matches the syntax for the
    /// Python DSL.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            InferType::TypeVar { id } => write!(f, "TypeVar{}", id),
            InferType::FuncType { in_ty, out_ty } => match (&**in_ty, &**out_ty) {
                (
                    InferType::RegType {
                        elem_ty: in_elem_ty,
                        dim: in_dim,
                    },
                    InferType::RegType {
                        elem_ty: out_elem_ty,
                        dim: out_dim,
                    },
                ) if *in_elem_ty != RegKind::Basis && *out_elem_ty != RegKind::Basis => {
                    let prefix = match (in_elem_ty, out_elem_ty) {
                        (RegKind::Qubit, RegKind::Qubit) => "q",
                        (RegKind::Qubit, RegKind::Bit) => "qb",
                        (RegKind::Bit, RegKind::Qubit) => "bq",
                        (RegKind::Bit, RegKind::Bit) => "b",
                        (RegKind::Basis, _) | (_, RegKind::Basis) => {
                            unreachable!("bases cannot be function arguments/results")
                        }
                    };
                    write!(f, "{}func[", prefix)?;
                    if in_elem_ty == out_elem_ty && in_dim == out_dim {
                        write!(f, "{}]", in_dim)
                    } else {
                        write!(f, "{},{}]", in_dim, out_dim)
                    }
                }
                _ => write!(f, "func[{},{}]", in_ty, out_ty),
            },
            InferType::RegType { elem_ty, dim } => match elem_ty {
                RegKind::Qubit => write!(f, "qubit[{}]", dim),
                RegKind::Bit => write!(f, "bit[{}]", dim),
                RegKind::Basis => write!(f, "basis[{}]", dim),
            },
            InferType::TupleType { tys } => {
                write!(f, "(")?;
                for (i, ty) in tys.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", ty)?;
                }
                write!(f, ")")
            }
            InferType::UnitType => write!(f, "None"),
        }
    }
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
    cfuncs: HashMap<String, (Option<DimExpr>, Option<DimExpr>)>,
}

impl TypeEnv {
    fn new(
        funcs_avail: &[(&str, AvailableFuncType)],
        args: &[(InferType, String)],
    ) -> Result<Self, LowerError> {
        let mut ret = Self {
            bindings: HashMap::new(),
            cfuncs: HashMap::new(),
        };

        for (func_name, avail_func_ty) in funcs_avail {
            match avail_func_ty {
                AvailableFuncType::Qpu(func_ty) => {
                    ret.insert(func_name.to_string(), func_ty.clone())?;
                }
                AvailableFuncType::Classical(in_dim, out_dim) => {
                    ret.insert_cfunc(func_name.to_string(), in_dim.clone(), out_dim.clone())?;
                }
            }
        }

        for (arg_ty, arg_name) in args {
            ret.insert(arg_name.to_string(), arg_ty.clone())?;
        }

        Ok(ret)
    }

    fn insert(&mut self, name: String, ty: InferType) -> Result<(), LowerError> {
        let existing_binding = self.bindings.insert(name.to_string(), ty);
        if existing_binding.is_some() {
            Err(LowerError {
                kind: LowerErrorKind::TypeError {
                    kind: TypeErrorKind::RedefinedVariable(name),
                },
                // TODO: set a debug location
                dbg: None,
            })
        } else {
            Ok(())
        }
    }

    fn insert_cfunc(
        &mut self,
        name: String,
        in_dim: Option<DimExpr>,
        out_dim: Option<DimExpr>,
    ) -> Result<(), LowerError> {
        let existing_cfunc = self.cfuncs.insert(name.to_string(), (in_dim, out_dim));
        if existing_cfunc.is_some() {
            Err(LowerError {
                kind: LowerErrorKind::TypeError {
                    // TODO: use more precise error? since this is just functions?
                    kind: TypeErrorKind::RedefinedVariable(name),
                },
                // TODO: set a debug location
                dbg: None,
            })
        } else {
            Ok(())
        }
    }

    fn get_type(&self, name: &str) -> Option<InferType> {
        self.bindings.get(name).cloned()
    }

    fn get_cfunc_dims(&self, name: &str) -> Option<(Option<DimExpr>, Option<DimExpr>)> {
        self.cfuncs.get(name).cloned()
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

pub trait ExprConstrainable {
    fn build_type_constraints(
        &self,
        tv_allocator: &mut TypeVarAllocator,
        env: &mut TypeEnv,
        ty_constraints: &mut TypeConstraints,
        dv_constraints: &mut DimVarConstraints,
    ) -> Result<InferType, LowerError>;
}

impl ExprConstrainable for qpu::MetaExpr {
    fn build_type_constraints(
        &self,
        tv_allocator: &mut TypeVarAllocator,
        env: &mut TypeEnv,
        ty_constraints: &mut TypeConstraints,
        dv_constraints: &mut DimVarConstraints,
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

            qpu::MetaExpr::EmbedClassical {
                func, embed_kind, ..
            } => {
                if let qpu::MetaExpr::Variable { name, dbg } = &**func {
                    if let Some((in_dim, out_dim)) = env.get_cfunc_dims(name) {
                        let qdim = match embed_kind {
                            EmbedKind::Sign => in_dim,
                            EmbedKind::Xor => in_dim.zip(out_dim).map(|(in_dim, out_dim)| {
                                // TODO: use a useful debug location
                                DimExpr::DimSum {
                                    left: Box::new(in_dim),
                                    right: Box::new(out_dim),
                                    dbg: None,
                                }
                            }),
                            EmbedKind::InPlace => {
                                if let Some((in_dim, out_dim)) = in_dim.clone().zip(out_dim.clone())
                                {
                                    dv_constraints.insert(DimVarConstraint::new(in_dim, out_dim));
                                }
                                in_dim.or(out_dim)
                            }
                        };
                        if let Some(dim) = qdim {
                            Ok(InferType::FuncType {
                                in_ty: Box::new(InferType::RegType {
                                    elem_ty: RegKind::Qubit,
                                    dim: dim.clone(),
                                }),
                                out_ty: Box::new(InferType::RegType {
                                    elem_ty: RegKind::Qubit,
                                    dim,
                                }),
                            })
                        } else {
                            Ok(tv_allocator.alloc_type_var())
                        }
                    } else {
                        Err(LowerError {
                            kind: LowerErrorKind::TypeError {
                                // TODO: use more specific error for cfuncs?
                                kind: TypeErrorKind::UndefinedVariable(name.to_string()),
                            },
                            dbg: dbg.clone(),
                        })
                    }
                } else {
                    Ok(tv_allocator.alloc_type_var())
                }
            }

            qpu::MetaExpr::Adjoint { func, .. } => {
                let func_ty =
                    func.build_type_constraints(tv_allocator, env, ty_constraints, dv_constraints)?;
                Ok(func_ty)
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
                let lhs_ty =
                    lhs.build_type_constraints(tv_allocator, env, ty_constraints, dv_constraints)?;
                let rhs_ty =
                    rhs.build_type_constraints(tv_allocator, env, ty_constraints, dv_constraints)?;

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

impl ExprConstrainable for classical::MetaExpr {
    fn build_type_constraints(
        &self,
        tv_allocator: &mut TypeVarAllocator,
        env: &mut TypeEnv,
        ty_constraints: &mut TypeConstraints,
        dv_constraints: &mut DimVarConstraints,
    ) -> Result<InferType, LowerError> {
        match self {
            // TODO: implement this
            classical::MetaExpr::Mod { .. } => Ok(tv_allocator.alloc_type_var()),

            classical::MetaExpr::Variable { name, dbg } => {
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

            classical::MetaExpr::Slice {
                val,
                lower,
                upper,
                dbg,
            } => {
                let _val_ty =
                    val.build_type_constraints(tv_allocator, env, ty_constraints, dv_constraints)?;

                // TODO: add a type constraint that this is bit[N] for a new dv N

                let ty = if let (
                    DimExpr::DimConst { val: lower_val, .. },
                    DimExpr::DimConst { val: upper_val, .. },
                ) = (lower, upper)
                    && upper_val >= lower_val
                {
                    InferType::RegType {
                        elem_ty: RegKind::Bit,
                        dim: DimExpr::DimConst {
                            val: upper_val - lower_val,
                            dbg: dbg.clone(),
                        },
                    }
                } else {
                    tv_allocator.alloc_type_var()
                };
                Ok(ty)
            }

            classical::MetaExpr::UnaryOp { val, .. } => {
                val.build_type_constraints(tv_allocator, env, ty_constraints, dv_constraints)
            }

            classical::MetaExpr::BinaryOp { left, right, .. } => {
                let left_ty =
                    left.build_type_constraints(tv_allocator, env, ty_constraints, dv_constraints)?;
                let right_ty = right.build_type_constraints(
                    tv_allocator,
                    env,
                    ty_constraints,
                    dv_constraints,
                )?;
                ty_constraints.insert(TypeConstraint::new(left_ty.clone(), right_ty));
                Ok(left_ty)
            }

            classical::MetaExpr::ReduceOp { val, .. } => {
                let _val_ty =
                    val.build_type_constraints(tv_allocator, env, ty_constraints, dv_constraints)?;

                // TODO: add a type constraint that this is bit[N] for a new dv N

                // TODO: use a helpful debug location
                let ty = InferType::RegType {
                    elem_ty: RegKind::Bit,
                    dim: DimExpr::DimConst {
                        val: IBig::ONE,
                        dbg: None,
                    },
                };
                Ok(ty)
            }

            classical::MetaExpr::BitLiteral { n_bits, .. } => {
                let ty = InferType::RegType {
                    elem_ty: RegKind::Bit,
                    dim: n_bits.clone(),
                };
                Ok(ty)
            }
        }
    }
}

pub trait StmtConstrainable {
    // TODO: move this elsewhere
    fn get_dbg(&self) -> Option<DebugLoc>;

    fn build_type_constraints(
        &self,
        tv_allocator: &mut TypeVarAllocator,
        env: &mut TypeEnv,
        ty_constraints: &mut TypeConstraints,
        dv_constraints: &mut DimVarConstraints,
    ) -> Result<Option<InferType>, LowerError>;
}

fn build_type_constraints_for_unpack_assign<E: ExprConstrainable>(
    lhs: &[String],
    rhs: &E,
    tv_allocator: &mut TypeVarAllocator,
    env: &mut TypeEnv,
    ty_constraints: &mut TypeConstraints,
    dv_constraints: &mut DimVarConstraints,
) -> Result<Option<InferType>, LowerError> {
    assert!(!lhs.is_empty(), "unpack must have at least left-hand name");

    let rhs_ty = rhs.build_type_constraints(tv_allocator, env, ty_constraints, dv_constraints)?;

    let lhs_tys: Vec<_> = match &rhs_ty {
        InferType::TypeVar { .. } => {
            // TODO: add more helpful constraints in this case
            Ok(std::iter::repeat_with(|| tv_allocator.alloc_type_var())
                .take(lhs.len())
                .collect())
        }

        InferType::RegType { elem_ty, dim } => {
            dv_constraints.insert(DimVarConstraint::new(
                dim.clone(),
                DimExpr::DimConst {
                    val: lhs.len().into(),
                    dbg: None,
                },
            ));
            // TODO: add helpful debug symbol
            Ok(std::iter::repeat(InferType::RegType {
                elem_ty: *elem_ty,
                dim: DimExpr::DimConst {
                    val: IBig::ONE,
                    dbg: None,
                },
            })
            .take(lhs.len())
            .collect())
        }

        InferType::FuncType { .. } | InferType::TupleType { .. } | InferType::UnitType => {
            Err(LowerError {
                kind: LowerErrorKind::TypeError {
                    kind: TypeErrorKind::InvalidType(format!(
                        "Can only unpack from register type, found: {}",
                        &rhs_ty
                    )),
                },
                // TODO: pass a helpful debug symbol
                dbg: None,
            })
        }
    }?;

    for (lhs_name, lhs_ty) in lhs.iter().zip(lhs_tys.into_iter()) {
        env.insert(lhs_name.to_string(), lhs_ty)?;
    }

    Ok(None)
}

impl StmtConstrainable for qpu::MetaStmt {
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
        dv_constraints: &mut DimVarConstraints,
    ) -> Result<Option<InferType>, LowerError> {
        match self {
            qpu::MetaStmt::Return { val, .. } => {
                let val_ty =
                    val.build_type_constraints(tv_allocator, env, ty_constraints, dv_constraints)?;
                Ok(Some(val_ty))
            }

            qpu::MetaStmt::Expr { expr } => {
                let _expr_ty =
                    expr.build_type_constraints(tv_allocator, env, ty_constraints, dv_constraints)?;
                Ok(None)
            }

            qpu::MetaStmt::Assign { lhs, rhs, .. } => {
                let rhs_ty =
                    rhs.build_type_constraints(tv_allocator, env, ty_constraints, dv_constraints)?;
                env.insert(lhs.to_string(), rhs_ty)?;
                Ok(None)
            }

            qpu::MetaStmt::UnpackAssign { lhs, rhs, .. } => {
                build_type_constraints_for_unpack_assign(
                    lhs,
                    rhs,
                    tv_allocator,
                    env,
                    ty_constraints,
                    dv_constraints,
                )
            }

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

impl StmtConstrainable for classical::MetaStmt {
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
        dv_constraints: &mut DimVarConstraints,
    ) -> Result<Option<InferType>, LowerError> {
        match self {
            classical::MetaStmt::Return { val, .. } => {
                let val_ty =
                    val.build_type_constraints(tv_allocator, env, ty_constraints, dv_constraints)?;
                Ok(Some(val_ty))
            }

            classical::MetaStmt::Expr { expr } => {
                let _expr_ty =
                    expr.build_type_constraints(tv_allocator, env, ty_constraints, dv_constraints)?;
                Ok(None)
            }

            classical::MetaStmt::Assign { lhs, rhs, .. } => {
                let rhs_ty =
                    rhs.build_type_constraints(tv_allocator, env, ty_constraints, dv_constraints)?;
                env.insert(lhs.to_string(), rhs_ty)?;
                Ok(None)
            }

            classical::MetaStmt::UnpackAssign { lhs, rhs, .. } => {
                build_type_constraints_for_unpack_assign(
                    lhs,
                    rhs,
                    tv_allocator,
                    env,
                    ty_constraints,
                    dv_constraints,
                )
            }
        }
    }
}

impl<S: StmtConstrainable> MetaFunctionDef<S> {
    fn build_type_constraints(
        &self,
        funcs_available: &[(&str, AvailableFuncType)],
        tv_allocator: &mut TypeVarAllocator,
        ty_constraints: &mut TypeConstraints,
        dv_constraints: &mut DimVarConstraints,
    ) -> Result<FuncTypeAssignment, LowerError> {
        let MetaFunctionDef {
            body,
            args,
            ret_type,
            ..
        } = self;
        let provided_args: Vec<_> = args
            .iter()
            .map(|(arg_ty, arg_name)| {
                (
                    InferType::from_stripped(tv_allocator, arg_ty.clone()),
                    arg_name.to_string(),
                )
            })
            .collect();
        let mut env = TypeEnv::new(&funcs_available, &provided_args)?;

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

            let opt_ret_ty = stmt.build_type_constraints(
                tv_allocator,
                &mut env,
                ty_constraints,
                dv_constraints,
            )?;
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

        let provided_arg_tys = provided_args
            .into_iter()
            .map(|(arg_ty, _arg_name)| arg_ty)
            .collect();
        Ok(FuncTypeAssignment::new(provided_arg_tys, provided_ret_ty))
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

fn stripped_progress(ty: &Option<MetaType>) -> Progress {
    match ty {
        Some(_) => Progress::Full,
        None => Progress::Partial,
    }
}

impl<S> MetaFunctionDef<S> {
    fn inference_progress(&self) -> Progress {
        stripped_progress(&self.ret_type).join(
            self.args
                .iter()
                .map(|(arg_ty, _arg_name)| stripped_progress(arg_ty))
                .fold(Progress::identity(), Progress::join),
        )
    }
}

#[derive(Debug, Clone)]
enum AvailableFuncType {
    Qpu(InferType),
    Classical(Option<DimExpr>, Option<DimExpr>),
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
        funcs_available: &[(&str, AvailableFuncType)],
        tv_allocator: &mut TypeVarAllocator,
        ty_constraints: &mut TypeConstraints,
        dv_constraints: &mut DimVarConstraints,
    ) -> Result<FuncTypeAssignment, LowerError> {
        match self {
            MetaFunc::Qpu(qpu_kernel) => qpu_kernel.build_type_constraints(
                funcs_available,
                tv_allocator,
                ty_constraints,
                dv_constraints,
            ),
            MetaFunc::Classical(classical_func) => classical_func.build_type_constraints(
                funcs_available,
                tv_allocator,
                ty_constraints,
                dv_constraints,
            ),
        }
    }

    fn inference_progress(&self) -> Progress {
        match self {
            MetaFunc::Qpu(qpu_kernel) => qpu_kernel.inference_progress(),
            MetaFunc::Classical(classical_func) => classical_func.inference_progress(),
        }
    }

    // TODO: this is a dumb hack
    fn get_avail_func_type(&self, func_ty_assign: &FuncTypeAssignment) -> AvailableFuncType {
        match self {
            MetaFunc::Qpu(_) => AvailableFuncType::Qpu(func_ty_assign.get_func_type()),
            MetaFunc::Classical(_) => {
                let (in_dim, out_dim) = func_ty_assign.get_in_out_dims();
                AvailableFuncType::Classical(in_dim, out_dim)
            }
        }
    }
}

impl MetaProgram {
    fn build_type_constraints(
        &self,
    ) -> Result<(TypeConstraints, DimVarConstraints, TypeAssignments), LowerError> {
        let MetaProgram { funcs, .. } = self;
        let mut tv_allocator = TypeVarAllocator::new();
        let mut ty_constraints = TypeConstraints::new();
        let mut dv_constraints = DimVarConstraints::new();

        let mut func_ty_assigns = vec![];
        for func in funcs {
            let funcs_available: Vec<_> = funcs
                .iter()
                .zip(&func_ty_assigns)
                .map(|(func, func_ty_assign)| {
                    (func.get_name(), func.get_avail_func_type(func_ty_assign))
                })
                .collect();
            let func_ty = func.build_type_constraints(
                &funcs_available,
                &mut tv_allocator,
                &mut ty_constraints,
                &mut dv_constraints,
            )?;
            func_ty_assigns.push(func_ty);
        }

        let ty_assign = TypeAssignments::new(func_ty_assigns);
        Ok((ty_constraints, dv_constraints, ty_assign))
    }

    fn unify_ty(
        &self,
        ty_constraints: &mut TypeConstraints,
        dv_constraints: &mut DimVarConstraints,
        ty_assign: &mut TypeAssignments,
    ) -> Result<(), LowerError> {
        while let Some(ty_constraint) = ty_constraints.pop() {
            match ty_constraint.as_pair() {
                (InferType::TypeVar { id: left_id }, InferType::TypeVar { id: right_id })
                    if left_id == right_id =>
                {
                    // Already matches
                    Ok(())
                }

                (InferType::TypeVar { id: left_id }, right)
                    if !right.contains_type_var(left_id) =>
                {
                    ty_assign.substitute_type_var(left_id, &right);
                    ty_constraints.substitute_type_var(left_id, &right);
                    Ok(())
                }

                (left, InferType::TypeVar { id: right_id })
                    if !left.contains_type_var(right_id) =>
                {
                    ty_assign.substitute_type_var(right_id, &left);
                    ty_constraints.substitute_type_var(right_id, &left);
                    Ok(())
                }

                (
                    InferType::FuncType {
                        in_ty: left_in_ty,
                        out_ty: left_out_ty,
                    },
                    InferType::FuncType {
                        in_ty: right_in_ty,
                        out_ty: right_out_ty,
                    },
                ) => {
                    ty_constraints.insert(TypeConstraint::new(*left_in_ty, *right_in_ty));
                    ty_constraints.insert(TypeConstraint::new(*left_out_ty, *right_out_ty));
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
                ) if left_elem_ty == right_elem_ty => {
                    dv_constraints.insert(DimVarConstraint::new(left_dim, right_dim));
                    Ok(())
                }

                (
                    InferType::TupleType { tys: left_tys },
                    InferType::TupleType { tys: right_tys },
                ) if left_tys.len() == right_tys.len() => {
                    for (left_ty, right_ty) in left_tys.into_iter().zip(right_tys.into_iter()) {
                        ty_constraints.insert(TypeConstraint::new(left_ty, right_ty));
                    }
                    Ok(())
                }

                (InferType::UnitType, InferType::UnitType) => Ok(()),

                (InferType::UnitType, InferType::RegType { elem_ty: _, dim })
                | (InferType::RegType { elem_ty: _, dim }, InferType::UnitType) => {
                    // Interesting special case: unify Unit and qubit[N] by
                    // adding a dimvar constraint that N=0.
                    // TODO: set a debug location
                    let zero = DimExpr::DimConst {
                        val: IBig::ZERO,
                        dbg: None,
                    };
                    dv_constraints.insert(DimVarConstraint::new(dim, zero));
                    Ok(())
                }

                // TODO: use a more specific pattern here so that when we add
                //       types in the future, we get a Rust compiler error here
                (left, right) => Err(LowerError {
                    kind: LowerErrorKind::TypeError {
                        kind: TypeErrorKind::MismatchedTypes {
                            expected: format!("{}", left),
                            found: format!("{}", right),
                        },
                    },
                    // TODO: pass some location at all, come on man!
                    dbg: None,
                }),
            }?;
        }

        Ok(())
    }

    fn unify_dv(
        &self,
        dv_constraints: &DimVarConstraints,
    ) -> Result<DimVarAssignments, LowerError> {
        Ok(DimVarAssignments::empty(self))
    }

    fn inference_progress(&self) -> Progress {
        self.funcs
            .iter()
            .map(MetaFunc::inference_progress)
            .fold(Progress::identity(), Progress::join)
    }

    fn find_assignments(&self) -> Result<(TypeAssignments, DimVarAssignments), LowerError> {
        let (mut ty_constraints, mut dv_constraints, mut ty_assign) =
            self.build_type_constraints()?;
        self.unify_ty(&mut ty_constraints, &mut dv_constraints, &mut ty_assign)?;
        let dv_assign = self.unify_dv(&dv_constraints)?;
        Ok((ty_assign, dv_assign))
    }

    pub fn infer(&self) -> Result<(MetaProgram, DimVarAssignments, Progress), LowerError> {
        let (ty_assign, dv_assign) = self.find_assignments()?;

        let new_funcs = ty_assign
            .into_iter_stripped()
            .zip(self.funcs.iter())
            .map(|((arg_tys, ret_ty), func)| func.with_tys(&arg_tys, &ret_ty))
            .collect();
        let new_prog = MetaProgram {
            funcs: new_funcs,
            dbg: self.dbg.clone(),
        };
        let progress = new_prog.inference_progress();

        Ok((new_prog, dv_assign, progress))
    }
}
