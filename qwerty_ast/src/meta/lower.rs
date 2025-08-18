use crate::{
    ast,
    dbg::DebugLoc,
    error::{LowerError, LowerErrorKind},
    meta::{
        DimExpr, DimVar, MetaFunc, MetaFunctionDef, MetaProgram, MetaType, classical,
        infer::DimVarAssignments, qpu,
    },
};
use std::collections::HashMap;

/// Allows expansion/inference to report on whether it is completed yet.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Progress {
    /// More rounds of expansion/inference are needed.
    Partial,

    /// Expansion/inference is finished. Extraction will work successfully.
    Full,
}

impl Progress {
    /// Returns the progress `P` such that `P.join(P') == P'` for any `P'`.
    pub fn identity() -> Self {
        Progress::Full
    }

    /// If a parent node contains two sub-nodes who reported progress
    /// `self` and `other`, then returns the progress of the parent node.
    pub fn join(self, other: Progress) -> Progress {
        match (self, other) {
            (Progress::Full, Progress::Full) => Progress::Full,
            (Progress::Partial, _) | (_, Progress::Partial) => Progress::Partial,
        }
    }

    /// Returns true if expansion/inference is finished.
    pub fn is_finished(self) -> bool {
        self == Progress::Full
    }
}

impl MetaType {
    pub fn substitute_dim_var(&self, dim_var: DimVar, new_dim_expr: DimExpr) -> MetaType {
        match self {
            MetaType::FuncType { in_ty, out_ty } => MetaType::FuncType {
                in_ty: Box::new(in_ty.substitute_dim_var(dim_var.clone(), new_dim_expr.clone())),
                out_ty: Box::new(out_ty.substitute_dim_var(dim_var, new_dim_expr)),
            },

            MetaType::RevFuncType { in_out_ty } => MetaType::RevFuncType {
                in_out_ty: Box::new(in_out_ty.substitute_dim_var(dim_var, new_dim_expr)),
            },

            MetaType::RegType { elem_ty, dim } => MetaType::RegType {
                elem_ty: *elem_ty,
                dim: dim.substitute_dim_var(dim_var, new_dim_expr),
            },

            MetaType::TupleType { tys } => {
                let new_tys = tys
                    .iter()
                    .map(|ty| ty.substitute_dim_var(dim_var.clone(), new_dim_expr.clone()))
                    .collect();
                MetaType::TupleType { tys: new_tys }
            }

            // Doesn't contain a dimvar expr
            MetaType::UnitType => MetaType::UnitType,
        }
    }
}

impl classical::MetaExpr {
    pub fn substitute_dim_var(
        &self,
        dim_var: DimVar,
        new_dim_expr: DimExpr,
    ) -> classical::MetaExpr {
        match self {
            classical::MetaExpr::Mod {
                dividend,
                divisor,
                dbg,
            } => classical::MetaExpr::Mod {
                dividend: Box::new(
                    dividend.substitute_dim_var(dim_var.clone(), new_dim_expr.clone()),
                ),
                divisor: divisor.substitute_dim_var(dim_var, new_dim_expr),
                dbg: dbg.clone(),
            },

            classical::MetaExpr::Slice {
                val,
                lower,
                upper,
                dbg,
            } => classical::MetaExpr::Slice {
                val: Box::new(val.substitute_dim_var(dim_var.clone(), new_dim_expr.clone())),
                lower: lower.substitute_dim_var(dim_var.clone(), new_dim_expr.clone()),
                upper: upper.substitute_dim_var(dim_var, new_dim_expr),
                dbg: dbg.clone(),
            },

            classical::MetaExpr::UnaryOp { kind, val, dbg } => classical::MetaExpr::UnaryOp {
                kind: *kind,
                val: Box::new(val.substitute_dim_var(dim_var, new_dim_expr)),
                dbg: dbg.clone(),
            },

            classical::MetaExpr::BinaryOp {
                kind,
                left,
                right,
                dbg,
            } => classical::MetaExpr::BinaryOp {
                kind: *kind,
                left: Box::new(left.substitute_dim_var(dim_var.clone(), new_dim_expr.clone())),
                right: Box::new(right.substitute_dim_var(dim_var, new_dim_expr)),
                dbg: dbg.clone(),
            },

            classical::MetaExpr::ReduceOp { kind, val, dbg } => classical::MetaExpr::ReduceOp {
                kind: *kind,
                val: Box::new(val.substitute_dim_var(dim_var, new_dim_expr)),
                dbg: dbg.clone(),
            },

            classical::MetaExpr::ModMul {
                x,
                j,
                y,
                mod_n,
                dbg,
            } => classical::MetaExpr::ModMul {
                x: x.substitute_dim_var(dim_var.clone(), new_dim_expr.clone()),
                j: j.substitute_dim_var(dim_var.clone(), new_dim_expr.clone()),
                y: Box::new(y.substitute_dim_var(dim_var.clone(), new_dim_expr.clone())),
                mod_n: mod_n.substitute_dim_var(dim_var.clone(), new_dim_expr.clone()),
                dbg: dbg.clone(),
            },

            classical::MetaExpr::BitLiteral { val, n_bits, dbg } => {
                classical::MetaExpr::BitLiteral {
                    val: val.clone(),
                    n_bits: n_bits.substitute_dim_var(dim_var, new_dim_expr),
                    dbg: dbg.clone(),
                }
            }

            // Doesn't contain an expr/dimvar expr
            classical::MetaExpr::Variable { .. } => self.clone(),
        }
    }
}

impl qpu::MetaExpr {
    fn expand_instantiations<F>(&self, mut f: F) -> Result<(qpu::MetaExpr, F), LowerError>
    where
        F: FnMut(String, DimExpr, Option<DebugLoc>) -> Result<String, LowerError>,
    {
        Ok(match self {
            qpu::MetaExpr::ExprMacro { name, arg, dbg } => {
                let (new_arg, moved_cb) = arg.expand_instantiations(f)?;
                (
                    qpu::MetaExpr::ExprMacro {
                        name: name.to_string(),
                        arg: Box::new(new_arg),
                        dbg: dbg.clone(),
                    },
                    moved_cb,
                )
            }

            qpu::MetaExpr::BroadcastTensor { val, factor, dbg } => {
                let (new_val, moved_cb) = val.expand_instantiations(f)?;
                (
                    qpu::MetaExpr::BroadcastTensor {
                        val: Box::new(new_val),
                        factor: factor.clone(),
                        dbg: dbg.clone(),
                    },
                    moved_cb,
                )
            }

            qpu::MetaExpr::Instantiate { name, param, dbg } => (
                qpu::MetaExpr::Variable {
                    name: f(name.to_string(), param.clone(), dbg.clone())?,
                    dbg: dbg.clone(),
                },
                f,
            ),

            qpu::MetaExpr::Repeat {
                for_each,
                iter_var,
                upper_bound,
                dbg,
            } => {
                let (new_for_each, moved_cb) = for_each.expand_instantiations(f)?;
                (
                    qpu::MetaExpr::Repeat {
                        for_each: Box::new(new_for_each),
                        iter_var: iter_var.to_string(),
                        upper_bound: upper_bound.clone(),
                        dbg: dbg.clone(),
                    },
                    moved_cb,
                )
            }

            qpu::MetaExpr::EmbedClassical {
                func,
                embed_kind,
                dbg,
            } => {
                let (new_func, moved_cb) = func.expand_instantiations(f)?;
                (
                    qpu::MetaExpr::EmbedClassical {
                        func: Box::new(new_func),
                        embed_kind: *embed_kind,
                        dbg: dbg.clone(),
                    },
                    moved_cb,
                )
            }

            qpu::MetaExpr::Adjoint { func, dbg } => {
                let (new_func, moved_cb) = func.expand_instantiations(f)?;
                (
                    qpu::MetaExpr::Adjoint {
                        func: Box::new(new_func),
                        dbg: dbg.clone(),
                    },
                    moved_cb,
                )
            }

            qpu::MetaExpr::Pipe { lhs, rhs, dbg } => {
                let (new_lhs, moved_cb) = lhs.expand_instantiations(f)?;
                let (new_rhs, moved_cb) = rhs.expand_instantiations(moved_cb)?;
                (
                    qpu::MetaExpr::Pipe {
                        lhs: Box::new(new_lhs),
                        rhs: Box::new(new_rhs),
                        dbg: dbg.clone(),
                    },
                    moved_cb,
                )
            }

            qpu::MetaExpr::BiTensor { left, right, dbg } => {
                let (new_left, moved_cb) = left.expand_instantiations(f)?;
                let (new_right, moved_cb) = right.expand_instantiations(moved_cb)?;
                (
                    qpu::MetaExpr::BiTensor {
                        left: Box::new(new_left),
                        right: Box::new(new_right),
                        dbg: dbg.clone(),
                    },
                    moved_cb,
                )
            }

            qpu::MetaExpr::Predicated {
                then_func,
                else_func,
                pred,
                dbg,
            } => {
                let (new_then_func, moved_cb) = then_func.expand_instantiations(f)?;
                let (new_else_func, moved_cb) = else_func.expand_instantiations(moved_cb)?;
                (
                    qpu::MetaExpr::Predicated {
                        then_func: Box::new(new_then_func),
                        else_func: Box::new(new_else_func),
                        pred: pred.clone(),
                        dbg: dbg.clone(),
                    },
                    moved_cb,
                )
            }

            qpu::MetaExpr::Conditional {
                then_expr,
                else_expr,
                cond,
                dbg,
            } => {
                let (new_then_expr, moved_cb) = then_expr.expand_instantiations(f)?;
                let (new_else_expr, moved_cb) = else_expr.expand_instantiations(moved_cb)?;
                let (new_cond, moved_cb) = cond.expand_instantiations(moved_cb)?;
                (
                    qpu::MetaExpr::Conditional {
                        then_expr: Box::new(new_then_expr),
                        else_expr: Box::new(new_else_expr),
                        cond: Box::new(new_cond),
                        dbg: dbg.clone(),
                    },
                    moved_cb,
                )
            }

            // Doesn't contain an expression
            qpu::MetaExpr::BasisMacro { .. }
            | qpu::MetaExpr::Variable { .. }
            | qpu::MetaExpr::UnitLiteral { .. }
            | qpu::MetaExpr::Measure { .. }
            | qpu::MetaExpr::Discard { .. }
            | qpu::MetaExpr::BasisTranslation { .. }
            | qpu::MetaExpr::NonUniformSuperpos { .. }
            | qpu::MetaExpr::Ensemble { .. }
            | qpu::MetaExpr::QLit { .. }
            | qpu::MetaExpr::BitLiteral { .. } => (self.clone(), f),
        })
    }
}

pub trait InstantationExpandable {
    fn expand_instantiations<F>(&self, f: F) -> Result<(Self, F), LowerError>
    where
        F: FnMut(String, DimExpr, Option<DebugLoc>) -> Result<String, LowerError>,
        Self: Sized;
}

impl InstantationExpandable for classical::MetaStmt {
    fn expand_instantiations<F>(&self, f: F) -> Result<(classical::MetaStmt, F), LowerError>
    where
        F: FnMut(String, DimExpr, Option<DebugLoc>) -> Result<String, LowerError>,
    {
        // @classical kernels don't have an instantiation expression
        Ok((self.clone(), f))
    }
}

impl InstantationExpandable for qpu::MetaStmt {
    fn expand_instantiations<F>(&self, f: F) -> Result<(qpu::MetaStmt, F), LowerError>
    where
        F: FnMut(String, DimExpr, Option<DebugLoc>) -> Result<String, LowerError>,
    {
        Ok(match self {
            qpu::MetaStmt::ExprMacroDef {
                lhs_pat,
                lhs_name,
                rhs,
                dbg,
            } => {
                let (new_rhs, moved_cb) = rhs.expand_instantiations(f)?;
                (
                    qpu::MetaStmt::ExprMacroDef {
                        lhs_pat: lhs_pat.clone(),
                        lhs_name: lhs_name.to_string(),
                        rhs: new_rhs,
                        dbg: dbg.clone(),
                    },
                    moved_cb,
                )
            }
            qpu::MetaStmt::BasisMacroDef {
                lhs_pat,
                lhs_name,
                rhs,
                dbg,
            } => {
                let (new_rhs, moved_cb) = rhs.expand_instantiations(f)?;
                (
                    qpu::MetaStmt::BasisMacroDef {
                        lhs_pat: lhs_pat.clone(),
                        lhs_name: lhs_name.to_string(),
                        rhs: new_rhs,
                        dbg: dbg.clone(),
                    },
                    moved_cb,
                )
            }
            qpu::MetaStmt::Expr { expr } => {
                let (new_expr, moved_cb) = expr.expand_instantiations(f)?;
                (qpu::MetaStmt::Expr { expr: new_expr }, moved_cb)
            }
            qpu::MetaStmt::Assign { lhs, rhs, dbg } => {
                let (new_rhs, moved_cb) = rhs.expand_instantiations(f)?;
                (
                    qpu::MetaStmt::Assign {
                        lhs: lhs.to_string(),
                        rhs: new_rhs,
                        dbg: dbg.clone(),
                    },
                    moved_cb,
                )
            }
            qpu::MetaStmt::UnpackAssign { lhs, rhs, dbg } => {
                let (new_rhs, moved_cb) = rhs.expand_instantiations(f)?;
                (
                    qpu::MetaStmt::UnpackAssign {
                        lhs: lhs.clone(),
                        rhs: new_rhs,
                        dbg: dbg.clone(),
                    },
                    moved_cb,
                )
            }
            qpu::MetaStmt::Return { val, dbg } => {
                let (new_val, moved_cb) = val.expand_instantiations(f)?;
                (
                    qpu::MetaStmt::Return {
                        val: new_val,
                        dbg: dbg.clone(),
                    },
                    moved_cb,
                )
            }

            // Doesn't contain an expression
            qpu::MetaStmt::BasisGeneratorMacroDef { .. }
            | qpu::MetaStmt::VectorSymbolDef { .. }
            | qpu::MetaStmt::BasisAliasDef { .. }
            | qpu::MetaStmt::BasisAliasRecDef { .. } => (self.clone(), f),
        })
    }
}

impl<S: InstantationExpandable> MetaFunctionDef<S> {
    fn expand_instantiations<F>(&self, f: F) -> Result<(MetaFunctionDef<S>, F), LowerError>
    where
        F: FnMut(String, DimExpr, Option<DebugLoc>) -> Result<String, LowerError>,
    {
        let MetaFunctionDef {
            name,
            args,
            ret_type,
            body,
            is_rev,
            dim_vars,
            dbg,
        } = self;

        let mut new_body = vec![];
        let mut next_callback = f;
        for stmt in body {
            let (new_stmt, moved_cb) = stmt.expand_instantiations(next_callback)?;
            new_body.push(new_stmt);
            next_callback = moved_cb;
        }

        Ok((
            MetaFunctionDef {
                name: name.to_string(),
                args: args.clone(),
                ret_type: ret_type.clone(),
                body: new_body,
                is_rev: *is_rev,
                dim_vars: dim_vars.clone(),
                dbg: dbg.clone(),
            },
            next_callback,
        ))
    }
}

pub trait Substitutable {
    fn substitute_dim_var(&self, dim_var: DimVar, new_dim_expr: DimExpr) -> Self;
}

impl Substitutable for classical::MetaStmt {
    fn substitute_dim_var(&self, dim_var: DimVar, new_dim_expr: DimExpr) -> classical::MetaStmt {
        match self {
            classical::MetaStmt::Expr { expr } => classical::MetaStmt::Expr {
                expr: expr.substitute_dim_var(dim_var, new_dim_expr),
            },
            classical::MetaStmt::Assign { lhs, rhs, dbg } => classical::MetaStmt::Assign {
                lhs: lhs.to_string(),
                rhs: rhs.substitute_dim_var(dim_var, new_dim_expr),
                dbg: dbg.clone(),
            },
            classical::MetaStmt::UnpackAssign { lhs, rhs, dbg } => {
                classical::MetaStmt::UnpackAssign {
                    lhs: lhs.clone(),
                    rhs: rhs.substitute_dim_var(dim_var, new_dim_expr),
                    dbg: dbg.clone(),
                }
            }
            classical::MetaStmt::Return { val, dbg } => classical::MetaStmt::Return {
                val: val.substitute_dim_var(dim_var, new_dim_expr),
                dbg: dbg.clone(),
            },
        }
    }
}

impl Substitutable for qpu::MetaStmt {
    fn substitute_dim_var(&self, dim_var: DimVar, new_dim_expr: DimExpr) -> qpu::MetaStmt {
        match self {
            qpu::MetaStmt::ExprMacroDef {
                lhs_pat,
                lhs_name,
                rhs,
                dbg,
            } => qpu::MetaStmt::ExprMacroDef {
                lhs_pat: lhs_pat.clone(),
                lhs_name: lhs_name.to_string(),
                rhs: rhs.substitute_dim_var(dim_var, new_dim_expr),
                dbg: dbg.clone(),
            },
            qpu::MetaStmt::BasisMacroDef {
                lhs_pat,
                lhs_name,
                rhs,
                dbg,
            } => qpu::MetaStmt::BasisMacroDef {
                lhs_pat: lhs_pat.clone(),
                lhs_name: lhs_name.to_string(),
                rhs: rhs.substitute_dim_var(dim_var, new_dim_expr),
                dbg: dbg.clone(),
            },
            qpu::MetaStmt::BasisGeneratorMacroDef {
                lhs_pat,
                lhs_name,
                rhs,
                dbg,
            } => qpu::MetaStmt::BasisGeneratorMacroDef {
                lhs_pat: lhs_pat.clone(),
                lhs_name: lhs_name.to_string(),
                rhs: rhs.substitute_dim_var(dim_var, new_dim_expr),
                dbg: dbg.clone(),
            },
            qpu::MetaStmt::VectorSymbolDef { lhs, rhs, dbg } => qpu::MetaStmt::VectorSymbolDef {
                lhs: *lhs,
                rhs: rhs.substitute_dim_var(dim_var, new_dim_expr),
                dbg: dbg.clone(),
            },
            qpu::MetaStmt::BasisAliasDef { lhs, rhs, dbg } => qpu::MetaStmt::BasisAliasDef {
                lhs: lhs.to_string(),
                rhs: rhs.substitute_dim_var(dim_var, new_dim_expr),
                dbg: dbg.clone(),
            },
            qpu::MetaStmt::BasisAliasRecDef {
                lhs,
                param,
                rhs,
                dbg,
            } => qpu::MetaStmt::BasisAliasRecDef {
                lhs: lhs.to_string(),
                param: param.clone(),
                rhs: rhs.substitute_dim_var(dim_var, new_dim_expr),
                dbg: dbg.clone(),
            },
            qpu::MetaStmt::Expr { expr } => qpu::MetaStmt::Expr {
                expr: expr.substitute_dim_var(dim_var, new_dim_expr),
            },
            qpu::MetaStmt::Assign { lhs, rhs, dbg } => qpu::MetaStmt::Assign {
                lhs: lhs.to_string(),
                rhs: rhs.substitute_dim_var(dim_var, new_dim_expr),
                dbg: dbg.clone(),
            },
            qpu::MetaStmt::UnpackAssign { lhs, rhs, dbg } => qpu::MetaStmt::UnpackAssign {
                lhs: lhs.clone(),
                rhs: rhs.substitute_dim_var(dim_var, new_dim_expr),
                dbg: dbg.clone(),
            },
            qpu::MetaStmt::Return { val, dbg } => qpu::MetaStmt::Return {
                val: val.substitute_dim_var(dim_var, new_dim_expr),
                dbg: dbg.clone(),
            },
        }
    }
}

impl<S: Substitutable> MetaFunctionDef<S> {
    fn instantiate(
        &self,
        new_name: String,
        dim_var_name: String,
        new_dim_expr: DimExpr,
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

        let dv = DimVar::FuncVar {
            var_name: dim_var_name.to_string(),
            func_name: name.to_string(),
        };

        let new_args: Vec<_> = args
            .iter()
            .map(|(arg_ty, arg_name)| {
                let new_arg_ty = arg_ty
                    .clone()
                    .map(|ty| ty.substitute_dim_var(dv.clone(), new_dim_expr.clone()));
                (new_arg_ty, arg_name.to_string())
            })
            .collect();
        let new_ret_type = ret_type
            .clone()
            .map(|ty| ty.substitute_dim_var(dv.clone(), new_dim_expr.clone()));
        let new_body: Vec<_> = body
            .iter()
            .map(|stmt| stmt.substitute_dim_var(dv.clone(), new_dim_expr.clone()))
            .collect();
        let new_dim_vars: Vec<_> = dim_vars
            .iter()
            .filter(|name| **name != dim_var_name)
            .cloned()
            .collect();
        assert_eq!(new_dim_vars.len() + 1, dim_vars.len());

        MetaFunctionDef {
            name: new_name,
            args: new_args,
            ret_type: new_ret_type,
            body: new_body,
            is_rev: *is_rev,
            dim_vars: new_dim_vars,
            dbg: dbg.clone(),
        }
    }
}

impl MetaFunc {
    fn instantiate(&self, new_name: String, dim_var_name: String, new_expr: DimExpr) -> MetaFunc {
        match self {
            MetaFunc::Qpu(qpu_func) => {
                MetaFunc::Qpu(qpu_func.instantiate(new_name, dim_var_name, new_expr))
            }
            MetaFunc::Classical(classical_func) => {
                MetaFunc::Classical(classical_func.instantiate(new_name, dim_var_name, new_expr))
            }
        }
    }

    fn expand_instantiations<F>(&self, f: F) -> Result<(MetaFunc, F), LowerError>
    where
        F: FnMut(String, DimExpr, Option<DebugLoc>) -> Result<String, LowerError>,
    {
        Ok(match self {
            MetaFunc::Qpu(qpu_func) => {
                let (new_func, moved_cb) = qpu_func.expand_instantiations(f)?;
                (MetaFunc::Qpu(new_func), moved_cb)
            }
            MetaFunc::Classical(classical_func) => {
                let (new_func, moved_cb) = classical_func.expand_instantiations(f)?;
                (MetaFunc::Classical(new_func), moved_cb)
            }
        })
    }
}

impl MetaProgram {
    fn missing_dim_vars(
        &self,
        dv_assign: &DimVarAssignments,
    ) -> impl Iterator<Item = (MetaFunc, Vec<String>)> {
        self.funcs
            .iter()
            .zip(dv_assign.iter())
            .filter_map(|(func, func_dvs)| {
                let missing_dv_names: Vec<_> = func
                    .get_dim_vars()
                    .into_iter()
                    .zip(func_dvs.clone().into_iter())
                    .filter_map(|(dv_name, dv_val)| {
                        if dv_val.is_none() {
                            Some(dv_name.to_string())
                        } else {
                            None
                        }
                    })
                    .collect();

                if missing_dv_names.is_empty() {
                    None
                } else {
                    Some((func.clone(), missing_dv_names))
                }
            })
    }

    fn do_instantiations(
        &self,
        dv_assign: DimVarAssignments,
    ) -> Result<(MetaProgram, DimVarAssignments), LowerError> {
        let candidates = self
            .missing_dim_vars(&dv_assign)
            .filter_map(|(func_def, missing_dvs)| {
                let func_name = func_def.get_name().to_string();
                if missing_dvs.len() > 1 {
                    Some(Err(LowerError {
                        kind: LowerErrorKind::CannotInferDimVar {
                            dim_var_names: missing_dvs,
                        },
                        dbg: func_def.get_dbg().clone(),
                    }))
                } else if missing_dvs.len() == 1 {
                    Some(Ok((func_name, (missing_dvs[0].to_string(), func_def))))
                } else {
                    None
                }
            })
            .collect::<Result<HashMap<_, _>, LowerError>>()?;

        let mut inst_counter = 0usize;
        // Mapping of (func name, param) -> index in the map below
        let mut instantiated_funcs = HashMap::new();
        // mapping of original func name -> Vec<MetaFunc>
        let mut func_instantiations = HashMap::new();

        let mut expanded_funcs = vec![];
        for func in self.funcs.iter().rev() {
            let expanded_func = if func_instantiations.contains_key(func.get_name()) {
                func.clone()
            } else {
                let callback = |func_name: String, param_val: DimExpr, dbg: Option<DebugLoc>| {
                    if let Some((missing_dv, func_def)) = candidates.get(&func_name) {
                        let instance_idx = instantiated_funcs
                            .entry((func_name.to_string(), param_val.clone()))
                            .or_insert_with(|| {
                                // TODO: make sure this is a unique function name
                                let new_name = format!("{}__inst{}", func_name, inst_counter);
                                inst_counter += 1;
                                let instance = func_def.instantiate(
                                    new_name,
                                    missing_dv.to_string(),
                                    param_val,
                                );
                                let instances = func_instantiations
                                    .entry(func_name.to_string())
                                    .or_insert_with(Vec::new);
                                instances.push(instance);
                                instances.len() - 1
                            });
                        let instance = &func_instantiations[&func_name][*instance_idx];
                        let instance_name = instance.get_name().to_string();
                        Ok(instance_name)
                    } else {
                        Err(LowerError {
                            kind: LowerErrorKind::CannotInstantiate {
                                func_name: func_name.to_string(),
                            },
                            dbg: dbg.clone(),
                        })
                    }
                };

                let (new_func, _moved_cb) = func.expand_instantiations(callback)?;
                new_func
            };
            expanded_funcs.push(expanded_func);
        }

        // mapping of original func name -> Vec<MetaFunc>
        //let mut func_instantiations = HashMap::new();
        //for ((func_name, _param_val), instance) in instantiated_funcs.drain() {
        //}

        // If we are introducing or removing functions then we have to update dv_assign
        let (new_funcs, new_func_dv_assigns): (Vec<_>, Vec<_>) = expanded_funcs
            .into_iter()
            .rev()
            .zip(dv_assign.into_iter())
            .flat_map(|(func, func_dv_assign)| {
                if let Some(instances) = func_instantiations.get(func.get_name()) {
                    let (removed_dv_name, _func) = &candidates[func.get_name()];
                    let instance_func_dv_assign =
                        func_dv_assign.without_dv(&func, &removed_dv_name);
                    instances
                        .iter()
                        .cloned()
                        .map(|instance| (instance, instance_func_dv_assign.clone()))
                        .collect()
                } else {
                    vec![(func, func_dv_assign)]
                }
            })
            .unzip();
        let new_dv_assign = DimVarAssignments::new(new_func_dv_assigns);

        Ok((
            MetaProgram {
                funcs: new_funcs,
                dbg: self.dbg.clone(),
            },
            new_dv_assign,
        ))
    }

    fn round_of_lowering(
        &self,
        init_dv_assign: DimVarAssignments,
    ) -> Result<(MetaProgram, DimVarAssignments, Progress), LowerError> {
        eprintln!("000000 ROUND OF LOWERING BEGINS: {:#?}", self);
        let mut dv_assign = init_dv_assign;
        let (mut program, _expand_progress) = self.expand(&dv_assign)?;

        loop {
            let (new_program, new_dv_assign, infer_progress) = program.infer(dv_assign)?;
            let (new_program, expand_progress) = new_program.expand(&new_dv_assign)?;
            let progress = infer_progress.join(expand_progress);

            if progress.is_finished() || program == new_program {
                return Ok((new_program, new_dv_assign, progress));
            } else {
                dv_assign = new_dv_assign;
                program = new_program;
                continue;
            }
        }
    }

    pub fn lower(&self) -> Result<ast::Program, LowerError> {
        let init_dv_assign = DimVarAssignments::empty(self);
        let (new_prog, dv_assign, _progress) = self.round_of_lowering(init_dv_assign)?;
        eprintln!("1111111 dimvars after 1 round: {:#?}", dv_assign);
        eprintln!("1111111 full program after 1 round: {:#?}", new_prog);
        let (new_prog, dv_assign) = new_prog.do_instantiations(dv_assign)?;
        let (new_prog, dv_assign, progress) = new_prog.round_of_lowering(dv_assign)?;

        if let Some((func, missing_dvs)) = new_prog.missing_dim_vars(&dv_assign).next() {
            Err(LowerError {
                kind: LowerErrorKind::CannotInferDimVar {
                    dim_var_names: missing_dvs,
                },
                dbg: func.get_dbg().clone(),
            })
        } else if !progress.is_finished() {
            Err(LowerError {
                kind: LowerErrorKind::Stuck,
                dbg: self.dbg.clone(),
            })
        } else {
            new_prog.extract()
        }
    }
}
