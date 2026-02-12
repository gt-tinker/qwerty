use crate::{
    ast,
    dbg::DebugLoc,
    error::{LowerError, LowerErrorKind},
    meta::{
        DimExpr, DimVar, MetaFunc, MetaFunctionDef, MetaProgram, MetaType, classical,
        expand::{Expandable, MacroEnv},
        infer::DimVarAssignments,
        qpu,
        type_dim::meta_type,
    },
    typecheck,
};
use qwerty_ast_macros::rebuild;
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
    pub fn substitute_dim_var(self, dim_var: &DimVar, new_dim_expr: &DimExpr) -> Self {
        rebuild!(MetaType, self, substitute_dim_var, dim_var, new_dim_expr)
    }
}

impl classical::MetaExpr {
    pub fn substitute_dim_var(self, dim_var: &DimVar, new_dim_expr: &DimExpr) -> Self {
        rebuild!(
            classical::MetaExpr,
            self,
            substitute_dim_var,
            dim_var,
            new_dim_expr
        )
    }
}

impl qpu::MetaExpr {
    /// Called by the `gen_rebuild` attribute macro invoked in `meta/qpu.rs`.
    pub(crate) fn expand_instantiations_rewriter<F>(self, mut f: F) -> Result<(Self, F), LowerError>
    where
        F: FnMut(String, DimExpr, Option<DebugLoc>) -> Result<String, LowerError>,
    {
        match self {
            qpu::MetaExpr::Instantiate { name, param, dbg } => Ok((
                qpu::MetaExpr::Variable {
                    name: f(name, param, dbg.clone())?,
                    dbg: dbg,
                },
                f,
            )),

            other => Ok((other, f)),
        }
    }

    pub fn expand_instantiations<F>(self, f: F) -> Result<(Self, F), LowerError>
    where
        F: FnMut(String, DimExpr, Option<DebugLoc>) -> Result<String, LowerError>,
    {
        rebuild!(qpu::MetaExpr, self, expand_instantiations, f)
    }
}

pub trait InstantationExpandable {
    fn expand_instantiations<F>(self, f: F) -> Result<(Self, F), LowerError>
    where
        F: FnMut(String, DimExpr, Option<DebugLoc>) -> Result<String, LowerError>,
        Self: Sized;
}

impl InstantationExpandable for classical::MetaStmt {
    fn expand_instantiations<F>(self, f: F) -> Result<(classical::MetaStmt, F), LowerError>
    where
        F: FnMut(String, DimExpr, Option<DebugLoc>) -> Result<String, LowerError>,
    {
        // @classical kernels don't have an instantiation expression
        Ok((self, f))
    }
}

impl InstantationExpandable for qpu::MetaStmt {
    fn expand_instantiations<F>(self, f: F) -> Result<(qpu::MetaStmt, F), LowerError>
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

            // Doesn't contain a @qpu expression
            qpu::MetaStmt::BasisGeneratorMacroDef { .. }
            | qpu::MetaStmt::VectorSymbolDef { .. }
            | qpu::MetaStmt::BasisAliasDef { .. }
            | qpu::MetaStmt::BasisAliasRecDef { .. }
            | qpu::MetaStmt::ClassicalLambdaDef { .. } => (self.clone(), f),
        })
    }
}

impl<S: InstantationExpandable> MetaFunctionDef<S> {
    fn expand_instantiations<F>(self, f: F) -> Result<(MetaFunctionDef<S>, F), LowerError>
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
                name: name,
                args: args,
                ret_type: ret_type,
                body: new_body,
                is_rev: is_rev,
                dim_vars: dim_vars,
                dbg: dbg,
            },
            next_callback,
        ))
    }
}

pub trait Substitutable {
    fn substitute_dim_var(self, dim_var: &DimVar, new_dim_expr: &DimExpr) -> Self;
}

impl Substitutable for classical::MetaStmt {
    fn substitute_dim_var(self, dim_var: &DimVar, new_dim_expr: &DimExpr) -> Self {
        rebuild!(
            classical::MetaStmt,
            self,
            substitute_dim_var,
            dim_var,
            new_dim_expr
        )
    }
}

impl Substitutable for qpu::MetaStmt {
    fn substitute_dim_var(self, dim_var: &DimVar, new_dim_expr: &DimExpr) -> Self {
        rebuild!(
            qpu::MetaStmt,
            self,
            substitute_dim_var,
            dim_var,
            new_dim_expr
        )
    }
}

impl<S: Substitutable> MetaFunctionDef<S> {
    fn instantiate(
        self,
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
            func_name: name,
        };

        let new_args: Vec<_> = args
            .into_iter()
            .map(|(arg_ty, arg_name)| {
                let new_arg_ty = arg_ty.map(|ty| ty.substitute_dim_var(&dv, &new_dim_expr));
                (new_arg_ty, arg_name)
            })
            .collect();
        let new_ret_type = ret_type.map(|ty| ty.substitute_dim_var(&dv, &new_dim_expr));
        let new_body: Vec<_> = body
            .into_iter()
            .map(|stmt| stmt.substitute_dim_var(&dv, &new_dim_expr))
            .collect();
        let expected_num_dim_vars = dim_vars.len();
        let new_dim_vars: Vec<_> = dim_vars
            .into_iter()
            .filter(|name| *name != dim_var_name)
            .collect();
        assert_eq!(new_dim_vars.len() + 1, expected_num_dim_vars);

        MetaFunctionDef {
            name: new_name,
            args: new_args,
            ret_type: new_ret_type,
            body: new_body,
            is_rev,
            dim_vars: new_dim_vars,
            dbg,
        }
    }
}

impl MetaFunc {
    fn instantiate(self, new_name: String, dim_var_name: String, new_expr: DimExpr) -> MetaFunc {
        match self {
            MetaFunc::Qpu(qpu_func) => {
                MetaFunc::Qpu(qpu_func.instantiate(new_name, dim_var_name, new_expr))
            }
            MetaFunc::Classical(classical_func) => {
                MetaFunc::Classical(classical_func.instantiate(new_name, dim_var_name, new_expr))
            }
        }
    }

    fn expand_instantiations<F>(self, f: F) -> Result<(MetaFunc, F), LowerError>
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
        self.funcs.iter().filter_map(|func| {
            let missing_dv_names: Vec<_> = func
                .get_dim_vars()
                .into_iter()
                .filter(|dv_name| {
                    let dv = DimVar::FuncVar {
                        var_name: dv_name.to_string(),
                        func_name: func.get_name().to_string(),
                    };
                    !dv_assign.contains_key(&dv)
                })
                .cloned()
                .collect();

            if missing_dv_names.is_empty() {
                None
            } else {
                Some((func.clone(), missing_dv_names))
            }
        })
    }

    fn do_instantiations(
        self,
        old_dv_assign: DimVarAssignments,
    ) -> Result<(MetaProgram, DimVarAssignments), LowerError> {
        let candidates = self
            .missing_dim_vars(&old_dv_assign)
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
                    Some(Ok((func_name, missing_dvs[0].to_string())))
                } else {
                    None
                }
            })
            .collect::<Result<HashMap<_, _>, LowerError>>()?;

        let mut inst_counter = 0usize;
        // Mapping of (func name, param) -> name
        let mut instantiated_funcs = HashMap::new();
        // Mapping of func name -> (name, dv_name, param)
        let mut funcs_instantiated: HashMap<String, Vec<(String, String, DimExpr)>> =
            HashMap::new();

        // Stack
        let mut dv_assign = old_dv_assign;
        let mut func_worklist: Vec<_> = self.funcs.clone();
        let mut new_funcs = vec![];

        while let Some(func) = func_worklist.pop() {
            if let Some(instances) = funcs_instantiated.get(func.get_name()) {
                let mut instance_names = vec![];
                for (instance_name, missing_dv, param_val) in instances {
                    let instance = func.clone().instantiate(
                        instance_name.to_string(),
                        missing_dv.to_string(),
                        param_val.clone(),
                    );
                    instance_names.push(instance_name.to_string());
                    func_worklist.push(instance);
                }
                dv_assign.move_dim_var_values_to_instances(&func, &instance_names);
            } else {
                let callback = |func_name: String, param_val: DimExpr, dbg: Option<DebugLoc>| {
                    if let Some(missing_dv) = candidates.get(&func_name) {
                        let instance_name = instantiated_funcs
                            .entry((func_name.to_string(), param_val.clone()))
                            .or_insert_with(|| {
                                // TODO: make sure this is a unique function name
                                let instance_name = format!("{}__inst{}", func_name, inst_counter);
                                inst_counter += 1;

                                // Make a note that we need to instantiate this
                                let instances = funcs_instantiated
                                    .entry(func_name.to_string())
                                    .or_insert_with(Vec::new);
                                instances.push((
                                    instance_name.to_string(),
                                    missing_dv.to_string(),
                                    param_val.clone(),
                                ));
                                instance_name
                            })
                            .to_string();
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
                new_funcs.push(new_func);
            }
        }

        new_funcs.reverse();

        Ok((
            MetaProgram {
                funcs: new_funcs,
                dbg: self.dbg.clone(),
            },
            dv_assign,
        ))
    }

    fn round_of_lowering(
        self,
        init_dv_assign: DimVarAssignments,
    ) -> Result<(MetaProgram, DimVarAssignments, Progress), LowerError> {
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

    pub fn lower(self) -> Result<ast::Program, LowerError> {
        let dbg = self.dbg.clone();
        let init_dv_assign = DimVarAssignments::empty();
        let (new_prog, dv_assign, _progress) = self.round_of_lowering(init_dv_assign)?;
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
                dbg,
            })
        } else {
            new_prog.extract()
        }
    }
}

impl qpu::MetaStmt {
    pub fn lower(
        self,
        init_env: &mut MacroEnv,
        plain_ty_env: &typecheck::TypeEnv,
    ) -> Result<qpu::LoweredStmt, LowerError> {
        let mut dv_assign = init_env.to_dv_assign();
        let (mut stmt, _expand_progress) = self.expand(&mut init_env.clone())?;

        loop {
            let new_dv_assign = stmt.infer(dv_assign, plain_ty_env)?;
            let mut env = init_env.clone();
            env.update_from_dv_assign(&new_dv_assign)?;
            let old_stmt = stmt;
            let (new_stmt, expand_progress) = old_stmt.clone().expand(&mut env)?;

            if expand_progress.is_finished() || old_stmt == new_stmt {
                // Use init_env as an output parameter
                *init_env = env;
                break new_stmt.extract();
            } else {
                dv_assign = new_dv_assign;
                stmt = new_stmt;
                continue;
            }
        }
    }
}
