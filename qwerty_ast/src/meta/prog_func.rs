use crate::{
    ast,
    dbg::DebugLoc,
    error::ExtractError,
    meta::{MetaType, classical, qpu},
};

/// A function (kernel) definition.
///
/// Example syntax:
/// ```text
/// @qpu[[N]]
/// def get_zero() -> qubit[N]:
///     return '0'**N
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct MetaFunctionDef<S> {
    pub name: String,
    pub args: Vec<(MetaType, String)>,
    pub ret_type: MetaType,
    pub body: Vec<S>,
    pub is_rev: bool,
    pub dim_vars: Vec<String>,
    pub dbg: Option<DebugLoc>,
}

// TODO: don't duplicate this with MetaFunctionDef<classical::MetaStmt>
impl MetaFunctionDef<qpu::MetaStmt> {
    /// Extract a plain `@qpu` `FunctionDef` from this MetaQwerty `@qpu`
    /// `MetaFunctionDef` or return an error if e.g. contained dimension variable
    /// expressions are not fully folded yet.
    pub fn extract(&self) -> Result<ast::FunctionDef<ast::qpu::Expr>, ExtractError> {
        let MetaFunctionDef {
            name,
            args,
            ret_type,
            body,
            is_rev,
            dim_vars: _,
            dbg,
        } = self;

        let ast_args = args
            .iter()
            .map(|(arg_ty, arg_name)| {
                arg_ty
                    .extract()
                    .map(|ast_arg_ty| (ast_arg_ty, arg_name.to_string()))
            })
            .collect::<Result<Vec<(ast::Type, String)>, ExtractError>>()?;
        let ast_body = body
            .iter()
            .map(|stmt| stmt.extract())
            .collect::<Result<Vec<ast::Stmt<ast::qpu::Expr>>, ExtractError>>()?;

        Ok(ast::FunctionDef {
            name: name.to_string(),
            args: ast_args,
            ret_type: ret_type.extract()?,
            body: ast_body,
            is_rev: *is_rev,
            dbg: dbg.clone(),
        })
    }
}

// TODO: don't duplicate this with MetaFunctionDef<qpu::MetaStmt>
impl MetaFunctionDef<classical::MetaStmt> {
    /// Extract a plain `@classical` `FunctionDef` from this MetaQwerty `@classical`
    /// `MetaFunctionDef` or return an error if e.g. contained dimension variable
    /// expressions are not fully folded yet.
    pub fn extract(&self) -> Result<ast::FunctionDef<ast::classical::Expr>, ExtractError> {
        let MetaFunctionDef {
            name,
            args,
            ret_type,
            body,
            is_rev,
            dim_vars: _,
            dbg,
        } = self;

        let ast_args = args
            .iter()
            .map(|(arg_ty, arg_name)| {
                arg_ty
                    .extract()
                    .map(|ast_arg_ty| (ast_arg_ty, arg_name.to_string()))
            })
            .collect::<Result<Vec<(ast::Type, String)>, ExtractError>>()?;
        let ast_body = body
            .iter()
            .map(|stmt| stmt.extract())
            .collect::<Result<Vec<ast::Stmt<ast::classical::Expr>>, ExtractError>>()?;

        Ok(ast::FunctionDef {
            name: name.to_string(),
            args: ast_args,
            ret_type: ret_type.extract()?,
            body: ast_body,
            is_rev: *is_rev,
            dbg: dbg.clone(),
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum MetaFunc {
    /// A `@qpu` kernel.
    Qpu(MetaFunctionDef<qpu::MetaStmt>),
    /// A `@classical` function.
    Classical(MetaFunctionDef<classical::MetaStmt>),
}

impl MetaFunc {
    /// Extract a plain [`ast::Func`] from this MetaQwerty [`MetaFunc`] or
    /// return an error if e.g. contained dimension variable expressions are
    /// not fully folded yet.
    pub fn extract(&self) -> Result<ast::Func, ExtractError> {
        Ok(match self {
            MetaFunc::Qpu(qpu_func) => ast::Func::Qpu(qpu_func.extract()?),
            MetaFunc::Classical(classical_func) => ast::Func::Classical(classical_func.extract()?),
        })
    }
}

/// The top-level node in a Qwerty program that holds all function defintiions.
///
/// In the current implementation, there is only one of these per Python
/// interpreter.
#[derive(Debug, Clone, PartialEq)]
pub struct MetaProgram {
    pub funcs: Vec<MetaFunc>,
    pub dbg: Option<DebugLoc>,
}

impl MetaProgram {
    /// Extract a plain Qwerty AST from a metaQwerty AST.
    pub fn extract(&self) -> Result<ast::Program, ExtractError> {
        let MetaProgram { funcs, dbg } = self;

        let ast_funcs = funcs
            .iter()
            .map(MetaFunc::extract)
            .collect::<Result<Vec<ast::Func>, ExtractError>>()?;

        Ok(ast::Program {
            funcs: ast_funcs,
            dbg: dbg.clone(),
        })
    }
}
