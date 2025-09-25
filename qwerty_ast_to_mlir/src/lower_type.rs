use crate::{ctx::MLIR_CTX, lower_prog_stmt::Lowerable};
use melior::{
    dialect::qwerty,
    ir::{self, Location, r#type::FunctionType},
};
use qwerty_ast::{
    ast::{self, Canonicalizable, FunctionDef, RegKind, Trivializable},
    dbg::DebugLoc,
};

/// Converts an AST debug location to an mlir::Location.
pub fn dbg_to_loc(dbg: Option<DebugLoc>) -> Location<'static> {
    dbg.map_or_else(
        || Location::unknown(&MLIR_CTX),
        |dbg| Location::new(&MLIR_CTX, &dbg.file, dbg.line, dbg.col),
    )
}

/// Converts AST types to mlir::Types. Returns a vec to account for tuples,
/// which will be represented by multiple MLIR values.
pub fn ast_ty_to_mlir_tys<E: Lowerable>(ty: &ast::Type) -> Vec<ir::Type<'static>> {
    match ty {
        ast::Type::FuncType { in_ty, out_ty } => {
            vec![
                qwerty::FunctionType::new(
                    &MLIR_CTX,
                    FunctionType::new(
                        &MLIR_CTX,
                        &ast_ty_to_mlir_tys::<E>(&**in_ty),
                        &ast_ty_to_mlir_tys::<E>(&**out_ty),
                    ),
                    /*reversible=*/ false,
                )
                .into(),
            ]
        }

        ast::Type::RevFuncType { in_out_ty } => {
            let in_out_mlir_tys = ast_ty_to_mlir_tys::<E>(&**in_out_ty);
            vec![
                qwerty::FunctionType::new(
                    &MLIR_CTX,
                    FunctionType::new(&MLIR_CTX, &in_out_mlir_tys, &in_out_mlir_tys),
                    /*reversible=*/ true,
                )
                .into(),
            ]
        }

        ast::Type::RegType {
            elem_ty: RegKind::Bit,
            dim,
        } if *dim > 0 => {
            vec![E::bit_register_ty(*dim)]
        }

        ast::Type::RegType {
            elem_ty: RegKind::Qubit,
            dim,
        } if *dim > 0 => {
            vec![qwerty::QBundleType::new(&MLIR_CTX, (*dim).try_into().unwrap()).into()]
        }

        ast::Type::RegType {
            elem_ty: RegKind::Basis,
            ..
        } => panic!("Basis has no MLIR type"),

        ast::Type::TupleType { tys } => tys.iter().flat_map(ast_ty_to_mlir_tys::<E>).collect(),

        // Fallthrough for RegType where *dim == 0
        ast::Type::UnitType | ast::Type::RegType { .. } => vec![],
    }
}

/// Returns the type of a FunctionDef AST node as an mlir::Type.
pub fn ast_func_mlir_ty<E>(func_def: &FunctionDef<E>) -> ir::Type<'static>
where
    E: Lowerable + Trivializable + Canonicalizable + Clone,
{
    let mlir_tys = ast_ty_to_mlir_tys::<E>(&func_def.get_type());
    assert_eq!(mlir_tys.len(), 1);
    mlir_tys[0]
}
