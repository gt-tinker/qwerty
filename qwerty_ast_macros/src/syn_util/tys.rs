//! Helpers for manipulating types.

use crate::syn_util::paths;
use syn::{
    AngleBracketedGenericArguments, GenericArgument, Ident, Path, PathArguments, PathSegment,
    Token, Type, TypePath, TypeTuple, punctuated::Pair, spanned::Spanned, token::Paren,
};

/// Returns `Some(T)` if `ty` is some identifier `T`.
pub fn ty_as_ident<'a>(ty: &'a Type) -> Option<&'a Ident> {
    if let Type::Path(TypePath { qself: None, path }) = ty {
        paths::path_as_ident(path)
    } else {
        None
    }
}

/// Returns `Some((T, S))` if `ty` is `T<S>` where `T` is an identifier and `S`
/// is some type.
pub fn ty_as_of_ty<'a>(ty: &'a Type) -> Option<(&'a Ident, &'a Type)> {
    if let Type::Path(TypePath {
        qself: None,
        path: Path {
            leading_colon: None,
            segments: path_segments,
        },
    }) = ty
        && path_segments.len() == 1
    {
        let first_seg = path_segments
            .first()
            .expect("path with 1 segment should have a first segment");
        if let PathSegment {
            ident: seg_ident,
            arguments:
                PathArguments::AngleBracketed(AngleBracketedGenericArguments {
                    colon2_token: None,
                    args: seg_args,
                    ..
                }),
        } = first_seg
            && seg_args.len() == 1
        {
            let first_arg = seg_args
                .first()
                .expect("generic args of length 1 should have a first generic arg");
            if let GenericArgument::Type(ty) = first_arg {
                Some((seg_ident, ty))
            } else {
                None
            }
        } else {
            None
        }
    } else {
        None
    }
}

/// Returns `Some(T)` if `ty` is a `Vec<T>`.
pub fn ty_as_vec<'a>(ty: &'a Type) -> Option<&'a Type> {
    if let Some((wrapper_ident, wrapped_ty)) = ty_as_of_ty(ty)
        && wrapper_ident.to_string() == "Vec"
    {
        Some(wrapped_ty)
    } else {
        None
    }
}

/// Returns `Some(vec![T1, T2, ...])` if `ty` is a `(T1, T2, ...)`.
pub fn ty_as_tuple<'a>(ty: &'a Type) -> Option<Vec<&'a Type>> {
    if let Type::Tuple(TypeTuple { elems, .. }) = ty {
        Some(elems.iter().collect())
    } else {
        None
    }
}

/// Returns `Some(T)` if `ty` is a `Box<T>`.
pub fn ty_as_box<'a>(ty: &'a Type) -> Option<&'a Type> {
    if let Some((box_ident, boxed_ty)) = ty_as_of_ty(ty)
        && box_ident.to_string() == "Box"
    {
        Some(boxed_ty)
    } else {
        None
    }
}

/// Returns `Some(T)` if `ty` is a `Option<T>`.
pub fn ty_as_option<'a>(ty: &'a Type) -> Option<&'a Type> {
    if let Some((box_ident, boxed_ty)) = ty_as_of_ty(ty)
        && box_ident.to_string() == "Option"
    {
        Some(boxed_ty)
    } else {
        None
    }
}

/// Returns `true` if `ty` is a `Box<boxed_ident>`.
pub fn ty_is_boxed_ty(ty: &Type, boxed_ident: &Ident) -> bool {
    if let Some(boxed_ty) = ty_as_box(ty) {
        if let Some(boxed_ty_ident) = ty_as_ident(boxed_ty) {
            boxed_ty_ident == boxed_ident
        } else {
            false
        }
    } else {
        false
    }
}

/// Returns `true` if `ty` is a `Vec<elem_ident>`
pub fn ty_is_vec_ty(ty: &Type, elem_ident: &Ident) -> bool {
    if let Some(elem_ty) = ty_as_vec(ty) {
        if let Some(elem_ty_ident) = ty_as_ident(elem_ty) {
            elem_ty_ident == elem_ident
        } else {
            false
        }
    } else {
        false
    }
}

/// Convert `T` into `Something<T>`.
fn ty_into_of_ty(ty: Type, of_ty_name: &str) -> Type {
    let span = ty.span();
    Type::Path(TypePath {
        qself: None,
        path: Path {
            leading_colon: None,
            segments: std::iter::once(Pair::End(PathSegment {
                ident: Ident::new(of_ty_name, span),
                arguments: PathArguments::AngleBracketed(AngleBracketedGenericArguments {
                    colon2_token: None,
                    lt_token: Token![<](span),
                    args: std::iter::once(Pair::End(GenericArgument::Type(ty))).collect(),
                    gt_token: Token![>](span),
                }),
            }))
            .collect(),
        },
    })
}

/// Convert `T` into `Option<T>`.
pub fn ty_into_option(ty: Type) -> Type {
    ty_into_of_ty(ty, "Option")
}

/// Convert `T` into `Vec<T>`.
pub fn ty_into_vec(ty: Type) -> Type {
    ty_into_of_ty(ty, "Vec")
}

/// Convert a list of types into `(T1, T2, ..., TN)`.
pub fn tys_into_tuple(tys: Vec<Type>) -> Type {
    Type::Tuple(TypeTuple {
        paren_token: Paren::default(),
        elems: tys.into_iter().collect(),
    })
}

/// Construct a type `T` given an identifier `T`.
pub fn ident_into_ty(ty_ident: Ident) -> Type {
    Type::Path(TypePath {
        qself: None,
        path: Path {
            leading_colon: None,
            segments: std::iter::once(Pair::End(PathSegment {
                ident: ty_ident.clone(),
                arguments: PathArguments::None,
            }))
            .collect(),
        },
    })
}

/// Returns `true` if this type is among the hardcoded list of types to
/// `clone()` rather than recursing into. This is currently things like
/// [`String`] or `DebugLoc` or `Option`s or `Box`es of those types.
pub fn should_skip_attr_ty(ty: &Type) -> bool {
    matches!(
        ty_as_ident(ty),
        Some(ty_ident)
        if matches!(
            &ty_ident.to_string()[..],
            "String" | "DebugLoc" | "UBig" | "f64" | "char" | "usize"
        )
    ) || matches!(
        ty_as_of_ty(ty),
        Some((ty_ident, arg))
        if should_skip_attr_ty(arg)
            && matches!(&ty_ident.to_string()[..], "Option" | "Box" | "Vec")
    )
}

/// Returns `true` if `field_ty` would be considered a child of
/// `enum_ident`.
pub fn is_child_ty(field_ty: &Type, enum_ident: &Ident) -> bool {
    ty_is_boxed_ty(field_ty, enum_ident) || ty_is_vec_ty(field_ty, enum_ident)
}
