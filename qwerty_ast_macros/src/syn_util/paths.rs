//! Helpers for manipulating paths, e.g., `meta::qpu::MetaExpr`.

use convert_case::{Case, Casing};
use syn::{
    Ident, Pat, PatIdent, PatOr, PatParen, PatPath, PatStruct, PatTupleStruct, Path, PathArguments,
    PathSegment,
    punctuated::{Pair, Punctuated},
    spanned::Spanned,
};

/// Returns `T` if a path is `gen_rebuild::T`.
pub fn path_as_starting_with_our_prefix<'a>(path: &'a Path) -> Option<&'a Ident> {
    let Path { segments, .. } = path;
    let segs: Vec<_> = segments.iter().take(2).collect();
    match &segs[..] {
        [
            PathSegment {
                ident: first_ident,
                arguments: PathArguments::None,
            },
            PathSegment {
                ident: second_ident,
                arguments: PathArguments::None,
            },
        ] if first_ident.to_string() == "gen_rebuild" => Some(second_ident),
        _ => None,
    }
}

/// Returns `Some(T)` if a `Path` is some identifier `T` (instead of being e.g.
/// `A::B::T`).
pub fn path_as_ident<'a>(path: &'a Path) -> Option<&'a Ident> {
    let Path { segments, .. } = path;
    if segments.len() == 1 {
        let first_seg = segments
            .first()
            .expect("path with 1 segment should have a first segment");
        if let PathSegment {
            ident,
            arguments: PathArguments::None,
        } = first_seg
        {
            Some(ident)
        } else {
            None
        }
    } else {
        None
    }
}

pub fn path_as_ident_string<'a>(path: &'a Path) -> Option<String> {
    path_as_ident(path).map(|ident| ident.to_string())
}

/// Rewrite `MetaExpr` as `meta_expr`
pub fn ident_to_snake_case(ident: &Ident) -> Ident {
    let span = ident.span();
    let snake_case_name = ident.to_string().to_case(Case::Snake);
    Ident::new(&snake_case_name, span)
}

/// Rewrite `MetaExpr<T>` as `meta_expr<T>`
fn path_seg_to_snake_case(path_seg: PathSegment) -> PathSegment {
    let PathSegment { ident, arguments } = path_seg;
    let new_ident = ident_to_snake_case(&ident);
    PathSegment {
        ident: new_ident,
        arguments,
    }
}

/// Rewrite `qpu::MetaExpr` as `qpu::meta_expr`
pub fn ty_name_to_snake_case(path: Path) -> Path {
    let Path {
        leading_colon,
        segments,
    } = path;
    let mut seg_vec: Vec<_> = segments.into_pairs().collect();
    let final_seg = seg_vec.pop().expect("Path cannot be empty");
    let rewritten_final_seg = match final_seg {
        Pair::Punctuated(path_seg, punct) => {
            Pair::Punctuated(path_seg_to_snake_case(path_seg), punct)
        }
        Pair::End(path_seg) => Pair::End(path_seg_to_snake_case(path_seg)),
    };
    seg_vec.push(rewritten_final_seg);
    Path {
        leading_colon,
        segments: Punctuated::from_iter(seg_vec.into_iter()),
    }
}

/// Concatenate two paths `A::B` and `C::D` to get `A::B::C::D`. If `rhs`
/// begins with a double colon `::`, it is returned unchanged.
fn concat_paths(mut lhs: Path, rhs: Path) -> Path {
    if rhs.leading_colon.is_none() {
        lhs.segments.extend(rhs.segments);
        lhs
    } else {
        rhs
    }
}

/// Prepends `prefix` to every path in the pattern `Pat`.
pub fn insert_pattern_prefix(pat: Pat, prefix: &Path) -> Pat {
    match pat {
        Pat::Ident(PatIdent {
            attrs,
            by_ref,
            mutability,
            ident,
            subpat,
        }) => {
            if let Some((at_tok, subpat)) = subpat {
                let subpat = Some((at_tok, Box::new(insert_pattern_prefix(*subpat, prefix))));
                Pat::Ident(PatIdent {
                    attrs,
                    by_ref,
                    mutability,
                    ident,
                    subpat,
                })
            } else if ident.to_string().is_case(Case::Pascal)
                && by_ref.is_none()
                && mutability.is_none()
            {
                // EXTREME hack: This is to deal with a syntactic ambiguity for
                // unit variants. If a programmer writes an arm like
                //
                //     UnitType => ast::Type::UnitType,
                //
                // then rustc throws an error because it's unclear if this is
                // meant as a PatPath or a PatIdent. It is parsed as the
                // latter, though. So if the ident in the PatIdent is in
                // PascalCase, then we can convert it into a PatPath. This is
                // some bona fide redneck engineering.
                let path = Path {
                    leading_colon: None,
                    segments: std::iter::once(PathSegment {
                        ident,
                        arguments: PathArguments::None,
                    })
                    .collect(),
                };
                let path = concat_paths(prefix.clone(), path);
                Pat::Path(PatPath {
                    attrs,
                    qself: None,
                    path,
                })
            } else {
                Pat::Ident(PatIdent {
                    attrs,
                    by_ref,
                    mutability,
                    ident,
                    subpat: None,
                })
            }
        }

        Pat::Or(PatOr {
            attrs,
            leading_vert,
            cases,
        }) => {
            let cases = cases
                .into_pairs()
                .map(|pair| match pair {
                    Pair::Punctuated(subpat, tok) => {
                        let subpat = insert_pattern_prefix(subpat, prefix);
                        Pair::Punctuated(subpat, tok)
                    }
                    Pair::End(subpat) => {
                        let subpat = insert_pattern_prefix(subpat, prefix);
                        Pair::End(subpat)
                    }
                })
                .collect();
            Pat::Or(PatOr {
                attrs,
                leading_vert,
                cases,
            })
        }

        Pat::Paren(PatParen {
            attrs,
            paren_token,
            pat,
        }) => {
            let pat = Box::new(insert_pattern_prefix(*pat, prefix));
            Pat::Paren(PatParen {
                attrs,
                paren_token,
                pat,
            })
        }

        Pat::Path(PatPath { attrs, qself, path }) => {
            let path = if qself.is_none() {
                concat_paths(prefix.clone(), path)
            } else {
                path
            };
            Pat::Path(PatPath { attrs, qself, path })
        }

        Pat::Struct(PatStruct {
            attrs,
            qself,
            path,
            brace_token,
            fields,
            rest,
        }) => {
            let path = if qself.is_none() {
                concat_paths(prefix.clone(), path)
            } else {
                path
            };
            Pat::Struct(PatStruct {
                attrs,
                qself,
                path,
                brace_token,
                fields,
                rest,
            })
        }

        Pat::TupleStruct(PatTupleStruct {
            attrs,
            qself,
            path,
            paren_token,
            elems,
        }) => {
            let path = if qself.is_none() {
                concat_paths(prefix.clone(), path)
            } else {
                path
            };
            Pat::TupleStruct(PatTupleStruct {
                attrs,
                qself,
                path,
                paren_token,
                elems,
            })
        }

        other => other,
    }
}

pub fn rewritten_enum_path(ty: Path, config_name: Ident) -> Path {
    let span = ty.span();
    let mut ty_as_namespace = ty_name_to_snake_case(ty);

    let config_seg = PathSegment {
        ident: config_name,
        arguments: PathArguments::None,
    };
    ty_as_namespace.segments.push(config_seg);

    let enum_ident = Ident::new("Rewritten", span);
    let enum_seg = PathSegment {
        ident: enum_ident,
        arguments: PathArguments::None,
    };
    ty_as_namespace.segments.push(enum_seg);

    ty_as_namespace
}
