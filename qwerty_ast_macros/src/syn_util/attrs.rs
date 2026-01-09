//! Helpers for manipulating Rust attributes, e.g.,
//! `#[gen_rebuild::skip_recurse]`.

use crate::syn_util::paths;
use proc_macro2::TokenStream as TokenStream2;
use syn::{
    AttrStyle, Attribute, Field, Fields, FieldsNamed, ItemEnum, ItemStruct, Meta, MetaList, Path,
    Variant, Visibility, punctuated::Pair,
};

/// Returns `Some((P, args))` if `attr` is `#[P(args)]`.
pub fn attr_as_path<'a>(attr: &'a Attribute) -> Option<(&'a Path, Option<&'a TokenStream2>)> {
    if let Attribute {
        style: AttrStyle::Outer,
        meta,
        ..
    } = attr
    {
        match meta {
            Meta::Path(path) => Some((path, None)),

            Meta::List(MetaList {
                path,
                delimiter: _,
                tokens,
            }) => Some((path, Some(tokens))),

            Meta::NameValue(_) => None,
        }
    } else {
        None
    }
}

/// Removes `#[gen_rebuild::*]` attributes from a list of attributes.
fn filter_out_our_attrs(attrs: Vec<Attribute>) -> Vec<Attribute> {
    attrs
        .into_iter()
        .filter(|attr| {
            !attr_as_path(attr)
                .is_some_and(|(path, _arg)| paths::path_as_starting_with_our_prefix(path).is_some())
        })
        .collect()
}

/// Removes `#[gen_rebuild::*]` attrs and visibility levels from a struct
/// field.
pub fn strip_our_attrs_and_vis_from_field(field: Field) -> Field {
    let Field {
        attrs,
        vis: _,
        mutability,
        ident,
        colon_token,
        ty,
    } = field;
    let attrs = filter_out_our_attrs(attrs);
    let vis = Visibility::Inherited;
    Field {
        attrs,
        vis,
        mutability,
        ident,
        colon_token,
        ty,
    }
}

/// Removes `#[gen_rebuild::*]` attrs from a struct field.
fn strip_our_attrs_from_field(field: Field) -> Field {
    let Field {
        attrs,
        vis,
        mutability,
        ident,
        colon_token,
        ty,
    } = field;
    let attrs = filter_out_our_attrs(attrs);
    Field {
        attrs,
        vis,
        mutability,
        ident,
        colon_token,
        ty,
    }
}

/// Removes `#[gen_rebuild::*]` attrs from fields of a struct.
fn strip_our_attrs_from_fields(fields: Fields) -> Fields {
    match fields {
        Fields::Named(FieldsNamed { brace_token, named }) => {
            let named = named
                .into_pairs()
                .map(|pair| match pair {
                    Pair::Punctuated(field, tok) => {
                        Pair::Punctuated(strip_our_attrs_from_field(field), tok)
                    }
                    Pair::End(field) => Pair::End(strip_our_attrs_from_field(field)),
                })
                .collect();
            Fields::Named(FieldsNamed { brace_token, named })
        }
        other => other,
    }
}

/// Removes `#[gen_rebuild::*]` attrs from an enum variant.
fn strip_our_attrs_from_variant(variant: Variant) -> Variant {
    let Variant {
        attrs,
        ident,
        fields,
        discriminant,
    } = variant;
    let fields = strip_our_attrs_from_fields(fields);
    Variant {
        attrs,
        ident,
        fields,
        discriminant,
    }
}

/// Removes `#[gen_rebuild::*]` attrs from an enum.
pub fn strip_our_attrs_from_enum(enum_item: ItemEnum) -> ItemEnum {
    let ItemEnum {
        attrs,
        vis,
        enum_token,
        ident,
        generics,
        brace_token,
        variants,
    } = enum_item;

    let variants = variants
        .into_pairs()
        .map(|pair| match pair {
            Pair::Punctuated(variant, tok) => {
                Pair::Punctuated(strip_our_attrs_from_variant(variant), tok)
            }
            Pair::End(variant) => Pair::End(strip_our_attrs_from_variant(variant)),
        })
        .collect();

    ItemEnum {
        attrs,
        vis,
        enum_token,
        ident,
        generics,
        brace_token,
        variants,
    }
}

/// Removes `#[gen_rebuild::*]` attrs from a struct.
pub fn strip_our_attrs_from_struct(struct_item: ItemStruct) -> ItemStruct {
    let ItemStruct {
        attrs,
        vis,
        struct_token,
        ident,
        generics,
        fields,
        semi_token,
    } = struct_item;

    let fields = strip_our_attrs_from_fields(fields);

    ItemStruct {
        attrs,
        vis,
        struct_token,
        ident,
        generics,
        fields,
        semi_token,
    }
}
