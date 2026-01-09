//! Implementations of the rebuild-related macros in this crate.

use crate::syn_util::{attrs, paths};
use proc_macro::TokenStream;
use quote::{ToTokens, quote_spanned};
use syn::{Arm, Error, ItemEnum, spanned::Spanned};

mod gen_rebuild;
mod parse;

/// Implements `#[gen_rebuild { ... }]`. See [`crate::gen_rebuild`].
pub fn impl_gen_rebuild(attr: TokenStream, item: TokenStream) -> Result<TokenStream, Error> {
    let configs: parse::RebuildConfigs = syn::parse(attr)?;
    let enum_item: ItemEnum = syn::parse(item)?;

    let enum_info = gen_rebuild::TheEnum::from_enum_with_internal_structs(&enum_item);
    let namespace_name = paths::ident_to_snake_case(&enum_item.ident);
    let inner_modules = configs
        .configs
        .into_iter()
        .map(|config| gen_rebuild::gen_rebuild_for_config(config, &enum_info))
        .collect::<Result<Vec<_>, _>>()?;

    let span = enum_item.span();
    let outer_module: TokenStream = quote_spanned! {span=>
        pub mod #namespace_name {
            #(#inner_modules)*
        }
    }
    .into();
    let mut result: TokenStream = attrs::strip_our_attrs_from_enum(enum_item)
        .into_token_stream()
        .into();
    result.extend(outer_module);
    Ok(result)
}

/// Implements `gen_rebuild_structs! { ... }`. See [`crate::gen_rebuild_structs`].
pub fn impl_gen_rebuild_structs(args: TokenStream) -> Result<TokenStream, Error> {
    let parse::RebuildStructsArgs {
        configs,
        variant_structs,
        the_enum,
    } = syn::parse(args)?;

    let enum_info =
        gen_rebuild::TheEnum::from_enum_and_external_structs(&the_enum, &variant_structs)?;
    let span = enum_info.span;
    let namespace_name = paths::ident_to_snake_case(&the_enum.ident);
    let inner_modules = configs
        .into_iter()
        .map(|config| gen_rebuild::gen_rebuild_for_config(config, &enum_info))
        .collect::<Result<Vec<_>, _>>()?;

    let outer_module: TokenStream = quote_spanned! {span=>
        pub mod #namespace_name {
            #(#inner_modules)*
        }
    }
    .into();

    let mut result = TokenStream::new();
    for variant_struct in variant_structs {
        result.extend(TokenStream::from(
            attrs::strip_our_attrs_from_struct(variant_struct).into_token_stream(),
        ));
    }
    result.extend(TokenStream::from(the_enum.into_token_stream()));
    result.extend(outer_module);
    Ok(result)
}

/// Implements `rebuild!(...)`. See [`crate::rebuild!`].
pub fn impl_rebuild(args: TokenStream) -> Result<TokenStream, Error> {
    let parse::RebuildCall {
        ty,
        self_arg,
        config_name,
        more_args,
    } = syn::parse(args)?;
    let span = ty.span();
    let ty_as_namespace = paths::ty_name_to_snake_case(ty);
    Ok(quote_spanned! {span=>
        #ty_as_namespace::#config_name::rebuild(#self_arg, #(#more_args),*)
    }
    .into())
}

/// Implements `rewrite_ty!(...)`. See [`crate::rewrite_ty`].
pub fn impl_rewrite_ty(args: TokenStream) -> Result<TokenStream, Error> {
    let parse::RewriteTypeCall { ty, config_name } = syn::parse(args)?;
    Ok(paths::rewritten_enum_path(ty, config_name)
        .into_token_stream()
        .into())
}

/// Implements `rewrite_match!{...}`. See [`crate::rewrite_match`].
pub fn impl_rewrite_match(args: TokenStream) -> Result<TokenStream, Error> {
    let parse::RewriteMatchCall {
        ty,
        config_name,
        expr,
        arms,
    } = syn::parse(args)?;
    let span = ty.span();
    let enum_path = paths::rewritten_enum_path(ty, config_name);

    let arms: Vec<_> = arms
        .into_iter()
        .map(
            |Arm {
                 attrs,
                 pat,
                 guard,
                 fat_arrow_token,
                 body,
                 comma,
             }| {
                let pat = paths::insert_pattern_prefix(pat, &enum_path);
                Arm {
                    attrs,
                    pat,
                    guard,
                    fat_arrow_token,
                    body,
                    comma,
                }
            },
        )
        .collect();

    Ok(quote_spanned! {span=>
        match #expr {
            #(#arms)*
        }
    }
    .into())
}
