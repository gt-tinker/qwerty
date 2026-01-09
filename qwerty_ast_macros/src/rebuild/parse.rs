//! Code for parsing macro/attribute arguments using the [`syn::parse::Parse`]
//! trait and storing the result in structs.

use crate::syn_util::{attrs, paths};
use proc_macro2::{Delimiter, Span};
use syn::{
    Arm, Attribute, Error, Expr, GenericParam, Ident, Item, ItemEnum, ItemStruct, Pat, PatIdent,
    PatType, Path, Token, Type, WherePredicate, braced, parenthesized,
    parse::{Parse, ParseStream},
    spanned::Spanned,
};

/// Represents a mapping from one type to another in `rewrite_to` such as
/// `DimExpr => usize`.
pub struct TypeMapping(pub Type, pub Type);

impl Parse for TypeMapping {
    fn parse(input: ParseStream) -> Result<Self, Error> {
        let lhs = input.parse()?;
        input.parse::<Token![=>]>()?;
        let rhs = input.parse()?;
        Ok(TypeMapping(lhs, rhs))
    }
}

/// Represents a configuration for `gen_rebuild` such as `expand` in
/// `#[gen_rebuild { expand(...) }]`.
pub struct RebuildConfig {
    pub name: Ident,
    pub option: bool,
    pub result_err: Option<Type>,
    pub progress: Option<Type>,
    pub rewrite: Option<Ident>,
    pub rewrite_to: Vec<TypeMapping>,
    pub recurse_attrs: bool,
    pub more_generic_params: Vec<GenericParam>,
    pub more_moved_args: Vec<(Ident, Type)>,
    pub more_copied_args: Vec<(Ident, Type)>,
    pub more_where: Vec<WherePredicate>,
}

/// Parse a `gen_rebuild` option for a config given a provided argument type
/// and the expected number of arguments.
macro_rules! parse_rebuild_option {
    (no_args: $option_name:ident, $opt_name:ident) => {
        if $opt_name.is_some() {
            return Err(Error::new_spanned($option_name, concat!("Duplicated option ", stringify!($opt_name))));
        }
        $opt_name = Some(true);
    };
    (one_arg: $option_name:ident, $opt_name:ident, $args:ident, $arg_ast_node_ty:ident) => {
        if $opt_name.is_some() {
            return Err(Error::new_spanned($option_name, concat!("Duplicated option ", stringify!($opt_name))));
        }
        let arg;
        parenthesized!(arg in $args);
        let arg_ast_node: $arg_ast_node_ty = arg.parse()?;
        $opt_name = Some(arg_ast_node);
    };
    (n_args: $option_name:ident, $opt_name:ident, $args:ident, $arg_ast_node_ty:ident) => {
        if $opt_name.is_some() {
            return Err(Error::new_spanned($option_name, concat!("Duplicated option ", stringify!($opt_name))));
        }
        let args_buf;
        parenthesized!(args_buf in $args);
        let punct = args_buf.parse_terminated(|stream| {
            stream.parse::<$arg_ast_node_ty>()
        }, Token![,])?;
        let ast_nodes = punct.into_iter().collect();
        $opt_name = Some(ast_nodes);
    };
}

/// Parse all all options for a given [`RebuildConfig`]. Avoids code
/// duplication in parsing logic.
macro_rules! parse_rebuild_options {
    (($option_name:ident, $args:ident), {
        boolean: [$($bool_opt_name:ident),*],
        ident: [$($ident_opt_name:ident),*],
        ty: [$($ty_opt_name:ident),*],
        ty_mappings: [$($ty_mappings_opt_name:ident),*],
        generic_params: [$($generic_param_opt_name:ident),*],
        pat_tys: [$($pat_ty_opt_name:ident),*],
        where_preds: [$($where_pred_opt_name:ident),*]
    $(,)?}) => {
        match &$option_name.to_string()[..] {
            $(
                stringify!($bool_opt_name) => {
                    parse_rebuild_option!(no_args: $option_name, $bool_opt_name);
                }
            )*
            $(
                stringify!($ident_opt_name) => {
                    parse_rebuild_option!(one_arg: $option_name, $ident_opt_name, $args, Ident);
                }
            )*
            $(
                stringify!($ty_opt_name) => {
                    parse_rebuild_option!(one_arg: $option_name, $ty_opt_name, $args, Type);
                }
            )*
            $(
                stringify!($ty_mappings_opt_name) => {
                    parse_rebuild_option!(n_args: $option_name, $ty_mappings_opt_name, $args, TypeMapping);
                }
            )*
            $(
                stringify!($generic_param_opt_name) => {
                    parse_rebuild_option!(n_args: $option_name, $generic_param_opt_name, $args, GenericParam);
                }
            )*
            $(
                stringify!($pat_ty_opt_name) => {
                    parse_rebuild_option!(n_args: $option_name, $pat_ty_opt_name, $args, PatType);
                }
            )*
            $(
                stringify!($where_pred_opt_name) => {
                    parse_rebuild_option!(n_args: $option_name, $where_pred_opt_name, $args, WherePredicate);
                }
            )*
            opt => {
                return Err(Error::new_spanned($option_name, format!("Unknown rebuild config option {}", opt)));
            }
        }
    };
}

/// Returns true if the parse stream contains a parenthesis next.
fn peek_parens(input: &ParseStream) -> bool {
    input.cursor().group(Delimiter::Parenthesis).is_some()
}

/// Parses a `arg: T` node into a pair of (arg, T) or throws an error if this is not possible.
fn validate_arg(pat_ty: PatType) -> Result<(Ident, Type), Error> {
    let span = pat_ty.span();
    let PatType {
        attrs,
        pat,
        colon_token: _,
        ty,
    } = pat_ty;
    if !attrs.is_empty() {
        return Err(Error::new(
            span,
            "Attributes not allowed on extra args".to_string(),
        ));
    }
    if let Pat::Ident(PatIdent {
        attrs,
        by_ref,
        mutability,
        ident,
        subpat,
    }) = *pat
    {
        if !attrs.is_empty() {
            return Err(Error::new(
                span,
                "Attributes not allowed in extra args patterns".to_string(),
            ));
        }
        if by_ref.is_some() {
            return Err(Error::new(
                span,
                "ref not allowed in extra args".to_string(),
            ));
        }
        if mutability.is_some() {
            return Err(Error::new(
                span,
                "mut not allowed in extra args".to_string(),
            ));
        }
        if subpat.is_some() {
            return Err(Error::new(span, "@ not allowed in extra args".to_string()));
        }
        return Ok((ident, *ty));
    } else {
        return Err(Error::new(
            span,
            "Arguments cannot be nontrivial patterns".to_string(),
        ));
    }
}

/// Common validation and transformation code for `more_moved_args` and
/// `more_copied_args`
fn tidy_up_extra_args(args: Option<Vec<PatType>>) -> Result<Vec<(Ident, Type)>, Error> {
    Ok(args
        .map(|args: Vec<PatType>| args.into_iter().map(validate_arg).collect())
        .transpose()?
        .unwrap_or_else(Vec::new))
}

impl Parse for RebuildConfig {
    fn parse(input: ParseStream) -> Result<Self, Error> {
        let name: Ident = input.parse()?;

        let mut option = None;
        let mut result_err = None;
        let mut progress = None;
        let mut rewrite = None;
        let mut rewrite_to = None;
        let mut recurse_attrs = None;
        let mut more_generic_params = None;
        let mut more_moved_args = None;
        let mut more_copied_args = None;
        let mut more_where = None;

        if peek_parens(&input) {
            let args;
            parenthesized!(args in input);
            // Parse arguments for this option. Based on the source code for
            // [`syn::Punctuated::parse_terminated_with`].
            loop {
                if args.is_empty() {
                    break;
                }
                let opt_name: Ident = args.parse()?;
                parse_rebuild_options!((opt_name, args), {
                    boolean: [option, recurse_attrs],
                    ident: [rewrite],
                    ty: [progress, result_err],
                    ty_mappings: [rewrite_to],
                    generic_params: [more_generic_params],
                    pat_tys: [more_moved_args, more_copied_args],
                    where_preds: [more_where],
                });
                if args.is_empty() {
                    break;
                }
                args.parse::<Token![,]>()?;
            }
        }

        if rewrite_to.is_some() && rewrite.is_none() {
            Err(Error::new(name.span(), "rewrite_to requries rewrite"))
        } else if result_err.is_some() && option.is_some() {
            Err(Error::new(
                name.span(),
                "Cannot use both result_err and option",
            ))
        } else {
            let option = option.unwrap_or(false);
            let rewrite_to = rewrite_to.unwrap_or_else(Vec::new);
            let recurse_attrs = recurse_attrs.unwrap_or(false);
            let more_generic_params = more_generic_params.unwrap_or_else(Vec::new);
            let more_moved_args = tidy_up_extra_args(more_moved_args)?;
            let more_copied_args = tidy_up_extra_args(more_copied_args)?;
            let more_where = more_where.unwrap_or_else(Vec::new);

            Ok(RebuildConfig {
                name,
                option,
                result_err,
                progress,
                rewrite,
                rewrite_to,
                recurse_attrs,
                more_generic_params,
                more_moved_args,
                more_copied_args,
                more_where,
            })
        }
    }
}

/// Holds all the rebuild configurations. That is, this holds all the parsed
/// arguments to `gen_rebuild`.
pub struct RebuildConfigs {
    pub configs: Vec<RebuildConfig>,
}

impl Parse for RebuildConfigs {
    fn parse(input: ParseStream) -> Result<Self, Error> {
        let punctuated = input.parse_terminated(RebuildConfig::parse, Token![,])?;
        let configs = punctuated.into_iter().collect();
        Ok(RebuildConfigs { configs })
    }
}

mod kw {
    syn::custom_keyword!(configs);
    syn::custom_keyword!(defs);
}

pub struct RebuildStructsArgs {
    pub configs: Vec<RebuildConfig>,
    pub variant_structs: Vec<ItemStruct>,
    pub the_enum: ItemEnum,
}

impl Parse for RebuildStructsArgs {
    fn parse(input: ParseStream) -> Result<Self, Error> {
        // Parse the configurations:
        //     configs { trivial, expand(...), ... }
        input.parse::<kw::configs>()?;
        let all_configs;
        braced!(all_configs in input);
        let punctuated = all_configs.parse_terminated(RebuildConfig::parse, Token![,])?;
        let configs = punctuated.into_iter().collect();

        // Parse the variant structs & enums:
        //     defs { struct Variable { ... } ... enum Expr { ... } }
        let defs_kw = input.parse::<kw::defs>()?;
        let all_defs;
        braced!(all_defs in input);
        let mut the_enum = None;
        let mut variant_structs = Vec::new();
        while !all_defs.is_empty() {
            //variant_structs.push(all_variant_structs.parse::<ItemStruct>()?);
            let item = all_defs.parse::<Item>()?;

            match item {
                Item::Enum(enum_item) => {
                    if the_enum.is_some() {
                        return Err(Error::new_spanned(
                            enum_item,
                            "Only one enum is allowed in defs {...}",
                        ));
                    } else {
                        the_enum = Some(enum_item);
                    }
                }

                Item::Struct(struct_item) => {
                    variant_structs.push(struct_item);
                }

                other_item => {
                    return Err(Error::new_spanned(
                        other_item,
                        "Only enums and structs are allowed in defs {...}",
                    ));
                }
            }
        }

        let the_enum =
            the_enum.ok_or_else(|| Error::new_spanned(defs_kw, "Missing enum from defs {...}"))?;

        Ok(RebuildStructsArgs {
            configs,
            variant_structs,
            the_enum,
        })
    }
}

/// Holds the parsed list of affected configurations for the
/// `#[gen_rebuild::skip_recurse]` attribute. For more information, see
/// [`crate::gen_rebuild`].
pub struct SkipRecurseAttrArgs {
    pub configs: Vec<Ident>,
}

impl Parse for SkipRecurseAttrArgs {
    fn parse(input: ParseStream) -> Result<Self, Error> {
        let punct = input.parse_terminated(|arg| arg.parse::<Ident>(), Token![,])?;
        let configs = punct.into_iter().collect();
        Ok(SkipRecurseAttrArgs { configs })
    }
}

/// Returns `Ok(true)` if any the path is for a `gen_rebuild::skip_recurse`
/// attribute, `Ok(false)` if not, or `Err` if this is an unknown
/// `gen_rebuild::*` path.
fn validate_attr_path(span: Span, path: &Path) -> Result<bool, Error> {
    if let Some(name) = paths::path_as_starting_with_our_prefix(path) {
        if name.to_string() == "skip_recurse" {
            Ok(true)
        } else {
            Err(Error::new(
                span,
                format!("Unknown gen_rebuild attribute {}", name),
            ))
        }
    } else {
        Ok(false)
    }
}

/// Returns `Ok(true)` if any of the Rust attributes `attrs` are the
/// `gen_rebuild::skip_recurse` attribute, `Ok(false)` if there are no such
/// attributes, or `Err` if there is an unknown `gen_rebuild::*` attribute.
pub fn has_skip_recurse_attr(
    config_ident: &Ident,
    span: Span,
    attrs: &Vec<Attribute>,
) -> Result<bool, Error> {
    Ok(attrs
        .iter()
        .map(|attr| {
            if let Some((path, maybe_arg)) = attrs::attr_as_path(attr) {
                if validate_attr_path(span, path)? {
                    if let Some(arg_stream) = maybe_arg {
                        let args: SkipRecurseAttrArgs = syn::parse2(arg_stream.clone())?;
                        Ok(args.configs.into_iter().any(|cfg| cfg == *config_ident))
                    } else {
                        // No "args" means don't recurse for any config
                        Ok(true)
                    }
                } else {
                    Ok(false)
                }
            } else {
                Ok(false)
            }
        })
        .collect::<Result<Vec<_>, Error>>()?
        .into_iter()
        .any(|yep| yep))
}

/// Holds the parsed arguments for a call to `rebuild!()`.
pub struct RebuildCall {
    pub ty: Path,
    pub self_arg: Expr,
    pub config_name: Ident,
    pub more_args: Vec<Expr>,
}

impl Parse for RebuildCall {
    fn parse(input: ParseStream) -> Result<Self, Error> {
        let ty = input.parse()?;
        input.parse::<Token![,]>()?;
        let self_arg = input.parse()?;
        input.parse::<Token![,]>()?;
        let config_name = input.parse()?;
        let more_args = if input.is_empty() {
            vec![]
        } else {
            input.parse::<Token![,]>()?;
            let punct = input.parse_terminated(|arg| arg.parse::<Expr>(), Token![,])?;
            punct.into_iter().collect()
        };
        Ok(RebuildCall {
            ty,
            self_arg,
            config_name,
            more_args,
        })
    }
}

/// Holds the parsed arguments for a call to `rewrite_ty!()`.
pub struct RewriteTypeCall {
    pub ty: Path,
    pub config_name: Ident,
}

impl Parse for RewriteTypeCall {
    fn parse(input: ParseStream) -> Result<Self, Error> {
        let ty = input.parse()?;
        input.parse::<Token![,]>()?;
        let config_name = input.parse()?;
        Ok(RewriteTypeCall { ty, config_name })
    }
}

/// Holds the parsed arguments for a call to `rewrite_match!{}`.
pub struct RewriteMatchCall {
    pub ty: Path,
    pub config_name: Ident,
    pub expr: Expr,
    pub arms: Vec<Arm>,
}

impl Parse for RewriteMatchCall {
    fn parse(input: ParseStream) -> Result<Self, Error> {
        let ty = input.parse()?;
        input.parse::<Token![,]>()?;
        let config_name = input.parse()?;
        input.parse::<Token![,]>()?;
        let expr = input.parse()?;
        input.parse::<Token![,]>()?;

        let mut arms = Vec::new();
        while !input.is_empty() {
            arms.push(input.parse()?);
        }

        Ok(RewriteMatchCall {
            ty,
            expr,
            config_name,
            arms,
        })
    }
}
