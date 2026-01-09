use crate::syn_util::paths;
use proc_macro::TokenStream;
use quote::quote_spanned;
use syn::{Arm, Block, Error, Expr, ExprBlock, Macro, Stmt, StmtMacro, Token, spanned::Spanned};

mod parse;

enum Chunk {
    Visit(Expr),
    Hook(Option<usize>, Vec<Stmt>),
}

pub fn impl_visitor_match(args: TokenStream) -> Result<TokenStream, Error> {
    let parse::VisitorMatchCall { ty, match_on, arms } = syn::parse(args)?;
    let span = ty.span();

    let mut hook_counter = 0;
    let mut ent_arms = Vec::new();

    for arm in arms {
        let Arm {
            attrs,
            pat,
            guard,
            fat_arrow_token: _,
            body,
            comma: _,
        } = arm;

        if !attrs.is_empty() {
            let first_attr = attrs
                .into_iter()
                .next()
                .expect("attrs is nonempty yet empty");
            return Err(Error::new_spanned(
                first_attr,
                "Attributes not allowed on match arms",
            ));
        }

        if let Some((if_tok, _)) = guard {
            return Err(Error::new_spanned(
                if_tok,
                "Guards not allowed on match arms",
            ));
        }

        let mut chunks = Vec::new();
        let mut pending_stmts = Vec::new();

        match *body {
            Expr::Block(ExprBlock {
                block: Block { stmts, .. },
                ..
            }) => {
                let mut seen_visit_already = false;
                for stmt in stmts {
                    match stmt {
                        Stmt::Macro(StmtMacro {
                            mac: Macro { path, tokens, .. },
                            ..
                        }) if paths::path_is_ident_str(&path, "visit") => {
                            if !pending_stmts.is_empty() {
                                // Don't increment the hook counter for the
                                // first hook because it will just be in the
                                // `Node` arm.
                                let counter = if seen_visit_already {
                                    let counter = Some(hook_counter);
                                    hook_counter += 1;
                                    counter
                                } else {
                                    None
                                };
                                chunks.push(Chunk::Hook(counter, pending_stmts));
                                pending_stmts = Vec::new();
                            }
                            seen_visit_already = true;

                            let arg: Expr = syn::parse2(tokens)?;
                            chunks.push(Chunk::Visit(arg));
                        }

                        other_stmt => {
                            pending_stmts.push(other_stmt);
                        }
                    }
                }

                if !pending_stmts.is_empty() {
                    let counter = if seen_visit_already {
                        let counter = Some(hook_counter);
                        hook_counter += 1;
                        counter
                    } else {
                        None
                    };
                    chunks.push(Chunk::Hook(counter, pending_stmts));
                }
            }

            other_expr => {
                chunks.push(Chunk::Hook(
                    None,
                    vec![Stmt::Expr(other_expr, Some(Token![;](span)))],
                ));
            }
        }

        let ent_arm = if chunks.is_empty() {
            quote_spanned! {span=>
                StackEntry::Node(#pat) => {}
            }
        } else {
            // So we can treat it as a stack
            chunks.reverse();

            let first_chunk = chunks.pop().expect("chunks is nonempty yet empty");
            let (leading_hook_stmts, extra_first_chunk) =
                if let Chunk::Hook(None, stmts) = first_chunk {
                    (stmts, None)
                } else {
                    (Vec::new(), Some(first_chunk))
                };

            let (push_stmts, mut hook_arms): (Vec<_>, Vec<_>) = extra_first_chunk
                .into_iter()
                .chain(chunks.into_iter())
                .map(|chunk| {
                    match chunk {
                        Chunk::Visit(expr) => {
                            (
                                quote_spanned! {span=>
                                    stack.push(StackEntry::Node(&#expr));
                                },
                                quote_spanned! {span=>
                                    // No hook arm needed
                                },
                            )
                        }

                        Chunk::Hook(Some(hook_id), stmts) => (
                            quote_spanned! {span=>
                                stack.push(StackEntry::Hook(#hook_id, node));
                            },
                            quote_spanned! {span=>
                                #[allow(unused)]
                                StackEntry::Hook(#hook_id, #pat) => {
                                    #(#stmts)*
                                }
                            },
                        ),

                        Chunk::Hook(None, _) => unreachable!("Only first hook should have no id"),
                    }
                })
                .unzip();
            // For readability of generated code
            hook_arms.reverse();

            quote_spanned! {span=>
                #[allow(unused)]
                StackEntry::Node(node @ (#pat)) => {
                    #(#leading_hook_stmts)*
                    #(#push_stmts)*
                }
                #(#hook_arms)*
            }
        };
        ent_arms.push(ent_arm);
    }

    Ok(quote_spanned! {span=>
        {
            let match_on = #match_on;

            enum StackEntry<'a> {
                Node(&'a #ty),
                Hook(usize, &'a #ty),
            }

            let mut stack = vec![StackEntry::Node(match_on)];

            while let Some(ent) = stack.pop() {
                let () = match ent {
                    #(#ent_arms)*
                    StackEntry::Hook(_, _) => unreachable!("Invalid hook stack entry"),
                };
            }
        }
    }
    .into())
}
