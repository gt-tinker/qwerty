use crate::syn_util::paths;
use proc_macro::TokenStream;
use proc_macro2::{Span, TokenStream as TokenStream2};
use quote::quote_spanned;
use syn::{
    Arm, Block, Error, Expr, ExprBlock, Macro, Pat, Stmt, StmtMacro, Token, spanned::Spanned,
};

mod parse;

enum Chunk {
    Visit(Expr),
    Hook(Vec<Stmt>),
}

fn block_into_chunks(block: Block) -> Result<Vec<Chunk>, Error> {
    let Block { stmts, .. } = block;

    let mut chunks = Vec::new();
    let mut pending_stmts = Vec::new();

    for stmt in stmts {
        match stmt {
            Stmt::Macro(StmtMacro {
                mac: Macro { path, tokens, .. },
                ..
            }) if paths::path_is_ident_str(&path, "visit") => {
                if !pending_stmts.is_empty() {
                    chunks.push(Chunk::Hook(pending_stmts));
                    pending_stmts = Vec::new();
                }

                let arg: Expr = syn::parse2(tokens)?;
                chunks.push(Chunk::Visit(arg));
            }

            other_stmt => {
                pending_stmts.push(other_stmt);
            }
        }
    }

    if !pending_stmts.is_empty() {
        chunks.push(Chunk::Hook(pending_stmts));
    }

    Ok(chunks)
}

fn expr_into_chunks(span: Span, expr: Expr) -> Vec<Chunk> {
    vec![Chunk::Hook(vec![Stmt::Expr(expr, Some(Token![;](span)))])]
}

fn chunks_into_ent_arm(pat: &Pat, span: Span, mut chunks: Vec<Chunk>) -> TokenStream2 {
    if chunks.is_empty() {
        quote_spanned! {span=>
            StackEntry::Node(#pat) => {}
        }
    } else {
        // So we can treat it as a stack
        chunks.reverse();

        let first_chunk = chunks.pop().expect("chunks is nonempty yet empty");
        let (leading_hook_stmts, extra_first_chunk) = if let Chunk::Hook(stmts) = first_chunk {
            (stmts, None)
        } else {
            (Vec::new(), Some(first_chunk))
        };

        let mut hook_counter = 0usize;
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

                    Chunk::Hook(stmts) => {
                        let hook_id = hook_counter;
                        hook_counter += 1;

                        (
                            quote_spanned! {span=>
                                stack.push(StackEntry::Hook(#hook_id, node));
                            },
                            quote_spanned! {span=>
                                #[allow(unused)]
                                StackEntry::Hook(#hook_id, #pat) => {
                                    #(#stmts)*
                                }
                            },
                        )
                    }
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
    }
}

pub fn impl_visitor_match(args: TokenStream) -> Result<TokenStream, Error> {
    let parse::VisitorMatchCall { ty, match_on, arms } = syn::parse(args)?;
    let span = ty.span();

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

        let chunks = match *body {
            Expr::Block(ExprBlock { block, .. }) => block_into_chunks(block)?,
            other_expr => expr_into_chunks(span, other_expr),
        };

        ent_arms.push(chunks_into_ent_arm(&pat, span, chunks));
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
