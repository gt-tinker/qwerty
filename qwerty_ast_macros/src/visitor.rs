use crate::syn_util::paths;
use proc_macro::TokenStream;
use proc_macro2::{Span, TokenStream as TokenStream2};
use quote::{ToTokens, quote_spanned};
use syn::{
    Arm, Block, Error, Expr, ExprBlock, Macro, Pat, Stmt, StmtMacro, Token, Type, spanned::Spanned,
};

mod parse;

enum Chunk {
    Visit(Expr),
    Hook(Vec<Stmt>),
    ForHook(Type, Pat, Expr, Vec<Chunk>),
}

struct NumChunks {
    num_chunks: Vec<NumChunk>,
    hook_defs: Vec<HookDef>,
}

enum NumChunk {
    Visit(Expr),
    CallHook(usize),
}

struct HookDef {
    hook_id: usize,
    stmts: Vec<TokenStream2>,
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
            }) if matches!(
                paths::path_as_ident_string(&path).as_deref(),
                Some("visit" | "visit_for")
            ) =>
            {
                if !pending_stmts.is_empty() {
                    chunks.push(Chunk::Hook(pending_stmts));
                    pending_stmts = Vec::new();
                }

                let chunk = match paths::path_as_ident_string(&path).as_deref() {
                    Some("visit") => {
                        let parse::VisitMacroCall { expr } = syn::parse2(tokens)?;
                        Chunk::Visit(expr)
                    }
                    Some("visit_for") => {
                        let parse::VisitForMacroCall {
                            pat_ty,
                            pat,
                            expr,
                            body,
                        } = syn::parse2(tokens)?;
                        let chunks = block_into_chunks(body)?;
                        Chunk::ForHook(pat_ty, pat, expr, chunks)
                    }
                    _ => unreachable!("Unknown visit macro"),
                };
                chunks.push(chunk);
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

fn number_chunks(span: Span, hook_counter: &mut usize, chunks: Vec<Chunk>) -> NumChunks {
    let (num_chunks, hook_defs): (Vec<_>, Vec<_>) = chunks
        .into_iter()
        .map(|chunk| match chunk {
            Chunk::Visit(expr) => (NumChunk::Visit(expr), vec![]),
            Chunk::Hook(stmts) => {
                let hook_id = *hook_counter;
                *hook_counter += 1;
                let stmts = stmts.into_iter().map(Stmt::into_token_stream).collect();
                (
                    NumChunk::CallHook(hook_id),
                    vec![HookDef { hook_id, stmts }],
                )
            }

            Chunk::ForHook(pat_ty, pat, expr, chunks) => {
                let NumChunks {
                    num_chunks: nested_num_chunks,
                    mut hook_defs,
                } = number_chunks(span, hook_counter, chunks);

                let push_stmts: Vec<_> = nested_num_chunks
                    .into_iter()
                    .map(|num_chunk| num_chunk_into_push_stmt(span, num_chunk))
                    .collect();

                let stmt = quote_spanned! {span=>
                    stack.extend({
                        let mut stack = Vec::new();
                        for #pat in #expr {
                            #(#push_stmts)*
                        }
                        stack.reverse();
                        stack
                    });
                };
                let hook_id = *hook_counter;
                *hook_counter += 1;
                hook_defs.push(HookDef {
                    hook_id,
                    stmts: vec![stmt],
                });

                (NumChunk::CallHook(hook_id), hook_defs)
            }
        })
        .unzip();
    let hook_defs = hook_defs.into_iter().flatten().collect();

    NumChunks {
        num_chunks,
        hook_defs,
    }
}

fn num_chunk_into_push_stmt(span: Span, num_chunk: NumChunk) -> TokenStream2 {
    match num_chunk {
        NumChunk::Visit(expr) => {
            quote_spanned! {span=>
                stack.push(StackEntry::Node(&#expr));
            }
        }

        NumChunk::CallHook(hook_id) => {
            quote_spanned! {span=>
                stack.push(StackEntry::Hook(#hook_id, node));
            }
        }
    }
}

fn hook_def_into_arm(span: Span, pat: &Pat, hook_def: HookDef) -> TokenStream2 {
    let HookDef { hook_id, stmts } = hook_def;

    quote_spanned! {span=>
        #[allow(unused)]
        StackEntry::Hook(#hook_id, #pat) => {
            #(#stmts)*
        }
    }
}

fn chunks_into_ent_arm(pat: &Pat, span: Span, chunks: Vec<Chunk>) -> TokenStream2 {
    let mut hook_counter = 0usize;
    let NumChunks {
        num_chunks,
        hook_defs,
    } = number_chunks(span, &mut hook_counter, chunks);

    if num_chunks.is_empty() {
        quote_spanned! {span=>
            StackEntry::Node(#pat) => {}
        }
    } else {
        let push_stmts: Vec<_> = num_chunks
            .into_iter()
            .map(|chunk| num_chunk_into_push_stmt(span, chunk))
            .rev()
            .collect();
        let hook_arms: Vec<_> = hook_defs
            .into_iter()
            .map(|def| hook_def_into_arm(span, pat, def))
            .collect();

        quote_spanned! {span=>
            #[allow(unused)]
            StackEntry::Node(node @ (#pat)) => {
                //#(#leading_stmts)*
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
