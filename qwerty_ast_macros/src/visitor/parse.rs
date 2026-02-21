use crate::syn_util::paths;
use proc_macro2::Span;
use syn::{
    Arm, Block, Error, Expr, ExprBinary, ExprBlock, ExprCall, ExprMacro, ExprMethodCall, ExprParen, ExprPath, ExprUnary, Ident, LitStr, Local, LocalInit, Macro, Stmt, Token, Type, parse::{Parse, ParseStream}, punctuated::Punctuated, spanned::Spanned
};

/// Holds the parsed arguments for a call to `visitor_write!{}` or
/// `visitor_expr!{}`.
pub struct VisitorMacroCall {
    pub ty: Type,
    pub match_on: Expr,
    pub arms: Vec<Arm>,
}

impl Parse for VisitorMacroCall {
    fn parse(input: ParseStream) -> Result<Self, Error> {
        let ty = input.parse()?;
        input.parse::<Token![,]>()?;
        let match_on = input.parse()?;
        input.parse::<Token![,]>()?;

        let mut arms = Vec::new();
        while !input.is_empty() {
            let arm: Arm = input.parse()?;
            if !arm.attrs.is_empty() {
                let first_attr = arm
                    .attrs
                    .into_iter()
                    .next()
                    .expect("attrs is nonempty yet empty");
                return Err(Error::new_spanned(
                    first_attr,
                    "Attributes not allowed on match arms",
                ));
            }
            arms.push(arm);
        }

        Ok(VisitorMacroCall { ty, match_on, arms })
    }
}

/// Represents a type of node visit.
pub enum VisitKind {
    /// Pushes this individual expression (a node) onto the stack.
    SingleVisit { node: Expr },
    /// Iterates over the iterator `nodes` and push each onto the stack with
    /// the string `sep` between each.
    MultiVisit { nodes: Expr, sep: Expr },
}

/// Represents one chunk of a format string. For example,
/// `write!(f, "foo {} bar {!} baz", my_attr, my_child)`
/// becomes three segments:
/// 1. `FormatString("foo {} bar ", [my_attr])`
/// 2. `Visit(my_child)`
/// 3. `FormatString(" baz", [])`
pub enum WriteCallSegment {
    FormatString(String, Vec<Expr>),
    Visit(VisitKind),
}

/// Holds the parsed arguments for a call to `write!(...)`.
pub struct WriteCallArgs {
    pub dest: Expr,
    pub segs: Vec<WriteCallSegment>,
}

/// Parses a string literal used in a `write!(...)` call into a list of
/// [`WriteCallSegment`]s.
fn parse_fmt_string_to_segs(
    fmt_str: LitStr,
    args: Vec<Expr>,
) -> Result<Vec<WriteCallSegment>, Error> {
    let mut arg_stack: Vec<_> = args.into_iter().rev().collect();

    #[derive(Debug, Clone, Copy)]
    enum State {
        Start,
        GotOpen,
        IgnoreUntilClose,
        GotBang,
        GotBangColon,
    }

    let mut segs = Vec::new();
    let mut cur_args = Vec::new();
    let mut cur_seg = String::new();
    let mut cur_comma_sep = false;
    let mut state = State::Start;

    for c in fmt_str.value().chars() {
        let (new_state, push_c1, push_c2) = match (state, c) {
            // "{}"
            //  ^
            (State::Start, '{') => Ok((State::GotOpen, None, None)),
            (State::Start, other_c) => Ok((State::Start, Some(other_c), None)),

            // "{{"
            //   ^
            (State::GotOpen, '{') => Ok((State::Start, Some('{'), Some('{'))),
            // "{!}"
            //   ^
            (State::GotOpen, '!') => Ok((State::GotBang, None, None)),
            // {}
            //  ^
            (State::GotOpen, '}') => {
                if let Some(arg) = arg_stack.pop() {
                    cur_args.push(arg);
                    Ok((State::Start, Some('{'), Some('}')))
                } else {
                    Err(Error::new_spanned(
                        &fmt_str,
                        "Invalid format string. Number of args does not match number of {}s",
                    ))
                }
            }
            // This is not our problem:
            // "{}"
            //   ^
            // or:
            // "{:b}"
            //   ^
            (State::GotOpen, other_c) => Ok((State::IgnoreUntilClose, Some('{'), Some(other_c))),

            // {:b}
            //    ^
            (State::IgnoreUntilClose, '}') => {
                if let Some(arg) = arg_stack.pop() {
                    cur_args.push(arg);
                    Ok((State::Start, Some('}'), None))
                } else {
                    Err(Error::new_spanned(
                        &fmt_str,
                        "Invalid format string. Number of args does not match number of {}s",
                    ))
                }
            }
            // {:b}
            //   ^
            (State::IgnoreUntilClose, other_c) => {
                Ok((State::IgnoreUntilClose, Some(other_c), None))
            }

            // "{!}"
            //    ^
            (State::GotBang | State::GotBangColon, '}') => {
                if !cur_seg.is_empty() {
                    segs.push(WriteCallSegment::FormatString(cur_seg, cur_args));
                    cur_seg = String::new();
                    cur_args = Vec::new();
                }
                if cur_comma_sep {
                    if let Some((nodes, sep)) = arg_stack.pop().zip(arg_stack.pop()) {
                        segs.push(WriteCallSegment::Visit(VisitKind::MultiVisit {
                            nodes,
                            sep,
                        }));
                    } else {
                        Err(Error::new_spanned(
                            &fmt_str,
                            "Invalid format string. No node expression for {!}",
                        ))?;
                    }
                } else {
                    if let Some(node) = arg_stack.pop() {
                        segs.push(WriteCallSegment::Visit(VisitKind::SingleVisit { node }));
                    } else {
                        Err(Error::new_spanned(
                            &fmt_str,
                            "Invalid format string. No node expression for {!}",
                        ))?;
                    }
                }
                cur_comma_sep = false;
                Ok((State::Start, None, None))
            }
            // "{!:,}"
            //    ^
            (State::GotBang, ':') => Ok((State::GotBangColon, None, None)),
            // {!hurr}
            //   ^
            (State::GotBang, _) => Err(Error::new_spanned(
                &fmt_str,
                "Invalid format string. Expected } after !.",
            )),

            // "{!:,}"
            //     ^
            (State::GotBangColon, ',') => {
                if cur_comma_sep {
                    Err(Error::new_spanned(
                        &fmt_str,
                        "Invalid format string. Option `,` specified multiple times.",
                    ))
                } else {
                    cur_comma_sep = true;
                    Ok((State::GotBangColon, None, None))
                }
            }
            // "{!:DUHHH}"
            //     ^
            (State::GotBangColon, _) => Err(Error::new_spanned(
                &fmt_str,
                "Invalid format string. Unexpected character after `!:`.",
            )),
        }?;
        state = new_state;
        if let Some(push_char) = push_c1 {
            cur_seg.push(push_char);
        }
        if let Some(push_char) = push_c2 {
            cur_seg.push(push_char);
        }
    }

    match state {
        State::Start => {
            if !cur_seg.is_empty() {
                segs.push(WriteCallSegment::FormatString(cur_seg, cur_args));
            }
            if !arg_stack.is_empty() {
                Err(Error::new_spanned(
                    &fmt_str,
                    "Wrong number of format arguments",
                ))
            } else {
                Ok(())
            }
        }

        State::GotOpen | State::IgnoreUntilClose | State::GotBang | State::GotBangColon => Err(
            Error::new_spanned(&fmt_str, "Unexpected ending of format string"),
        ),
    }?;

    Ok(segs)
}

impl Parse for WriteCallArgs {
    /// Parses this part of of a `write!()` call:
    /// ```text
    /// write!(f, "foo{}", "bar")
    ///        ^^^^^^^^^^^^^^^^^
    /// ```
    fn parse(input: ParseStream) -> Result<Self, Error> {
        let dest = input.parse()?;
        input.parse::<Token![,]>()?;
        let fmt_str = input.parse::<LitStr>()?;
        let fmt_args: Vec<Expr> = if input.peek(Token![,]) {
            input.parse::<Token![,]>()?;
            // .rev() so that we can use this as a stack
            input
                .parse_terminated(Expr::parse, Token![,])?
                .into_iter()
                .collect()
        } else {
            vec![]
        };
        let segs = parse_fmt_string_to_segs(fmt_str, fmt_args)?;
        Ok(WriteCallArgs { dest, segs })
    }
}

/// Parses the arm of a `visitor_write!{...}` as a `write!(...)` expression.
pub fn parse_visitor_write_arm_expr(body: Expr) -> Result<WriteCallArgs, Error> {
    match body {
        Expr::Macro(ExprMacro {
            attrs,
            mac:
                Macro {
                    path,
                    bang_token: _,
                    delimiter: _,
                    tokens,
                },
        }) if paths::path_is_ident_str(&path, "write") => {
            if !attrs.is_empty() {
                let first_attr = attrs
                    .into_iter()
                    .next()
                    .expect("attrs is nonempty yet empty");
                Err(Error::new_spanned(
                    first_attr,
                    "Attributes not allowed on write!(...)",
                ))
            } else {
                syn::parse2(tokens)
            }
        }

        Expr::Block(ExprBlock {
            attrs,
            label,
            block: Block { brace_token, stmts },
        }) => {
            if !attrs.is_empty() {
                let first_attr = attrs
                    .into_iter()
                    .next()
                    .expect("attrs is nonempty yet empty");
                Err(Error::new_spanned(
                    first_attr,
                    "Attributes not allowed on block inside visitor_write!{...} arm",
                ))
            } else if let Some(label) = label {
                Err(Error::new_spanned(
                    label,
                    "Label not allowed on block inside visitor_write!{...} arm",
                ))
            } else if stmts.is_empty() {
                Err(Error::new(
                    brace_token.span.open(),
                    "Empty block in visitor_write!{...} arm is not allowed",
                ))
            } else if stmts.len() > 1 {
                Err(Error::new(
                    brace_token.span.open(),
                    "Block in visitor_write!{...} arm must have exactly one expression statement",
                ))
            } else {
                let stmt = stmts
                    .into_iter()
                    .next()
                    .expect("stmts has length one yet is empty");
                match stmt {
                    Stmt::Expr(expr, None) => parse_visitor_write_arm_expr(expr),
                    other_stmt => Err(Error::new_spanned(
                        other_stmt,
                        "Block in visitor_write!{...} arm must have exactly one expression with no semicolon",
                    )),
                }
            }
        }

        other_expr => Err(Error::new_spanned(
            other_expr,
            "Expected a call to write!(...), got a different expression",
        )),
    }
}

/// Recursive subroutine used by [`parse_visitor_expr_arm_expr`] to rebuild
/// expressions without `visit!(...)` calls.
fn parse_visitor_expr_arm_expr_helper(
    body: Expr,
    visit_var_name_gen: fn(usize, Span) -> Ident,
    visit_exprs_out: &mut Vec<Expr>,
) -> Result<Expr, Error> {
    match body {
        Expr::Binary(ExprBinary {
            attrs,
            left,
            op,
            right,
        }) => {
            let left = Box::new(parse_visitor_expr_arm_expr_helper(
                *left,
                visit_var_name_gen,
                visit_exprs_out,
            )?);
            let right = Box::new(parse_visitor_expr_arm_expr_helper(
                *right,
                visit_var_name_gen,
                visit_exprs_out,
            )?);
            Ok(Expr::Binary(ExprBinary {
                attrs,
                left,
                op,
                right,
            }))
        }

        Expr::Unary(ExprUnary { attrs, op, expr }) => {
            let expr = Box::new(parse_visitor_expr_arm_expr_helper(
                *expr,
                visit_var_name_gen,
                visit_exprs_out,
            )?);
            Ok(Expr::Unary(ExprUnary { attrs, op, expr }))
        }
        Expr::Block(ExprBlock { attrs, label, block }) => {
			let Block { brace_token, stmts } = block;
			let stmts = stmts
				.into_iter()
				.map(|stmt| {
					match stmt {
						Stmt::Expr(expr, semi_token) => {
							let expr = parse_visitor_expr_arm_expr_helper(expr, visit_var_name_gen, visit_exprs_out)?;
							Ok(Stmt::Expr(expr, semi_token))
						},
						Stmt::Local(Local {attrs, let_token, pat, init, semi_token}) => {
							// If we have a local initialization i.e let x = 3
							let init = match init {
								Some(LocalInit { eq_token, expr, diverge }) => {
									let diverge = match diverge {
										Some((else_token, diverge_expr)) => {
											let diverge_expr = parse_visitor_expr_arm_expr_helper(*diverge_expr, visit_var_name_gen, visit_exprs_out)?;
											Some((else_token, Box::new(diverge_expr)))
										},
										None => None,
									};
									let expr = parse_visitor_expr_arm_expr_helper(*expr, visit_var_name_gen, visit_exprs_out)?;
									Some(LocalInit{eq_token, expr: Box::new(expr), diverge: diverge})
								}
								None => None,
							};
							Ok(Stmt::Local(Local {attrs, let_token, pat, init, semi_token}))
						},
						// Check to make sure someone is not trying to use the {} syntax with visit!
						// All normal visit calls should be parsed as Expr::Macro
						Stmt::Macro(ref stmt_mac)
							if (paths::path_is_ident_str(&stmt_mac.mac.path, "visit")) => {
								Err(Error::new_spanned(
									stmt_mac,
									"visit! {...} with braces is not supported in visitor_expr!, \
									use visit!(...) instead"
								))
							},
						passthru @ (Stmt::Item(_) | Stmt::Macro(_)) => Ok(passthru),
					}
				}).collect::<Result<Vec<Stmt>, Error>>()?;

			let block = Block {brace_token, stmts: stmts};

			Ok(Expr::Block(ExprBlock{attrs, label, block}))
        },
        Expr::Paren(ExprParen {attrs, paren_token, expr}) => {
			let expr = parse_visitor_expr_arm_expr_helper(*expr, visit_var_name_gen, visit_exprs_out)?;
			Ok(Expr::Paren(ExprParen {attrs, paren_token, expr: Box::new(expr)}))
        }
        // Note: func is of type Box<Expr>, we do not recuse on this type
        Expr::Call(ExprCall {attrs, func, paren_token, args}) => {
	        let args = args
		        .into_iter()
		        .map(|arg_expr| {
					parse_visitor_expr_arm_expr_helper(arg_expr, visit_var_name_gen, visit_exprs_out)
			    }).collect::<Result<Punctuated<Expr, Token![,]>, Error>>()?;
			Ok(Expr::Call(ExprCall {attrs, func, paren_token, args}))
        }
        Expr::MethodCall(ExprMethodCall {attrs, receiver, dot_token, method, turbofish, paren_token, args}) => {
	        let receiver_expr = parse_visitor_expr_arm_expr_helper(*receiver, visit_var_name_gen, visit_exprs_out)?;

	        let args = args
		        .into_iter()
		        .map(|arg_expr| {
					parse_visitor_expr_arm_expr_helper(arg_expr, visit_var_name_gen, visit_exprs_out)
			    }).collect::<Result<Punctuated<Expr, Token![,]>, Error>>()?;
		    Ok(Expr::MethodCall(ExprMethodCall { attrs, receiver: Box::new(receiver_expr), dot_token, method, turbofish, paren_token, args }))
        },
        Expr::Macro(ExprMacro {
            attrs,
            mac:
                Macro {
                    path,
                    bang_token: _,
                    delimiter: _,
                    tokens,
                },
        }) if paths::path_is_ident_str(&path, "visit") => {
            let span = path.span();
            if !attrs.is_empty() {
                let first_attr = attrs
                    .into_iter()
                    .next()
                    .expect("attrs is nonempty yet empty");
                Err(Error::new_spanned(
                    first_attr,
                    "Attributes not allowed on visit!()",
                ))
            } else {
                let visit_var_name = visit_var_name_gen(visit_exprs_out.len(), span);
                visit_exprs_out.push(syn::parse2(tokens)?);
                let path = paths::ident_to_var_name_path(visit_var_name);
                let var_name_expr = Expr::Path(ExprPath {
                    attrs: Vec::new(),
                    qself: None,
                    path,
                });
                Ok(var_name_expr)
            }
        }

        passthru @ (Expr::Lit(_) | Expr::Path(_)) => Ok(passthru),

        other_expr => Err(Error::new_spanned(
            other_expr,
            "Unknown expression inside visitor_expr! arm",
        )),
    }
}

/// Parses the arm of a `visitor_expr!{...}`. Returns the list of expressions
/// to visit and then a rewritten expression. The callback passed is used to
/// generate variable names for `visit!()` invocations.
pub fn parse_visitor_expr_arm_expr(
    body: Expr,
    visit_var_name_gen: fn(usize, Span) -> Ident,
) -> Result<(Vec<Expr>, Expr), Error> {
    let mut visit_exprs = Vec::new();
    let expr = parse_visitor_expr_arm_expr_helper(body, visit_var_name_gen, &mut visit_exprs)?;
    Ok((visit_exprs, expr))
}
