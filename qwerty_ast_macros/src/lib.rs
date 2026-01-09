//! This crate defines the `gen_rebuild` framework used to modify AST data
//! structures  without tedious handwritten recursive code that blows up the
//! stack and my wrists. For example, expanding `DimExpr`s even if they are
//! deep inside a giant tree of `MetaExpr`s and `MetaBasis`es requires
//! handwritten recursive calls for each of the fields of each of the variants
//! of these enums.
use proc_macro::TokenStream;

mod rebuild;
mod syn_util;
mod visitor;

fn result_to_token_stream(res: Result<TokenStream, syn::Error>) -> TokenStream {
    res.unwrap_or_else(|err| err.into_compile_error().into())
}

/// Generates a function that rebuilds this data structure.
///
/// Written as an enum attribute on an enum as seen in the following example:
/// ```
/// # mod example { // From: https://github.com/rust-lang/rust/issues/130274#issuecomment-2656216123
/// # use qwerty_ast_macros::gen_rebuild;
/// # type DimVar = ();
/// # type DimExpr = ();
/// # type Progress = ();
/// # type MacroEnv = ();
/// # type LowerError = ();
/// #[gen_rebuild {
///     substitute_dim_var(
///         more_copied_args(dim_var: &DimVar, new_dim_expr: &DimExpr),
///         recurse_attrs,
///     ),
///     expand(
///         rewrite(expand_rewriter),
///         progress(Progress),
///         more_copied_args(env: &MacroEnv),
///         result_err(LowerError),
///         recurse_attrs,
///     ),
/// }]
/// #[derive(Debug, Clone, PartialEq)]
/// pub enum FloatExpr {
/// # }
/// # }
/// ```
/// Here, `substitute_dim_var` and `expand` are example configuration names,
/// whereas `more_copied_args` and `recurse_attrs` are config options for
/// `substitute_dim_var`. The next section describes options.
///
/// The enum this is attached to must satisfy the following:
/// 1. The variant fields through which to recurse should be found directly in
///    the fields of struct variants. That is, this is wrong:
///    ```
///    # use qwerty_ast_macros::gen_rebuild;
///    # type DimExpr = ();
///    # type DebugLoc = ();
///    struct FloatDimExpr {
///        expr: DimExpr,
///        dbg: Option<DebugLoc>,
///    }
///    #[gen_rebuild { /*...*/ }]
///    pub enum FloatExpr {
///        FloatDimExpr(FloatDimExpr),
///    }
///    ```
///    but this is correct:
///    ```
///    # use qwerty_ast_macros::gen_rebuild;
///    # type DimExpr = ();
///    # type DebugLoc = ();
///    #[gen_rebuild { /*...*/ }]
///    pub enum FloatExpr {
///        FloatDimExpr {
///            expr: DimExpr,
///            dbg: Option<DebugLoc>,
///        },
///    }
///    ```
/// 2. The enum must have only unit and struct variants
/// 3. The enum must have no empty struct variants (use unit variants instead)
///
/// In the remainder of this documentation, `enum_name` represents the name of
/// the enum marked with the `gen_rebuild` attribute. In the examples above,
/// for instance, `enum_name` would be `FloatExpr`.
///
/// In each struct variant, if a field has type `Box<enum_name>` or
/// `Vec<enum_name>`, then it is a _child_. Otherwise, it is an _attr_.
/// Recursion into children is automatic, and each child is pushed onto a stack
/// stored on the heap rather than directly calling a method to avoid stack
/// overflow. By default, recursion into attributes (such as `expr: DimExpr`
/// above) is disabled and uses a direct method call, not the aforementioned
/// stack. That is, by default, attributes are simply moved over as-is.
///
/// # `gen_rebuild::skip_recurse` Attribute
///
/// If this attribute is found on a struct field and the field is a child, the
/// field will be treated as an attr with the same attribute. If the
/// `recurse_attrs` option of `gen_rebuild` is set, then recursion will be
/// skipped for any attr marked with `gen_rebuild::skip_recurse`. For some
/// hardcoded primitive types such as `f64` or `char`, this attribute is
/// implied.
///
/// # Options
/// The following options may be passed to each configuration:
///
/// * `recurse_attrs` (no arguments, default false when not present): Call
///   `attr.config_name()` unless the field is marked with the
///   `#[gen_rebuild::skip_recurse]` attribute.
/// * `rewrite(method_name)` (one argument, an identifier): After rebuilding
///   each `child`, call `child.method_name()`. See
///   `MetaBasis::substitute_basis_alias_rewriter` for an example.
/// * `progress(progress_ty)` (one argument, a type): return progress from the
///   rewriter (if present) and the ultimate generated rebuild function. That
///   is, each returns `(enum_name, progress_ty)` rather than just `enum_name`.
///   `progress_ty` should have a `join()` method that merges progress and a
///   `progress_ty::identity()` associated function.
/// * `result_err(err_ty)` (one argument, a type): return
///   `Result<enum_name, err_ty>` from the rewriter (if present). Inserts more
///   `?` operators into the generated code than you could ever dream.
/// * `rewrite_to(orig_ty => rewritten_ty, ...)` (many arguments, each a type
///   followed by a `=>` token and another type). A way to convert `enum_name`
///   to a different type. A rewriter specified with `rewrite` is required and
///   takes a generated enum as an argument that contains rewritten types in each
///   variant. See `MetaBasisGenerator::extract_rewriter` for an example.
/// * `more_generic_params(T, ...)` (many arguments, each a generic parameter):
///   additional generic parameters to add to the generated rebuild function.
///   See `qpu::MetaExpr::expand_instantiations` for an example.
/// * `more_where(T, ...)` (many arguments, each a generic parameter):
///   additional clauses to include in the `where` portion of the generated
///   rebuild function. See `qpu::MetaExpr::expand_instantiations` for an
///   example.
/// * `more_copied_args(arg1: arg1_ty, arg2: arg2_ty, ...)`: More arguments
///   passed to the rewriter and expected by the rebuild function. These
///   arguments must implement the [`Copy`] trait, since each will be copied
///   for each call to the rewriter.
/// * `more_moved_args(arg1: arg1_ty, arg2: arg2_ty, ...)`: More arguments
///   passed to the rewriter and expected by the rebuild function. These
///   are moved into each call and returned from each call.
///
/// # Example
/// Consider the folowing example mini-AST enum:
/// ```
/// # mod example { // From: https://github.com/rust-lang/rust/issues/130274#issuecomment-2656216123
/// use qwerty_ast_macros::{gen_rebuild, rebuild};
///
/// #[gen_rebuild {
///     swap_operands(
///         rewrite(swap_operands_rewriter),
///     ),
/// }]
/// # #[derive(Debug, PartialEq)]
/// enum Expr {
///     Constant { val: u32 },
///     Add { lhs: Box<Expr>, rhs: Box<Expr> },
/// }
///
/// impl Expr {
///     pub(crate) fn swap_operands_rewriter(self) -> Self {
///         match self {
///             Expr::Add { lhs, rhs } => Expr::Add { lhs: rhs, rhs: lhs },
///             constant @ Expr::Constant { .. } => constant,
///         }
///     }
///
///     pub fn swap_operands(self) -> Self {
///         rebuild!(Expr, self, swap_operands)
///     }
/// }
/// # pub fn example() {
/// #     let input = Expr::Add {
/// #         lhs: Box::new(Expr::Constant { val: 1 }),
/// #         rhs: Box::new(Expr::Constant { val: 3 }),
/// #     };
/// #     let actual = input.swap_operands();
/// #     let expected = Expr::Add {
/// #         lhs: Box::new(Expr::Constant { val: 3 }),
/// #         rhs: Box::new(Expr::Constant { val: 1 }),
/// #     };
/// #     assert_eq!(actual, expected);
/// # }
/// # }
/// # example::example();
/// ```
///
/// The generated `rebuild()` code looks _roughly_ like the following
/// pseudocode:
/// ```
/// # mod example { // From: https://github.com/rust-lang/rust/issues/130274#issuecomment-2656216123
/// # #[derive(Debug, PartialEq)]
/// enum Expr {
///     Constant { val: u32 },
///     Add { lhs: Box<Expr>, rhs: Box<Expr> },
/// }
///
/// impl Expr {
///     pub(crate) fn swap_operands_rewriter(self) -> Self {
///         match self {
///             Expr::Add { lhs, rhs } => Expr::Add { lhs: rhs, rhs: lhs },
///             constant @ Expr::Constant { .. } => constant,
///         }
///     }
///
///     pub fn swap_operands(self) -> Self {
///         expr::swap_operands::rebuild(self)
///     }
/// }
///
/// pub mod expr {
///     pub mod swap_operands {
///         use super::super::*;
///
///         enum Reconstruct {
///             Constant { val: u32 },
///             Add,
///         }
///
///         enum RebuildStackEntry {
///             Deconstruct(Expr),
///             Reconstruct(Reconstruct),
///         }
///
///         pub fn rebuild(root: Expr) -> Expr {
///             let mut in_stack = vec![RebuildStackEntry::Deconstruct(root)];
///             let mut out_stack = vec![];
///
///             while let Some(node) = in_stack.pop() {
///                 match node {
///                     RebuildStackEntry::Deconstruct(Expr::Constant { val }) => {
///                         in_stack.push(RebuildStackEntry::Reconstruct(Reconstruct::Constant { val }));
///                     }
///                     RebuildStackEntry::Deconstruct(Expr::Add { lhs, rhs }) => {
///                         in_stack.push(RebuildStackEntry::Reconstruct(Reconstruct::Add));
///                         in_stack.push(RebuildStackEntry::Deconstruct(*lhs));
///                         in_stack.push(RebuildStackEntry::Deconstruct(*rhs));
///                     }
///                     RebuildStackEntry::Reconstruct(Reconstruct::Constant { val }) => {
///                         out_stack.push(Expr::Constant { val }.swap_operands_rewriter());
///                     }
///                     RebuildStackEntry::Reconstruct(Reconstruct::Add) => {
///                         let lhs = Box::new(out_stack.pop().unwrap());
///                         let rhs = Box::new(out_stack.pop().unwrap());
///                         out_stack.push(Expr::Add { lhs, rhs }.swap_operands_rewriter());
///                     }
///                 }
///             }
///
///             out_stack.pop().unwrap()
///         }
///     }
/// }
/// # pub fn example() {
/// #     let input = Expr::Add {
/// #         lhs: Box::new(Expr::Constant { val: 1 }),
/// #         rhs: Box::new(Expr::Constant { val: 3 }),
/// #     };
/// #     let actual = input.swap_operands();
/// #     let expected = Expr::Add {
/// #         lhs: Box::new(Expr::Constant { val: 3 }),
/// #         rhs: Box::new(Expr::Constant { val: 1 }),
/// #     };
/// #     assert_eq!(actual, expected);
/// # }
/// # }
/// # example::example();
/// ```
#[proc_macro_attribute]
pub fn gen_rebuild(attr: TokenStream, item: TokenStream) -> TokenStream {
    result_to_token_stream(rebuild::impl_gen_rebuild(attr, item))
}

/// Generates rebuild code when variant structs are defined outside the enum.
///
/// That is, [`gen_rebuild_structs`] generates code nearly identically to
/// [`gen_rebuild`] except that it expects each enum variant to be a tuple
/// variant which contains only another variant-specific struct that shares the
/// variant's name. Unlike the attribute macro [`gen_rebuild`],
/// [`gen_rebuild_structs`] is a procedural macro that must be explicitly
/// called. It expects two arguments:
///
/// 1. `configs { ... }`, which contains the same config arguments that
///    [`gen_rebuild`] expects.
/// 2. `defs { ... }`, which contains the definitions of the variant structs
///    and the definition of the enum in any order.
///
/// Example:
/// ```
/// # mod example { // From: https://github.com/rust-lang/rust/issues/130274#issuecomment-2656216123
/// use qwerty_ast_macros::{gen_rebuild_structs, rebuild};
///
/// gen_rebuild_structs! {
///     configs {
///         swap_operands(
///             rewrite(swap_operands_rewriter),
///         ),
///     }
///
///     defs {
/// #       #[derive(Debug, PartialEq)]
///         struct Constant { val: u32 }
///
/// #       #[derive(Debug, PartialEq)]
///         struct Add { lhs: Box<Expr>, rhs: Box<Expr> }
///
/// #       #[derive(Debug, PartialEq)]
///         enum Expr {
///             Constant(Constant),
///             Add(Add),
///         }
///     }
/// }
///
/// impl Expr {
///     pub(crate) fn swap_operands_rewriter(self) -> Self {
///         match self {
///             Expr::Add(Add { lhs, rhs }) => Expr::Add(Add { lhs: rhs, rhs: lhs }),
///             constant @ Expr::Constant(_) => constant,
///         }
///     }
///
///     pub fn swap_operands(self) -> Self {
///         rebuild!(Expr, self, swap_operands)
///     }
/// }
/// # pub fn example() {
/// #     let input = Expr::Add(Add {
/// #         lhs: Box::new(Expr::Constant(Constant { val: 1 })),
/// #         rhs: Box::new(Expr::Constant(Constant { val: 3 })),
/// #     });
/// #     let actual = input.swap_operands();
/// #     let expected = Expr::Add(Add {
/// #         lhs: Box::new(Expr::Constant(Constant { val: 3 })),
/// #         rhs: Box::new(Expr::Constant(Constant { val: 1 })),
/// #     });
/// #     assert_eq!(actual, expected);
/// # }
/// # }
/// # example::example();
/// ```
/// This expands similarly to the [`gen_rebuild`] example.
#[proc_macro]
pub fn gen_rebuild_structs(args: TokenStream) -> TokenStream {
    result_to_token_stream(rebuild::impl_gen_rebuild_structs(args))
}

/// Calls a function generated by [`gen_rebuild`].
///
/// Expects the following arguments:
/// 1. A path to the enum (e.g., `qpu::MetaVector`)
/// 2. An expression of type `enum_name` (usually `self`)
/// 3. The configuration name, an identifier
/// 4. Any additional arguments (expressions) as needed
///
/// Example of implementing a public-facing method that calls into the code
/// generated by [`gen_rebuild`] using [`rebuild!`]:
/// ```
/// # mod example { // From: https://github.com/rust-lang/rust/issues/130274#issuecomment-2656216123
/// # use qwerty_ast_macros::rebuild;
/// # type DimVar = ();
/// # type DimExpr = ();
/// # struct FloatExpr;
/// # pub mod float_expr {
/// # pub mod substitute_dim_var {
/// # use super::super::*;
/// # pub fn rebuild(root: FloatExpr, _dim_var: &DimVar, _new_dim_expr: &DimExpr) -> FloatExpr { root }
/// # }
/// # }
/// impl FloatExpr {
///     pub fn substitute_dim_var(self, dim_var: &DimVar, new_dim_expr: &DimExpr) -> FloatExpr {
///         rebuild!(FloatExpr, self, substitute_dim_var, dim_var, new_dim_expr)
///     }
/// }
/// # }
/// ```
/// The generated code would be (roughly):
/// ```
/// # mod example { // From: https://github.com/rust-lang/rust/issues/130274#issuecomment-2656216123
/// # use qwerty_ast_macros::rebuild;
/// # type DimVar = ();
/// # type DimExpr = ();
/// # struct FloatExpr;
/// # pub mod float_expr {
/// # pub mod substitute_dim_var {
/// # use super::super::*;
/// # pub fn rebuild(root: FloatExpr, _dim_var: &DimVar, _new_dim_expr: &DimExpr) -> FloatExpr { root }
/// # }
/// # }
/// impl FloatExpr {
///     pub fn substitute_dim_var(self, dim_var: &DimVar, new_dim_expr: &DimExpr) -> FloatExpr {
///         float_expr::substitute_dim_var::rebuild(self, dim_var, new_dim_expr)
///     }
/// }
/// # }
/// ```
#[proc_macro]
pub fn rebuild(args: TokenStream) -> TokenStream {
    result_to_token_stream(rebuild::impl_rebuild(args))
}

/// The input type to a rewriter when using `rewrite_to`.
///
/// When `rewrite_to` is passed to [`gen_rebuild`], the `rewriter` function
/// must take a generated enum type as input rather than `enum_name`. Recursion
/// results in children (and possibly attrs) having different types than before,
/// making it impossible to have an argument of type `enum_name`. To avoid
/// having to hardcode the name of this generated enum in wrapper code, the
/// macro `rewrite_ty(...)` expands to the name of the generated rewritten enum
/// type.
///
/// See [`rewrite_match`] for a full example.
#[proc_macro]
pub fn rewrite_ty(args: TokenStream) -> TokenStream {
    result_to_token_stream(rebuild::impl_rewrite_ty(args))
}

/// Generates a `match` for a rewriter when using `rewrite_to`.
///
/// See [`rewrite_ty`] for details on the generated rewritten enum.
///
/// This macro expects four arguments:
/// 1. A path to the enum (e.g., `qpu::MetaVector`)
/// 2. The configuration name, an identifier
/// 3. An expression whose type is the rewritten enum (usually an argument of type `rewrite_ty!(...)`)
/// 4. Arms of the match statement. The variant names should **not** be
///    qualified. In the example below, for instance, notice how the first arm
///    is `Constant` instead of `rewrite_ty!(...)::Constant`. (The macro will
///    qualify the names for you.)
///
/// Example which also uses [`rewrite_ty`]:
/// ```
/// # mod example { // From: https://github.com/rust-lang/rust/issues/130274#issuecomment-2656216123
/// # use qwerty_ast_macros::{gen_rebuild, rebuild, rewrite_ty, rewrite_match};
/// #
/// #[gen_rebuild {
///     calculate(
///         rewrite(calculate_rewriter),
///         rewrite_to(Expr => u32),
///     ),
/// }]
/// enum Expr {
///     Constant { val: u32 },
///     Add { lhs: Box<Expr>, rhs: Box<Expr> },
/// }
///
/// impl Expr {
///     pub(crate) fn calculate_rewriter(
///         rewritten: rewrite_ty!(Expr, calculate)
///     ) -> u32 {
///         rewrite_match!{ Expr, calculate, rewritten,
///             Constant { val } => val,
///             Add { lhs, rhs } => lhs + rhs,
///         }
///     }
///
///     pub fn calculate(self) -> u32 {
///         rebuild!(Expr, self, calculate)
///     }
/// }
/// # pub fn example() {
/// #     let input = Expr::Add {
/// #         lhs: Box::new(Expr::Constant { val: 1 }),
/// #         rhs: Box::new(Expr::Constant { val: 3 }),
/// #     };
/// #     let actual = input.calculate();
/// #     let expected = 4;
/// #     assert_eq!(actual, expected);
/// # }
/// # }
/// # example::example();
/// ```
/// The last portion expands as follows:
/// ```
/// # mod example { // From: https://github.com/rust-lang/rust/issues/130274#issuecomment-2656216123
/// # use qwerty_ast_macros::gen_rebuild;
/// #
/// #[gen_rebuild {
///     calculate(
///         rewrite(calculate_rewriter),
///         rewrite_to(Expr => u32),
///     ),
/// }]
/// enum Expr {
///     Constant { val: u32 },
///     Add { lhs: Box<Expr>, rhs: Box<Expr> },
/// }
///
/// impl Expr {
///     pub(crate) fn calculate_rewriter(
///         rewritten: expr::calculate::Rewritten
///     ) -> u32 {
///         match rewritten {
///             expr::calculate::Rewritten::Constant { val } => val,
///             expr::calculate::Rewritten::Add { lhs, rhs } => lhs + rhs,
///         }
///     }
///
///     pub fn calculate(self) -> u32 {
///         expr::calculate::rebuild(self)
///     }
/// }
/// # pub fn example() {
/// #     let input = Expr::Add {
/// #         lhs: Box::new(Expr::Constant { val: 1 }),
/// #         rhs: Box::new(Expr::Constant { val: 3 }),
/// #     };
/// #     let actual = input.calculate();
/// #     let expected = 4;
/// #     assert_eq!(actual, expected);
/// # }
/// # }
/// # example::example();
/// ```
#[proc_macro]
pub fn rewrite_match(args: TokenStream) -> TokenStream {
    result_to_token_stream(rebuild::impl_rewrite_match(args))
}

#[proc_macro]
pub fn visitor_match(args: TokenStream) -> TokenStream {
    result_to_token_stream(visitor::impl_visitor_match(args))
}
