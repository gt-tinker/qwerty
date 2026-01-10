use syn::{
    Arm, Block, Error, Expr, ExprForLoop, Pat, Token, Type,
    parse::{Parse, ParseStream},
};

/// Holds the parsed arguments for a call to `visitor_match!{}`.
pub struct VisitorMatchCall {
    pub ty: Type,
    pub match_on: Expr,
    pub arms: Vec<Arm>,
}

impl Parse for VisitorMatchCall {
    fn parse(input: ParseStream) -> Result<Self, Error> {
        let ty = input.parse()?;
        input.parse::<Token![,]>()?;
        let match_on = input.parse()?;
        input.parse::<Token![,]>()?;

        let mut arms = Vec::new();
        while !input.is_empty() {
            arms.push(input.parse()?);
        }

        Ok(VisitorMatchCall { ty, match_on, arms })
    }
}

pub struct VisitMacroCall {
    pub expr: Expr,
}

impl Parse for VisitMacroCall {
    fn parse(input: ParseStream) -> Result<Self, Error> {
        let expr = input.parse()?;
        Ok(VisitMacroCall { expr })
    }
}

pub struct VisitForMacroCall {
    pub pat_ty: Type,
    pub pat: Pat,
    pub expr: Expr,
    pub body: Block,
}

impl Parse for VisitForMacroCall {
    fn parse(input: ParseStream) -> Result<Self, Error> {
        let pat_ty = input.parse()?;
        input.parse::<Token![,]>()?;

        let ExprForLoop {
            attrs,
            label,
            for_token: _,
            pat,
            in_token: _,
            expr,
            body,
        } = input.parse()?;

        if !attrs.is_empty() {
            let first_attr = attrs
                .into_iter()
                .next()
                .expect("attrs is nonempty yet empty");
            return Err(Error::new_spanned(
                first_attr,
                "Attributes not allowed on visit for loops",
            ));
        }

        if let Some(label) = label {
            return Err(Error::new_spanned(
                label,
                "Labels not allowed on visit for loops",
            ));
        }

        let pat = *pat;
        let expr = *expr;
        Ok(VisitForMacroCall {
            pat_ty,
            pat,
            expr,
            body,
        })
    }
}
