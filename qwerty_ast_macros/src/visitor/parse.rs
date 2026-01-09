use syn::{
    Arm, Error, Expr, Token, Type,
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
