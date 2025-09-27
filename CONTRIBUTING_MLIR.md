# Contributing to the MLIR portion of Qwerty's Compiler

When trying to add an attribute to the Qwerty dialect, the process is broadly as follows:
1. Edit `qwerty_mlir/include/Qwerty/IR/QwertyAttributes.td`.
2. Add to `qwerty_mlir/include/CAPI/Qwerty.h` the following declarations for your new attribute:
```c++
/// Creates an qwerty::<your_attribute>
MLIR_CAPI_EXPORTED MlirAttribute mlirQwerty<your_attribute>Get(
        MlirContext ctx, <your inputs>);

/// Returns true if this is a qwerty::<your_attribute>.
MLIR_CAPI_EXPORTED bool mlirAttributeIsAQwerty<your_attribute>(MlirAttribute attr);
```
The best practice here is to look at other examples fortheir casing and formatting, so the bindings stick.
3. In `qwerty_mlir/include/CAPI/Qwerty.cpp`, provide implementations for the functions you templated above:
```c++

MlirAttribute mlirQwerty<your_attribute>Get(
        MlirContext ctx, <your inputs>) {

    return wrap(qwerty::<your_attribute>::get(unwrap(ctx),
            // your casts and construction of your attribute,
            // based on the Tablegen specification of it));
}

bool mlirAttributeIsAQwerty<your_attribute>(MlirAttribute attr) {
    return llvm::isa<qwerty::<your_attribute>>(unwrap(attr));
}
```
We use `qwerty_melior` as the interface between Rust and MLIR; earlier, we were creating C bindings for MLIR, and now Melior will use those to hook the MLIR up to Rust. Technically, there is a middle step but we don't need to worry about this for now.
4. In `qwerty_melior` (not `qwerty_mlir`), find `qwerty_melior/melior/src/ir/attribute/attribute_like.rs` and in the `attribute_check_functions!` macro call, add your `mlirAttributeIsAQwerty<your_attribute>` to the list.
5. In `qwerty_melior/melior/src/ir/dialect/qwerty.rs`, we now want to add the Rust structs and `impl`s that correspond to our C/MLIR functions. We do this as follows:
```rust
#[derive(Clone, Copy)]
pub struct <your_attribute><'c> {
    attribute: Attribute<'c>,
}

impl<'c> <your_attribute><'c> {
    pub fn new(
        context: &'c Context,
        // your inputs here
    ) -> Self {
        unsafe {
            Self::from_raw(mlirQwerty<your_attribute>Get( // same function as before!
                context.to_raw(),
                // your inputs .to_raw();
            ))
        }
    }
}

// We need this macro call to implement a series of helper functions
// This is why we registered our "IsA" function in the previous step; take a peek in
// `qwerty_melior/src/ir/attribute/macro.rs` to learn more!
// The formatting of the "is_qwerty" statement should reflect the lowercase snakecase
// version of your struct's name, without the "Attribute"
attribute_traits!(
    <your_attribute_struct>,
    is_qwerty_<your_attribute_snakecase>,
    "qwerty some description of your attribute"
);
```
Remember to follow the naming conventions of the `qwerty.rs` file, as they are different to `Qwerty.h/cpp` (Attr vs Attribute in Rust, for example)
6. Smile, for if you run `maturin develop -vv`, the powers at be should smile upon ye.
