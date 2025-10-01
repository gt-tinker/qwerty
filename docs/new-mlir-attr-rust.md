Plumbing a New MLIR Attribute Through C++, C, and Rust
======================================================

Modifying our MLIR dialects is an exercise in understanding the interplay
between four languages: C++, C, Rust, and Tablegen. This guide walks through
the process of adding an attribute to the `qwerty` MLIR dialect as an example.

If we are adding an attribute name `YourAttr` to the `qwerty` dialect, the
process is broadly as follows:

 1. Edit `qwerty_mlir/include/Qwerty/IR/QwertyAttributes.td`.

 2. Add to `qwerty_mlir/include/CAPI/Qwerty.h` the following declarations for
    your new attribute:
    ```c++
    /// Creates an qwerty::your_attribute
    MLIR_CAPI_EXPORTED MlirAttribute mlirQwertyYourAttrGet(
            MlirContext ctx, your_inputs_here);

    /// Returns true if this is a qwerty::YourAttr
    MLIR_CAPI_EXPORTED bool mlirAttributeIsAQwertyYour(MlirAttribute attr);
    ```
    The best practice here is to look at other examples for their casing and
    formatting, so the bindings stick.

 3. In `qwerty_mlir/include/CAPI/Qwerty.cpp`, use C++ to define the functions
    that you declared in the C code above:
    ```c++

    MlirAttribute mlirQwertyYourAttributeGet(
            MlirContext ctx, your_inputs_here) {

        return wrap(qwerty::YourAttr::get(unwrap(ctx),
                // your casts and construction of your attribute,
                // based on the Tablegen specification of it));
    }

    bool mlirAttributeIsAQwertyYourAttribute(MlirAttribute attr) {
        return llvm::isa<qwerty::YourAttr>(unwrap(attr));
    }
    ```
    We use `qwerty_melior` as the interface between Rust and MLIR; earlier, we
    were creating C bindings for MLIR, and now Melior will use those to hook
    the MLIR up to Rust. Technically, there is a middle step but we don't need
    to worry about this for now.

 4. In `qwerty_melior` (not `qwerty_mlir`), find
    `qwerty_melior/melior/src/ir/attribute/attribute_like.rs` and in the
    `attribute_check_functions!` macro call, add your
    `mlirAttributeIsAQwertyYour` to the list.

 5. In `qwerty_melior/melior/src/ir/dialect/qwerty.rs`, we now want to add the
    Rust structs and `impl`s that correspond to our C/MLIR functions. We do
    this as follows:
    ```rust
    #[derive(Clone, Copy)]
    pub struct YourAttribute<'c> {
        attribute: Attribute<'c>,
    }

    impl<'c> YourAttribute<'c> {
        pub fn new(
            context: &'c Context,
            // your inputs here
        ) -> Self {
            unsafe {
                Self::from_raw(mlirQwertyYourAttributeGet( // same function as before!
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
        YourAttribute,
        is_qwerty_<your_attribute_snakecase>,
        "qwerty some description of your attribute"
    );
    ```
    Remember to follow the naming conventions of the `qwerty.rs` file, as they
    are different to `Qwerty.h/cpp` (Attr vs Attribute in Rust, for example)

 6. Smile, for if you run `maturin develop -vv`, the powers at be should smile
    upon ye.
