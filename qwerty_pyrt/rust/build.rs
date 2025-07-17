//! This is a goofy hack to get reliable access to the qwerty-opt built by
//! qwerty-mlir-sys. The build.rs for `qwerty-mlir-sys` sets the metadata `bin_dir`
//! with the path to where it installed qwerty-opt and qwerty-translate. Then,
//! if the `qwerty-opt` feature is enabled, this crate depends on
//! `qwerty-mlir-sys`, so our `build.rs` recieves that piece of metadata. From
//! there, we can symlink that binary dir to /bin in the repo root. This is
//! intended only for local development; the final wheel should be built with
//! `--no-default-features`.

use std::{env, fs, path::Path};

fn main() {
    let manifest_dir_str = env::var("CARGO_MANIFEST_DIR").unwrap();
    let manifest_dir = Path::new(&manifest_dir_str);
    let repo_root = manifest_dir.parent().unwrap().parent().unwrap();
    let bin_dir_link = repo_root.join("bin");

    if fs::exists(&bin_dir_link).unwrap() {
        fs::remove_dir_all(&bin_dir_link).unwrap();
    }

    if let Err(_) = env::var("CARGO_FEATURE_QWERTY_OPT") {
        println!("cargo::warning=Not symlinking bin dir because qwerty-opt feature is not enabled");
    } else {
        let env_var = "DEP_MLIR_BIN_DIR";
        if let Ok(bin_dir_str) = env::var(env_var) {
            let bin_dir = Path::new(&bin_dir_str);

            #[cfg(unix)]
            {
                std::os::unix::fs::symlink(&bin_dir, &bin_dir_link).unwrap();
                println!(
                    "cargo::warning=Linked {} -> {}",
                    bin_dir_link.display(),
                    bin_dir.display()
                );
            }
            #[cfg(windows)]
            {
                std::os::windows::fs::symlink_dir(&bin_dir, &bin_dir_link).unwrap();
                println!(
                    "cargo::warning=Linked {} -> {}",
                    bin_dir_link.display(),
                    bin_dir.display()
                );
            }
            #[cfg(not(any(unix, windows)))]
            {
                println!(
                    concat!(
                        "cargo::warning=I don't know how to symlink on your "
                        "OS, but you can find qwerty-opt etc. at: {}",
                    ),
                    bin_dir.display()
                );
            }
        } else {
            println!(
                concat!(
                    "cargo::warning=Environment variable ${} not set, so no ",
                    "symlinks to qwerty-opt will be created.",
                ),
                env_var
            );
        }
    }
}
