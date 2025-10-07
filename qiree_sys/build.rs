//! Based on the `build.rs` in `qwerty_mlir_sys`, which is in turn based on
//! both the `cmake` crate documentation and the `build.rs` in `mlir_sys`.

use std::{
    env,
    error::Error,
    fs::read_dir,
    path::{Path, PathBuf},
    process::{Command, exit},
};

const LLVM_MAJOR_VERSION: usize = 21;
// Found in the qiree repo in src/qiree/CMakeLists.txt
const LLVM_COMPONENTS: &str = "Core irreader MCJIT native";

fn main() {
    if let Err(error) = run() {
        eprintln!("{}", error);
        exit(1);
    }
}

fn run() -> Result<(), Box<dyn Error>> {
    // The cmake crate panic()s on failure, so we do too throughout
    // build_qiree()
    let built_qiree = build_qiree();

    for rerun_if_changed_entry in built_qiree.rerun_if_changed.iter() {
        println!(
            "cargo::rerun-if-changed={}",
            rerun_if_changed_entry.display()
        );
    }

    run_bindgen(built_qiree)
}

struct BuiltQiree {
    rerun_if_changed: Vec<PathBuf>,
    include_dir: PathBuf,
    lib_dir: PathBuf,
    static_lib_names: Vec<String>,
}

fn build_qiree() -> BuiltQiree {
    let qiree_dir = PathBuf::from("..").join("qiree");

    let rerun_if_changed = vec![
        qiree_dir.join("qiree"),
    ];

    let install_dir = cmake::Config::new(qiree_dir)
        .generator("Ninja")
        // XACC is a wonderful tool but its reliance on dynamic linking will
        // make it quite challenging to integrate into the Qwerty Python
        // extension module
        .define("QIREE_USE_XACC", "OFF")
        // ...so use the QSim backend instead
        .define("QIREE_USE_QSIM", "ON")
        // No need to pull in gtest
        .define("QIREE_BUILD_TESTS", "OFF")
        // Statically link
        .define("BUILD_SHARED_LIBS", "OFF")
        // ...yet do -fPIC so that we can link all this into the Python
        // extension module (a shared library)
        .define("CMAKE_POSITION_INDEPENDENT_CODE", "ON")
        .build();
    let include_dir = install_dir.join("include");
    let lib_dir = install_dir.join("lib");

    // Check if include_dir is empty
    if let None = read_dir(&include_dir).unwrap().next() {
        panic!(
            "{} is an empty directory. Expected it to contain qiree header files",
            include_dir.display(),
        );
    }

    // We have to be careful with the ordering of linker args here. We need to
    // pass a topological ordering of this dependency graph:
    //
    //     libcqiree.a
    //          |
    //          V
    //     libqirqsim.a
    //          |
    //          V
    //       libqiree.a
    //
    // There is only one possible topological ordering, which we write below.

    let static_lib_names = vec![
        "libcqiree.a".to_string(),
        "libqirqsim.a".to_string(),
        "libqiree.a".to_string(),
    ];

    BuiltQiree {
        rerun_if_changed,
        include_dir,
        lib_dir,
        static_lib_names,
    }
}

fn run_bindgen(built_qiree: BuiltQiree) -> Result<(), Box<dyn Error>> {
    let version = llvm_config("--version")?;

    if !version.starts_with(&format!("{LLVM_MAJOR_VERSION}.",)) {
        return Err(format!(
            "failed to find correct version ({LLVM_MAJOR_VERSION}.x.x) of llvm-config (found {version})"
        )
        .into());
    }

    println!("cargo:rerun-if-changed=wrapper.h");

    println!(
        "cargo:rustc-link-search={}",
        built_qiree.lib_dir.display()
    );
    for static_lib_name in built_qiree.static_lib_names {
        if let Some(name) = parse_archive_name(&static_lib_name) {
            println!("cargo:rustc-link-lib=static={name}");
        }
    }

    println!("cargo:rustc-link-search={}", llvm_config("--libdir")?);

    for name in llvm_config(&format!("--libnames {}", LLVM_COMPONENTS))?.trim().split(' ') {
        if let Some(name) = parse_archive_name(name) {
            println!("cargo:rustc-link-lib={name}");
        }
    }

    for flag in llvm_config("--system-libs")?.trim().split(' ') {
        let flag = flag.trim_start_matches("-l");

        if flag.starts_with('/') {
            // llvm-config returns absolute paths for dynamically linked libraries.
            let path = Path::new(flag);

            println!(
                "cargo:rustc-link-search={}",
                path.parent().unwrap().display()
            );
            println!(
                "cargo:rustc-link-lib={}",
                path.file_stem()
                    .unwrap()
                    .to_str()
                    .unwrap()
                    .trim_start_matches("lib")
            );
        } else {
            println!("cargo:rustc-link-lib={flag}");
        }
    }

    if let Some(name) = get_system_libcpp() {
        println!("cargo:rustc-link-lib={name}");
    }

    bindgen::builder()
        .header("wrapper.h")
        .clang_args(vec![
            format!("-I{}", llvm_config("--includedir")?),
            format!("-I{}", built_qiree.include_dir.display()),
        ])
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .unwrap()
        .write_to_file(Path::new(&env::var("OUT_DIR")?).join("bindings.rs"))?;

    Ok(())
}

fn get_system_libcpp() -> Option<&'static str> {
    if cfg!(target_env = "msvc") {
        None
    } else if cfg!(target_os = "macos") {
        Some("c++")
    } else {
        Some("stdc++")
    }
}

fn llvm_config(argument: &str) -> Result<String, Box<dyn Error>> {
    let prefix = env::var(format!("MLIR_SYS_{LLVM_MAJOR_VERSION}0_PREFIX"))
        .map(|path| Path::new(&path).join("bin"))
        .unwrap_or_default();
    let llvm_config_exe = if cfg!(target_os = "windows") {
        "llvm-config.exe"
    } else {
        "llvm-config"
    };

    let call = format!(
        "{} --link-static {argument}",
        prefix.join(llvm_config_exe).display(),
    );

    Ok(str::from_utf8(
        &if cfg!(target_os = "windows") {
            Command::new("cmd").args(["/C", &call]).output()?
        } else {
            Command::new("sh").arg("-c").arg(&call).output()?
        }
        .stdout,
    )?
    .trim()
    .to_string())
}

fn parse_archive_name(name: &str) -> Option<&str> {
    if let Some(name) = name.strip_prefix("lib") {
        name.strip_suffix(".a")
    } else {
        None
    }
}
