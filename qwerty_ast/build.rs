fn main() {
    let dst = cmake::Config::new("..").generator("Ninja").build();
    println!("dest dir: {}", dst.display());
    println!("cargo::rerun-if-changed=../CMakeLists.txt");
    println!("cargo::rerun-if-changed=../qwerty_mlir/");
    println!("cargo::rerun-if-changed=../qwerty_util/");
}
