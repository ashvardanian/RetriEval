//! Compiles `src/faiss.cpp` under the FAISS backend, which sets the binary HNSW expansions FAISS's C API
//! leaves unreachable through its C++ types.

fn main() {
    if std::env::var_os("CARGO_FEATURE_FAISS_BACKEND").is_some() {
        println!("cargo:rerun-if-changed=src/faiss.cpp");
        cc::Build::new()
            .cpp(true)
            .std("c++17")
            .file("src/faiss.cpp")
            .compile("retrieval_faiss");
        println!("cargo:rustc-link-lib=faiss");
        // `src/faiss.rs` calls `omp_set_num_threads` itself, which a shared `libfaiss_c` does not re-export.
        println!("cargo:rustc-link-lib=gomp");
    }
}
