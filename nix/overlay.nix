{self, pkgs}:
final: prev: {
  python = prev.python311 // {
    pkgs = prev.python311.pkgs.overrideScope (
      import ./python-overlay.nix {inherit self pkgs;}
    );
  };
  pythonPackages = final.python.pkgs;


  faiss = prev.faiss.overrideAttrs(old:
    #decrease closure size of faiss package;
    #strip static libraries from dependencies
    #We need to redefine cudaJoined dependency from original faiss derivation
    let cudaJoined = final.symlinkJoin {
          name = "cuda-packages-unsplit";
          paths = with final.cudaPackages; [
            cuda_cudart.dev # cuda_runtime.h and libraries
            cuda_cudart.lib
            cuda_cudart.static
            libcublas.lib
            libcublas.dev
            libcurand.dev
            libcurand.lib
            cuda_cccl.dev
            cuda_profiler_api
            cuda_nvprof
          ];
        };
    in
      {
        buildInputs = (final.lib.init old.buildInputs) ++ [cudaJoined];
        cmakeFlags = (final.lib.init old.buildInputs) ++ [
          "-DCUDAToolkit_INCLUDE_DIR=${cudaJoined}/include"
        ];
      });
}
