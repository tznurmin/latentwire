{
  description = "LatentWire reference experiment environment";
  inputs.nixpkgs.url = "github:NixOS/nixpkgs/e7439b6b14ad3cc35d05608ebca9bce01a25f5f8";
  outputs = { nixpkgs, ... }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs {
        inherit system;
        config = { allowUnfree = true; cudaSupport = true; };
      };
      runtime = pkgs.python313.withPackages (ps: [
        ps.torchWithCuda ps.torchvision ps.transformers ps.numpy
      ]);
    in {
      packages.${system}.default = runtime;
      devShells.${system}.default = pkgs.mkShellNoCC {
        packages = [ runtime pkgs.coreutils ];
        HF_HUB_OFFLINE = "1";
        TRANSFORMERS_OFFLINE = "1";
        USE_HUB_KERNELS = "0";
        TOKENIZERS_PARALLELISM = "false";
        PYTHONDONTWRITEBYTECODE = "1";
        OMP_NUM_THREADS = "1";
        OPENBLAS_NUM_THREADS = "1";
        MKL_NUM_THREADS = "1";
        NUMEXPR_NUM_THREADS = "1";
        VECLIB_MAXIMUM_THREADS = "1";
        BLIS_NUM_THREADS = "1";
      };
    };
}
