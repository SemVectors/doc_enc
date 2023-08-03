{
  description = "Encoding texts as dense vectors";

  inputs.nixpkgs.url = "nixpkgs/d0c7a8f1c9a2ebfebd3d99960cdb7c4eec442dc9";

  outputs = { self, nixpkgs }:
    let cuda-overlay = final: prev:{
          #enable cuda support
          nvidia-thrust = prev.nvidia-thrust.override{deviceSystem="CUDA";};
          faiss = prev.faiss.override{cudaSupport=true;};
        };
        pkgs = import nixpkgs {
          system = "x86_64-linux";
          overlays = [ cuda-overlay self.overlays.default ];
          config = {allowUnfree = true;};
          };
        pypkgs = pkgs.python310Packages;
    in {
      overlays.default = final: prev: {
        python310Packages = prev.python310Packages.overrideScope (
          import ./nix/python-overlay.nix {inherit self pkgs;}
        );
        python = prev.python310;
      };

      packages.x86_64-linux = {
        default = pypkgs.doc_enc;
        train = pypkgs.doc_enc_train;
      };

      trainDockerImage =  import ./nix/docker.nix {inherit pkgs;
                                                   doc-enc-pkg = pypkgs.doc_enc_train;
                                                   version=pypkgs.doc_enc_train.version;
                                                   name-suffix="_train";};
      inferenceDockerImage = import ./nix/docker.nix {inherit pkgs;
                                                      doc-enc-pkg = pypkgs.doc_enc;
                                                      version=pypkgs.doc_enc.version;};

      devShells.x86_64-linux.default =
        pkgs.mkShell {
          inputsFrom = [ pypkgs.doc_enc_train ];
          buildInputs = [
            pkgs.nodePackages.pyright
            pkgs.nodePackages.bash-language-server
            pkgs.shellcheck
            pkgs.yamllint
            pypkgs.ruff-lsp
            #pypkgs.pylint
            pypkgs.black
            # pypkgs.debugpy
            pypkgs.ipykernel
          ];

          shellHook=''
            #https://github.com/NixOS/nixpkgs/issues/11390
            export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/nvidia/current/:$LD_LIBRARY_PATH
            [ -z "$PS1" ] || setuptoolsShellHook
            '';
        };
    };

}
