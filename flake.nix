{
  description = "Encoding texts as dense vectors";

  inputs.nixpkgs.url = "nixpkgs/d0c7a8f1c9a2ebfebd3d99960cdb7c4eec442dc9";

  outputs = { self, nixpkgs }:
    let cuda-overlay = final: prev:{
          #enable cuda support
          nvidia-thrust = prev.nvidia-thrust.override{deviceSystem="CUDA";};
          faiss = prev.faiss.override{cudaSupport=true;};
        } ;
        pkgs = import nixpkgs {
          system = "x86_64-linux";
          overlays = [ cuda-overlay self.overlays.default ];
          config = {allowUnfree = true;};
        };
        python-overlay = pyfinal: pyprev: {
          doc_enc = pyfinal.callPackage ./nix {
            src=self;
          };
          doc_enc_train = pyfinal.callPackage ./nix {
            src=self;
            with-training-pkgs=true;
          };
        };
        overridePython = py-overlay: final: prev: (
          prev.python310.override (old: {
            self = pkgs.python;
            packageOverrides = final.lib.composeExtensions
              (old.packageOverrides or (_: _: { }))
              py-overlay;
          })
        );
        pypkgs = pkgs.python.pkgs;
    in {
      overlays.default = final: prev: {
        python = overridePython python-overlay final prev;
      };

      packages.x86_64-linux = {
        inherit (pkgs)
          python;
        default = pypkgs.doc_enc;
        train = pypkgs.doc_enc_train;
      };

      trainDockerImage =  import ./nix/docker.nix {inherit pkgs;
                                                   doc-enc-pkg = pypkgs.doc_enc_train;
                                                   name-suffix="_train";};
      inferenceDockerImage = import ./nix/docker.nix {inherit pkgs;
                                                      doc-enc-pkg = pypkgs.doc_enc;};

      devShells.x86_64-linux.default =
        pkgs.mkShell {
          inputsFrom = [ pypkgs.doc_enc_train ];
          buildInputs = [
            pkgs.enchant
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
