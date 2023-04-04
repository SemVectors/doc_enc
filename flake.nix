{
  description = "Encoding texts as dense vectors";

  inputs.nixpkgs.url = "nixpkgs";

  outputs = { self, nixpkgs }:
    let pkgs = import nixpkgs {
          system = "x86_64-linux";
          overlays = [ self.overlays.default ];
          config = {allowUnfree = true;};
        };
        python-overlay = pyfinal: pyprev: {
          doc_enc = pyfinal.callPackage ./nix {
            src=self;
          };
          doc_enc_full = pyfinal.callPackage ./nix {
            src=self;
            with-eval=true;
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
        default = pypkgs.doc_enc_full;
      };

      trainDockerImage =  import ./nix/docker.nix {inherit pkgs;
                                                   doc-enc-pkg = pypkgs.doc_enc_full;
                                                   name-suffix="_train";};
      inferenceDockerImage = import ./nix/docker.nix {inherit pkgs;
                                                      doc-enc-pkg = pypkgs.doc_enc;};

      devShells.x86_64-linux.default =
        pkgs.mkShell {
          inputsFrom = [ pypkgs.doc_enc_full ];
          buildInputs = [
            pkgs.nodePackages.pyright
            pkgs.nodePackages.bash-language-server
            pkgs.shellcheck
            pkgs.yamllint
            pypkgs.pylint
            pypkgs.black
            pypkgs.debugpy
            pypkgs.ipykernel
          ];

          shellHook=''
            #https://github.com/NixOS/nixpkgs/issues/11390
            export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/nvidia/current/:$LD_LIBRARY_PATH
            '';
        };
    };

}
