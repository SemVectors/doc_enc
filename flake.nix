{
  description = "Encoding texts as dense vectors";

  inputs.nixpkgs.url = "nixpkgs/9b008d60392981ad674e04016d25619281550a9d";

  outputs = { self, nixpkgs }:
    let
      pkgs = import nixpkgs {
        system = "x86_64-linux";
        overlays = [ self.overlays.default ];
        config = {
          allowUnfree = true;
          cudaSupport = true;
        };
      };
      pypkgs = pkgs.python3Packages;
      pycu11pkgs = pkgs.pycu11.pkgs;
    in {
      overlays = {
        default = import ./nix/overlay.nix {inherit self pkgs;};
        cu11 = import ./nix/overlay-cu11.nix {inherit self pkgs;};
      };

      packages.x86_64-linux = {
        default = pypkgs.doc_enc;
        doc_enc_train = pypkgs.doc_enc_train;
        doc_enc_cu11 = pycu11pkgs.doc_enc;
      };

      trainDockerImage =  import ./nix/docker.nix {inherit pkgs;
                                                   doc-enc-pkg = pypkgs.doc_enc_train;
                                                   version=pypkgs.doc_enc_train.version;
                                                   name-suffix="_train";};
      inferenceDockerImage = import ./nix/docker.nix {inherit pkgs;
                                                      doc-enc-pkg = pypkgs.doc_enc;
                                                      version=pypkgs.doc_enc.version;};

      devShells.x86_64-linux =
        let python-dev-hook =
              ''
              #execute this hook only for interactive sessions
              if [ -n "$PS1" ]; then
                echo "Executing pythonDevHook"

                eval "${pkgs.python3.interpreter} -m pip install -e . --prefix .dev --no-build-isolation >&2"
                export PYTHONPATH=$(pwd)/.dev/${pkgs.python3.sitePackages}:$PYTHONPATH
                export PATH=$(pwd)/.dev/bin:$PATH
                export NIX_PYTHONPATH="$(pwd)/.dev/${pkgs.python3.sitePackages}:$NIX_PYTHONPATH"
              fi
              '';
            common-hook = ''
            #https://github.com/NixOS/nixpkgs/issues/11390
            export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/nvidia/current/:$LD_LIBRARY_PATH
            '';
        in {
          
          default = pkgs.mkShell {
            inputsFrom = [ pypkgs.doc_enc_train ];
            buildInputs = [
              pkgs.pyright
              pkgs.nodePackages.bash-language-server
              pkgs.shellcheck
              pkgs.yamllint
              pkgs.ruff
              #pypkgs.pylint
              pypkgs.black
              pypkgs.pip
              # pypkgs.debugpy
              pypkgs.ipykernel
            ];

            shellHook= python-dev-hook + common-hook;
          };
          cu11 = pkgs.mkShell {
            inputsFrom = [ pycu11pkgs.doc_enc ];
            shellHook= python-dev-hook + common-hook;
          };
        };

    };

}
