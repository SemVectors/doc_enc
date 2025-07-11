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
      pypkgs = pkgs.pythonPackages;
    in {
      overlays.default = import ./nix/overlay.nix {inherit self pkgs;};

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
            pkgs.pyright
            pkgs.nodePackages.bash-language-server
            pkgs.shellcheck
            pkgs.yamllint
            pkgs.ruff
            #pypkgs.pylint
            pypkgs.black
            # pypkgs.debugpy
            pypkgs.ipykernel
          ];

          shellHook=''
            #https://github.com/NixOS/nixpkgs/issues/11390
            export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/nvidia/current/:$LD_LIBRARY_PATH
            '';
        };
    };

}
