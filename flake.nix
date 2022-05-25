{
  description = "Encoding texts as dense vectors";

  inputs.nixpkgs.url = "nixpkgs";

  outputs = { self, nixpkgs }:
    let pkgs = import nixpkgs {
          system = "x86_64-linux";
          overlays = [ self.overlay ];
          config = {allowUnfree = true;};
        };
        python-overlay = pyfinal: pyprev: {
          pytorch-bin = pyprev.pytorch-bin.overridePythonAttrs(old: {
            version="1.12.0-rc1";
            src= pkgs.fetchurl {
              name = "torch-1.12.0-cp39-cp39-linux_x86_64.whl";
                url = "https://download.pytorch.org/whl/nightly/cu113/torch-1.12.0.dev20220520%2Bcu113-cp39-cp39-linux_x86_64.whl";
                sha256 = "0bq0q196v45cdzh0z3dg8cfwpn627bq5bnnszry4ybxlr79rhjsb";
            };
          });
          mlflow-skinny = pyfinal.callPackage ./nix/mlflow-skinny.nix {};
          faiss = pyfinal.callPackage ./nix/faiss.nix {swig=pkgs.swig4;};
          doc_enc = pyfinal.callPackage ./nix {
            src=self;
          };
          doc_enc_full = pyfinal.callPackage ./nix {
            src=self;
            with-training=true;
            with-eval=true;
          };
        };
        overridePython = py-overlay: final: prev: (
          prev.python39.override (old: {
            self = pkgs.python;
            packageOverrides = final.lib.composeExtensions
              (old.packageOverrides or (_: _: { }))
              py-overlay;
          })
        );
    in {
      overlay = final: prev: {
        python = overridePython python-overlay final prev;
      };

      packages.x86_64-linux = {
        inherit (pkgs)
          python;
      };

      defaultPackage.x86_64-linux = pkgs.python.pkgs.doc_enc_full;
      trainDockerImage = pkgs.dockerTools.streamLayeredImage {
        name = "tsa04.isa.ru:5050/semvectors/doc_enc/train";
        tag = pkgs.python.pkgs.doc_enc_full.version;

        contents = [
          pkgs.bashInteractive pkgs.coreutils
        ];
        #paths should be relative in extraCommands
        # extraCommands = '' ln -s lib lib/x86_64-linux-gnu '';
        config = {

          Entrypoint = [ "${pkgs.python.pkgs.doc_enc_full}/bin/run_training" ];

          #LD_LIBRARY_PATH for debian11. Also you need to install libnvidia-container1 and libnvidia-container-tools > 1.9.0
          Env = [
            "TRAIN_CONFIG_PATH=/train/conf"
            "NVIDIA_DRIVER_CAPABILITIES=compute,utility"
            "LD_LIBRARY_PATH=/usr/lib64"
          ];


          WorkingDir = "/train";
          Volumes = { "/train" = { }; };
        };
      };

      devShell.x86_64-linux =
        let pypkgs = pkgs.python.pkgs;
        in
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
              pypkgs.jupyter_server
            ];

            shellHook=''
                        #https://github.com/NixOS/nixpkgs/issues/11390
                        export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/nvidia/current/:$LD_LIBRARY_PATH
            '';
          };
    };

}
