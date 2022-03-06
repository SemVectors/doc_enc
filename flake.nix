{
  description = "Encoding texts as dense vectors";

  inputs.nixpkgs.url = "nixpkgs";

  outputs = { self, nixpkgs }:
    let pkgs = import nixpkgs {
          system = "x86_64-linux";
          overlays = [ self.overlay ];
          config = {allowUnfree = true;};
        };
        python-overlay = pyfinal: pyprev: {doc_enc = pyfinal.callPackage ./nix {src=self;};};
        overridePython = py-overlay: final: prev: (
          prev.python39.override (old: {
            packageOverrides = final.lib.composeExtensions
              (old.packageOverrides or (_: _: { }))
              py-overlay;
          })
        );
    in {
      overlay = final: prev: {python = overridePython python-overlay final prev;};

      packages.x86_64-linux = {
        inherit (pkgs)
          python;
      };

      defaultPackage.x86_64-linux = pkgs.python.pkgs.doc_enc;
      trainDockerImage = pkgs.dockerTools.streamLayeredImage {
        name = "tsa04.isa.ru:5050/semvectors/doc_enc/train";
        tag = pkgs.python.pkgs.doc_enc.version;

        contents = [pkgs.bashInteractive pkgs.coreutils pkgs.glibc pkgs.glibc.bin ];
        # contents = [pkgs.bashInteractive pkgs.coreutils ];
        #paths should be relative in extraCommands
        # extraCommands = '' ln -s lib lib/x86_64-linux-gnu '';
        config = {

          Entrypoint = [ "${pkgs.python.pkgs.doc_enc}/bin/run_training" ];

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
            inputsFrom = [ pypkgs.doc_enc ];
            buildInputs = [
              pkgs.nodePackages.pyright
              pkgs.nodePackages.bash-language-server
              pkgs.shellcheck
              pypkgs.pylint
              pypkgs.black
              pypkgs.debugpy
            ];

            shellHook=''
                        #https://github.com/NixOS/nixpkgs/issues/11390
                        LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/nvidia/current/:$LD_LIBRARY_PATH
            '';
          };
    };

}
