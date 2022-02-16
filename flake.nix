{
  description = "Encoding texts as dense vectors";

  # inputs.textapp-pkgs.url = "git+ssh://git@tsa04.isa.ru/textapp/textapp-pkgs?ref=flakes";
  inputs.nixpkgs.url = "nixpkgs";

  outputs = { self, nixpkgs }:
    let pkgs = import nixpkgs {
          system = "x86_64-linux";
          # overlays = [ self.overlay ];
          config = {allowUnfree = true;};
        };
        # python-overlay = pyfinal: pyprev: {doc_enc = pyfinal.callPackage ./nix {src=self;};};
    in {
      # overlay = final: prev: {python = textapp-pkgs.lib.overridePython python-overlay final prev;};

      # packages.x86_64-linux = {
      #   inherit (pkgs)
      #     python;
      # };

      # defaultPackage.x86_64-linux = pkgs.python.pkgs.doc_enc;
      devShell.x86_64-linux =
        let pypkgs = pkgs.python39.pkgs;
            # tpkgs = textapp-pkgs.packages.x86_64-linux;
        in
          pkgs.mkShell {
            inputsFrom = [ (pypkgs.callPackage ./nix {src=self;}) ];
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
            LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/nvidia/current/:$LD_LIBRARY_PATH'';
          };
    };

}
