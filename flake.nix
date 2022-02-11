{
  description = "Encoding texts as dense vectors";

  inputs.textapp-pkgs.url = "git+ssh://git@tsa04.isa.ru/textapp/textapp-pkgs?ref=flakes";

  outputs = { self, textapp-pkgs }:
    let pkgs = import textapp-pkgs.inputs.nixpkgs {
          system = "x86_64-linux";
          overlays = [ textapp-pkgs.overlay self.overlay ];
          config = {allowUnfree = true;};
        };
        python-overlay = pyfinal: pyprev: {doc_enc = pyfinal.callPackage ./nix {src=self;};};
    in {
      overlay = final: prev: {python = textapp-pkgs.lib.overridePython python-overlay final prev;};

      packages.x86_64-linux = {
        inherit (pkgs)
          python;
      };

      defaultPackage.x86_64-linux = pkgs.python.pkgs.doc_enc;
      devShell.x86_64-linux =
        let pypkgs = pkgs.python.pkgs;
            tpkgs = textapp-pkgs.packages.x86_64-linux;
        in
          pkgs.mkShell {
            inputsFrom = [ pypkgs.doc_enc ];
            buildInputs = [
              tpkgs.pyright
              tpkgs.bash-language-server
              tpkgs.shellcheck
              pypkgs.pylint
              pypkgs.black
            ];

          };
    };

}
