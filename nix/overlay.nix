{self, pkgs}:
let py-overlay = import ./python-overlay.nix {inherit self pkgs;};
in
final: prev: {
  pythonPackagesExtensions = prev.pythonPackagesExtensions ++ [py-overlay];

  pycu11 = prev.python312 // {
    pkgs = prev.python312.pkgs.overrideScope (
      import ./python-cu11-overlay.nix {inherit self pkgs;}
    );
  };
}
