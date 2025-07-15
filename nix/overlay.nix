{self, pkgs}:
final: prev: {
  python = prev.python3 // {
    pkgs = prev.python3.pkgs.overrideScope (
      import ./python-overlay.nix {inherit self pkgs;}
    );
  };
  pythonPackages = final.python.pkgs;

  pycu11 = prev.python312 // {
    pkgs = prev.python312.pkgs.overrideScope (
      import ./python-cu11-overlay.nix {inherit self pkgs;}
    );
  };
}
