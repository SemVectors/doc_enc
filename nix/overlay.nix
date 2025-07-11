{self, pkgs}:
final: prev: {
  python = prev.python3 // {
    pkgs = prev.python3.pkgs.overrideScope (
      import ./python-overlay.nix {inherit self pkgs;}
    );
  };
  pythonPackages = final.python.pkgs;
}
