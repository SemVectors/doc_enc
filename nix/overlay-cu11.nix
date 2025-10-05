{self, pkgs}:
final: prev: {
  python312 = prev.python312 // {
    pkgs = prev.python312.pkgs.overrideScope (
      import ./python-cu11-overlay.nix {inherit self pkgs;}
    );
  };
}
