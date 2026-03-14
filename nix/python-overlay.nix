{ self, pkgs }:
pyfinal: pyprev: {
  flash-attn = pyfinal.callPackage ./flash-attn.nix {};

  doc_enc = pyfinal.callPackage ./default.nix {
    src=self;
  } ;
  doc_enc_train = pyfinal.callPackage ./default.nix {
    src=self;
    with-training-pkgs=true;
  };
}
