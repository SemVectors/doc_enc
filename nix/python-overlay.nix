{ self, pkgs }:
pyfinal: pyprev: {
  sentence-transformers = pyprev.sentence-transformers.overridePythonAttrs(old: {
    # we use pytorch-bin do not pull full torch distribution
    propagatedBuildInputs = (pkgs.lib.subtractLists
      [pyprev.torch pyprev.torchvision]
      old.propagatedBuildInputs
    ) ++ [pyprev.torch-bin pyprev.pillow];
    prePatch = ''
      substituteInPlace setup.py \
      --replace "'torchvision'," ""
    '';
  });
  doc_enc = pyfinal.callPackage ./default.nix {
    src=self;
  } ;
  doc_enc_train = pyfinal.callPackage ./default.nix {
    src=self;
    with-training-pkgs=true;
  };
}
