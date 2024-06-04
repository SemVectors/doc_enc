{ self, pkgs }:
pyfinal: pyprev: {
  sentence-transformers = pyprev.sentence-transformers.overridePythonAttrs(old: {
    # torchvision is optional dep, we do not use it, so remove it from the closure
    dependencies = (pkgs.lib.subtractLists
      [pyprev.torchvision]
      old.dependencies
    );
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
