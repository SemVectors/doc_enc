{ self, pkgs }:
pyfinal: pyprev: {

  torch = (pyprev.torch-bin.override {cudaPackages = pkgs.cudaPackages_11;}).overridePythonAttrs(
    old: {
      src = pkgs.fetchurl
        {
          name = "torch-2.7.1-cu118-cp312-cp312-linux_x86_64.whl";
          url = "https://download.pytorch.org/whl/cu118/torch-2.7.1%2Bcu118-cp312-cp312-manylinux_2_28_x86_64.whl";
          sha256 = "91454dcfdb81f181fdf216d6d6d9912fbd8795578b90384b3b8b8132737072bb";
        };
    });
  doc_enc = pyfinal.callPackage ./default.nix {
    src=self;
    torch = pyfinal.torch;
  };
}
