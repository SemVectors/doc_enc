{ pkgs, doc-enc-pkg, name-suffix ? "", private-repo ? ""}:
let pypkgs = pkgs.python.pkgs;
    py-env = pkgs.python.buildEnv.override{
      extraLibs = [doc-enc-pkg pypkgs.pip];
    };
in
pkgs.dockerTools.streamLayeredImage {
  name = "${private-repo}semvectors/doc_enc${name-suffix}";
  tag = pypkgs.doc_enc_train.version;

  maxLayers = 110;
  contents = [
    py-env
    pkgs.bashInteractive
    pkgs.coreutils
  ];
  config = {

    #LD_LIBRARY_PATH for debian11. Also you need to install libnvidia-container1 and libnvidia-container-tools > 1.9.0
    Env = [
      "TRAIN_CONFIG_PATH=/train/conf"
      "NVIDIA_DRIVER_CAPABILITIES=compute,utility"
      "LD_LIBRARY_PATH=/usr/lib64"
    ];

    WorkingDir = "/train";
    Volumes = { "/train" = { }; };
  };
}
