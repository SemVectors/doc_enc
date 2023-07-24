{
  src,
  lib,
  buildPythonPackage,
  scikitlearn,
  numpy,
  sentencepiece,
  pytorch-bin,
  hydra-core,
  transformers,
  faiss,
  pytest,
  with-training-pkgs ? false,
}:

buildPythonPackage {
  pname = "doc_enc";
  version = "0.1.0";
  inherit src;


  buildInputs = [];
  propagatedBuildInputs=[
    scikitlearn
    numpy
    pytorch-bin
    sentencepiece
    hydra-core
    transformers
  ]
  ++ lib.optionals with-training-pkgs [
    faiss
   ]
  ;

  nativeCheckInputs = [pytest];

  checkPhase = "pytest";
}
