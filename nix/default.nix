{
  src,
  lib,
  buildPythonPackage,
  setuptools,
  scikit-learn,
  numpy,
  sentencepiece,
  torch,
  peft,
  hydra-core,
  transformers,
  flash-attn,
  sentence-transformers,
  faiss,
  pytest,
  with-training-pkgs ? false,
}:

buildPythonPackage {
  pname = "doc_enc";
  version = "0.2.1";
  inherit src;
  pyproject = true;
  build-system = [ setuptools ];


  buildInputs = [];
  dependencies=[
    scikit-learn
    numpy
    torch
    peft
    sentencepiece
    hydra-core
    transformers
    flash-attn
    sentence-transformers
  ]
  ++ lib.optionals with-training-pkgs [
    faiss
  ]
  ;

  nativeCheckInputs = [pytest];

  checkPhase = "pytest";
}
