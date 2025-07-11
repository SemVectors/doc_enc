{
  src,
  lib,
  buildPythonPackage,
  setuptools,
  scikit-learn,
  numpy,
  sentencepiece,
  torch,
  hydra-core,
  transformers,
  sentence-transformers,
  faiss,
  pytest,
  with-training-pkgs ? false,
}:

buildPythonPackage {
  pname = "doc_enc";
  version = "0.2.0";
  inherit src;
  pyproject = true;
  build-system = [ setuptools ];


  buildInputs = [];
  propagatedBuildInputs=[
    scikit-learn
    numpy
    torch
    sentencepiece
    hydra-core
    transformers
    sentence-transformers
  ]
  ++ lib.optionals with-training-pkgs [
    faiss
  ]
  ;

  nativeCheckInputs = [pytest];

  checkPhase = "pytest";
}
