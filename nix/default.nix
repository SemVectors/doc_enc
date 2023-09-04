{
  src,
  lib,
  buildPythonPackage,
  scikit-learn,
  numpy,
  sentencepiece,
  pytorch-bin,
  hydra-core,
  transformers,
  sentence-transformers,
  faiss,
  pytest,
  with-training-pkgs ? false,
}:

buildPythonPackage {
  pname = "doc_enc";
  version = "0.1.1";
  inherit src;


  buildInputs = [];
  propagatedBuildInputs=[
    scikit-learn
    numpy
    pytorch-bin
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
