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
  with-eval ? false,
}:

buildPythonPackage {
  pname = "doc_enc";
  version = "0.0.10";
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
   ++ lib.optionals with-eval [
     faiss
   ]
  ;

  checkInputs = [pytest];

  checkPhase = "pytest";
}
