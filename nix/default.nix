{
  src,
  lib,
  buildPythonPackage,
  scikitlearn,
  numpy,
  sentencepiece,
  pytorch-bin,
  hydra,
  boto3,
  mlflow-skinny,
  faiss,
  pytest,
  with-training ? false,
  with-eval ? false

}:

buildPythonPackage {
  pname = "doc_enc";
  version = "0.0.8";
  inherit src;


  buildInputs = [];
  propagatedBuildInputs=[
    scikitlearn
    numpy
    pytorch-bin
    sentencepiece
    hydra ]
  ++ lib.optionals with-training [
    boto3
    mlflow-skinny ]
  ++ lib.optionals with-eval [
    mlflow-skinny
    faiss
  ];

 checkInputs = [pytest];

 checkPhase = "pytest";
}
