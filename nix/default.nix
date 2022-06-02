{
  src,
  lib,
  buildPythonPackage,
  scikitlearn,
  numpy,
  sentencepiece,
  pytorch-bin,
  hydra,
  transformers,
  boto3,
  mlflow-skinny,
  faiss,
  pytest,
  with-training ? false,
  with-eval ? false

}:

buildPythonPackage {
  pname = "doc_enc";
  version = "0.0.9";
  inherit src;


  buildInputs = [];
  propagatedBuildInputs=[
    scikitlearn
    numpy
    pytorch-bin
    sentencepiece
    hydra
    transformers ]
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
