{
  src,
  buildPythonPackage,
  scikitlearn,
  numpy,
  sentencepiece,
  pytorch-bin,
  hydra,
  boto3,
  mlflow-skinny,
  pytest,

}:

buildPythonPackage {
  pname = "doc_enc";
  version = "0.0.7";
  inherit src;


  buildInputs = [];
  propagatedBuildInputs=[scikitlearn numpy pytorch-bin sentencepiece hydra boto3 mlflow-skinny];
  checkInputs = [pytest];

  checkPhase = "pytest";
}
