{
  src,
  buildPythonPackage,
  scikitlearn,
  numpy,
  sentencepiece,
  pytorch-bin,
  hydra,
  mlflow-skinny,
  pytest,

}:

buildPythonPackage {
  pname = "doc_enc";
  version = "0.0.6";
  inherit src;


  buildInputs = [];
  propagatedBuildInputs=[scikitlearn numpy pytorch-bin sentencepiece hydra mlflow-skinny];
  checkInputs = [pytest];

  checkPhase = "pytest";
}
