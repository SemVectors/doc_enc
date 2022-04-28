{
  src,
  buildPythonPackage,
  scikitlearn,
  numpy,
  sentencepiece,
  pytorch-bin,
  hydra,
  pytest,

}:

buildPythonPackage {
  pname = "doc_enc";
  version = "0.0.1";
  inherit src;


  buildInputs = [];
  propagatedBuildInputs=[scikitlearn numpy pytorch-bin sentencepiece hydra];
  checkInputs = [pytest];

  checkPhase = "pytest";
}
