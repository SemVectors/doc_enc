{
  src,
  buildPythonPackage,
  scikitlearn,
  numpy,
  sentencepiece,
  pytorch
}:

buildPythonPackage {
  pname = "doc_enc";
  version = "0.0.0";
  inherit src;


  buildInputs = [];
  propagatedBuildInputs=[scikitlearn numpy pytorch sentencepiece];

}
