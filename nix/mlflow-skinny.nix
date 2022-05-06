# WARNING: This file was automatically generated. You should avoid editing it.
# If you run pynixify again, the file will be either overwritten or
# deleted, and you will lose the changes you made to it.

{
  buildPythonPackage,
  fetchPypi,
  lib,
  GitPython,
  click,
  cloudpickle,
  databricks-cli,
  entrypoints,
  importlib-metadata,
  packaging,
  protobuf,
  pytz,
  pyyaml,
  requests
}:

buildPythonPackage rec {
  pname = "mlflow-skinny";
  version = "1.25.1";

  src = fetchPypi {
    inherit pname version;
    sha256 = "19q1ba7i122qfa62a6z99vf53xb2qnryq9885c8z9p2sbv5dz00w";
  };

  propagatedBuildInputs = [
    click
    cloudpickle
    databricks-cli
    entrypoints
    GitPython
    pyyaml
    protobuf
    pytz
    requests
    packaging
    importlib-metadata
  ];
  MLFLOW_SKINNY="1";

  # TODO FIXME
  doCheck = false;

  meta = with lib; {
    description = "MLflow: A Platform for ML Development and Productionization";
    homepage = "https://mlflow.org/";
  };
}
