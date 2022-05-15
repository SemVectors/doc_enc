{ lib,
  fetchFromGitHub,
  stdenv,
  cmake,
  blas,
  swig,
  python,
  setuptools,
  pip,
  wheel,
  numpy,
  optLevel ? let
    optLevels =
      lib.optional stdenv.hostPlatform.avx2Support "avx2"
      ++ lib.optional stdenv.hostPlatform.sse4_1Support "sse4"
      ++ [ "generic" ];
  in
    # Choose the maximum available optimization level
    builtins.head optLevels
}:

let
  pname = "faiss";
  version = "1.7.2";
in
stdenv.mkDerivation {
  inherit pname version;

  outputs = [ "out" ];

  src = fetchFromGitHub {
    owner = "facebookresearch";
    repo = pname;
    rev = "v${version}";
    hash = "sha256-Tklf5AaqJbOs9qtBZVcxXPLAp+K54EViZLSOvEhmswg=";
  };

  buildInputs = [
    blas
    swig
    setuptools
    pip
    wheel
  ];

  propagatedBuildInputs =  [
    numpy
  ];

  nativeBuildInputs = [
    cmake
    python
  ];

  passthru.extra-requires.all = [
    numpy
  ];

  cmakeFlags = [
    "-DFAISS_ENABLE_GPU=OFF"
    "-DFAISS_ENABLE_PYTHON=ON"
    "-DFAISS_OPT_LEVEL=${optLevel}"
  ];


  # pip wheel->pip install commands copied over from opencv4

  buildPhase = ''
    make -j faiss
    make -j swigfaiss
    (cd faiss/python &&
     python -m pip wheel --verbose --no-index --no-deps --no-clean --no-build-isolation --wheel-dir dist .)
  '';

  installPhase = ''
    make install
    mkdir -p $out/${python.sitePackages}
    (cd faiss/python && python -m pip install dist/*.whl --no-index --no-warn-script-location --prefix="$out" --no-cache)
  '';


  meta = with lib; {
    description = "A library for efficient similarity search and clustering of dense vectors by Facebook Research";
    homepage = "https://github.com/facebookresearch/faiss";
    license = licenses.mit;
    platforms = platforms.unix; # Never tested windows
    maintainers = with maintainers; [ SomeoneSerge ];
  };
}
