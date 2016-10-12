with import <nixpkgs> {};
with pkgs.python27Packages;

rec {
  lasagne = buildPythonPackage rec {
      name = "Lasagne-0.1";

      src = pkgs.fetchurl rec {
        sha256 = "0cqj86rdm6c7y5vq3i13qy76fg5xi3yjp4r0hpqy8hvynv54wqrw";
        url = "https://pypi.python.org/packages/98/bf/4b2336e4dbc8c8859c4dd81b1cff18eef2066b4973a1bd2b0ca2e5435f35/${name}.tar.gz";
      };
      propagatedBuildInputs = with self; [
        numpy
      ];
    };

  theano = buildPythonPackage rec {
      name = "Theano-0.8.2";

      src = pkgs.fetchurl rec {
        sha256 = "0c49mz3bg57vigkyfz3yd6302587hsikhvgkh7w7ny0sxpvwhqvl";
        url = "https://pypi.python.org/packages/30/3d/2354fac96ca9594b755ec22d91133522a7db0caa0877165a522337d0ed73/${name}.tar.gz";
      };
      propagatedBuildInputs = with self; [
        six
        scipy
      ];
    };

  rnnEnv = stdenv.mkDerivation rec {
    name = "rnnEnv";
    src = ".";
    propagatedBuildInputs = [
      numpy
      matplotlib
      lasagne
      theano
    ];
  };
}
