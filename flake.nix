{
  description = "Python devShell with Jupyter, ipdb, and pip dependencies";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.05";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };
        python = pkgs.python311;

        pythonEnv = python.withPackages (ps: with ps; [
          pip
          ipdb
          ipykernel
        ]);

      in {
        devShell = pkgs.mkShell {
          packages = [
            pythonEnv
            pkgs.tree
          ];

          shellHook = ''
            export PYTHONBREAKPOINT=ipdb.set_trace

            if [ ! -d .venv ]; then
              echo "🧪 Creating virtualenv in .venv"
              python -m venv .venv
              source .venv/bin/activate
              pip install --upgrade pip

              if [ -f requirements.txt ]; then
                echo "📦 Installing pip dependencies from requirements.txt"
                pip install -r requirements.txt
              fi

              echo "📌 Registering ipykernel for Jupyter"
              pip install ipykernel
              python -m ipykernel install --user --name=dev-env --display-name "Python (dev-env)"
            else
              source .venv/bin/activate
            fi

            echo "✅ Python dev environment ready"
          '';
        };
      });
}
