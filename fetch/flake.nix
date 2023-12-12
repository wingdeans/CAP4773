{
  inputs.nixpkgs.url = "github:nixos/nixpkgs/nixpkgs-unstable";
  outputs = { self, nixpkgs }: {
    legacyPackages."x86_64-linux" = nixpkgs.legacyPackages."x86_64-linux".extend (final: prev:
      let
        useGccGo = buildGoModule: args: (buildGoModule args).overrideAttrs (old: {
          nativeBuildInputs = (builtins.filter (e:
            !builtins.hasAttr "pname" e || e.pname != "go"
          ) old.nativeBuildInputs) ++ [ prev.gccgo ];
          dontStrip = true;
        });
      in {
        buildGoModule = useGccGo prev.buildGoModule;
        buildGo118Module = useGccGo prev.buildGo118Module;
        buildGo119Module = useGccGo prev.buildGo119Module;
        buildGo120Module = useGccGo prev.buildGo120Module;
        buildGo121Module = useGccGo prev.buildGo121Module;
      }
    );
  };
}
