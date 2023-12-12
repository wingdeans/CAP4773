#!/usr/bin/env -S nix eval -I nixpkgs=flake:nixpkgs --show-trace --file

let
  pkgs = import <nixpkgs> {};
  mapv = f: pkgs.lib.mapAttrs (k: f);
  filterv = f: pkgs.lib.filterAttrs (k: f);
in
  with pkgs.lib; with builtins;
  pipe pkgs [
    (mapv tryEval)
    (filterv (e: e.success && isAttrs e.value))
    # Filter Go binaries
    (mapv (e: e.value))
    (filterv (e: hasAttr "GOOS" e))
    # Extract
    (filterv (e: (hasAttr "modRoot" e) || (hasAttr "sourceRoot" e)))
    (mapv (e: if hasAttr "modRoot" e then e.modRoot else removePrefix "source/" e.sourceRoot))
    toJSON
  ]
