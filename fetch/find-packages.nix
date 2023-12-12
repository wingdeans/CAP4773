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
    # Filter unfree
    # (mapv (e: tryEval e.outPath))
    # (filterv (e: e.success))
    (filterv (e: (tryEval e.outPath).success))
    (mapv (e: tryEval e.passthru))
    (filterv (e: e.success))
    (mapv (e: e.value))
    (filterv (e: hasAttr "goModules" e))
    attrNames
  ]
