#!/usr/bin/env -S nix eval -I nixpkgs=flake:nixpkgs --show-trace --file

let
  pkgs = import <nixpkgs> {};
  filterv = f: pkgs.lib.filterAttrs (k: f);
in
  with pkgs.lib; with builtins; attrNames (filterAttrs (k: v:
    let tryV = tryEval v;
    in tryV.success && isAttrs tryV.value && hasAttr "outPath" tryV.value &&
      (let tryPassthru = tryEval v.passthru;
      in tryPassthru.success && hasAttr "goModules" tryPassthru.value)
  ) pkgs)
#  pipe pkgs [
#    (filterv (v: let e = tryEval v; in e.success && isAttrs e.value))
#    # Filter Go binaries
#    (filterv (v: hasAttr "GOOS" v))
#    # Filter unfree
#    (filterv (v: (tryEval v.outPath).success))
#    (filterv (v: (tryEval v.passthru).success))
#    # (filterv (v: hasAttr "goModules" v))
#    # (mapAttrs (k: v: v.passthru.goModules))
#    (filterv (v: hasAttr "vendorHash" v.passthru && v.passthru.vendorHash == null))
#    # (mapAttrs (k: v: v.passthru.vendorHash))
#    # attrValues
#    attrNames
#  ]
