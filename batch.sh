#!/usr/bin/env -S ./nix-user-chroot ./nix bash
source ./devrc

echo "Directory: $1"
pushd "$1"
shift

echo "$@"
echo "Running"
eval "$@"
echo "Done"

popd
