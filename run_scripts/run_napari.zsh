#!/usr/bin/env zsh

repo_dir="$HOME/repos/napari-neuron-navigator"

if [[ ! -d "$repo_dir" ]]; then
    echo "Could not find the repository at:"
    echo "  $repo_dir"
    echo
    echo "Clone the repository to ~/repos/napari-neuron-navigator, or edit this script to use your actual path."
    echo
    printf "Press Return to close this window..."
    read -r reply
    exit 1
fi

cd "$repo_dir" || exit 1

pixi run napari
status=$?

echo
if [[ "$status" -eq 0 ]]; then
    echo "napari has closed."
else
    echo "pixi run napari exited with status $status."
fi

printf "Press Return to close this window..."
read -r reply
exit "$status"
