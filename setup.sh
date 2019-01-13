#!/usr/bin/env bash

set -euo pipefail

function place_robot_model() {
  local models_path=third_party/roboschool/roboschool/models_robot/robot_models
  # Include out model to the search path
  if [ ! -h $models_path ]; then
    ln -vsr robot_models $models_path
  fi

  # Download and patch YamaX model
  wget -O robot_models/yamax.urdf https://github.com/Y-modify/YamaX/releases/download/v6.0.1/YamaX_v6.0.1.urdf
  patch robot_models/yamax.urdf < yamax.urdf.patch
}

function install_all() {
  place_robot_model
  install
}

function main() {
  if [ "$#" == "0" ]; then
    install_all
  else
    $@
  fi
}

main $@
