#!/usr/bin/env bash

set -euo pipefail

function build_bullet3() {
  local roboschool_path=$(realpath ./third_party/roboschool)
  # Bullet installation for roboschool
  mkdir -p third_party/bullet3/build
  pushd    third_party/bullet3/build
  cmake -DBUILD_SHARED_LIBS=ON -DUSE_DOUBLE_PRECISION=1 -DCMAKE_INSTALL_PREFIX:PATH=$roboschool_path/roboschool/cpp-household/bullet_local_install -DBUILD_CPU_DEMOS=OFF -DBUILD_BULLET2_DEMOS=OFF -DBUILD_EXTRAS=OFF  -DBUILD_UNIT_TESTS=OFF -DBUILD_CLSOCKET=OFF -DBUILD_ENET=OFF -DBUILD_OPENGL3_DEMOS=OFF ..
  make -j"$(nproc)"
  make install
  popd
}

function install_without_roboschool() {
  mv Pipfile Pipfile.bak
  sed '/roboschool = /d' Pipfile.bak > Pipfile
  pipenv install --skip-lock
  mv Pipfile.bak Pipfile
}

function build_boost_python() {
  pushd third_party/boost-python
  pipenv run faber
  popd
}

function place_robot_model() {
  local models_path=third_party/roboschool/roboschool/models_robot/robot_models
  # Include out model to the search path
  if [ ! -e $models_path ]; then
    ln -vsr robot_models $models_path
  fi

  # Download and patch YamaX model
  wget -O robot_models/yamax.urdf https://github.com/Y-modify/YamaX/releases/download/v6.0.1/YamaX_v6.0.1.urdf
  patch robot_models/yamax.urdf < yamax.urdf.patch
}

function install_all() {
  local boost_python_lib="$(realpath $(find third_party/boost-python -name '*boost_python*.so'))"
  local boost_python_include="$(realpath third_party/boost-python/include)"

  # Roboschool installation needs to be done in virtualenv
  local python_version=$(pipenv run python -V | awk '{print $2}')
  PKG_CONFIG_PATH=$HOME/.pyenv/versions/$python_version/lib/pkgconfig CPLUS_INCLUDE_PATH=$boost_python_include LIBRARY_PATH="$(dirname $boost_python_lib)" pipenv run pipenv install
}

build_bullet3
install_without_roboschool
build_boost_python
place_robot_model
install_all
