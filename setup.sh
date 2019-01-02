#!/usr/bin/env bash

set -euo pipefail

# Bullet installation for roboschool
roboschool_path=$(realpath ./third_party/roboschool)
mkdir -p third_party/bullet3/build
pushd    third_party/bullet3/build
cmake -DBUILD_SHARED_LIBS=ON -DUSE_DOUBLE_PRECISION=1 -DCMAKE_INSTALL_PREFIX:PATH=$roboschool_path/roboschool/cpp-household/bullet_local_install -DBUILD_CPU_DEMOS=OFF -DBUILD_BULLET2_DEMOS=OFF -DBUILD_EXTRAS=OFF  -DBUILD_UNIT_TESTS=OFF -DBUILD_CLSOCKET=OFF -DBUILD_ENET=OFF -DBUILD_OPENGL3_DEMOS=OFF ..  make -j"$(nproc)" make install
popd

mv Pipfile Pipfile.bak
sed '/roboschool = /d' Pipfile.bak > Pipfile
pipenv install --skip-lock
mv Pipfile.bak Pipfile

pushd third_party/boost-python
pipenv run faber
popd

boost_python_lib="$(realpath $(find third_party/boost-python -name '*boost_python*.so'))"
boost_python_include="$(realpath third_party/boost-python/include)"

# Include out model to the search path
if [ ! -e $roboschool_path/roboschool/models_robot/robot_models ]; then
  ln -vsr robot_models $roboschool_path/roboschool/models_robot/robot_models
fi

# Download and patch YamaX model
wget -O robot_models/yamax.urdf https://github.com/Y-modify/YamaX/releases/download/v6.0.1/YamaX_v6.0.1.urdf
patch robot_models/yamax.urdf < yamax.urdf.patch

# Roboschool installation needs to be done in virtualenv
PYTHON_VERSION=$(pipenv run python -V | awk '{print $2}')
PKG_CONFIG_PATH=$HOME/.pyenv/versions/$PYTHON_VERSION/lib/pkgconfig CPLUS_INCLUDE_PATH=$boost_python_include LIBRARY_PATH="$(dirname $boost_python_lib)" pipenv run pipenv install
