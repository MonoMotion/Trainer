#!/usr/bin/env bash

set -eu

# Bullet installation for roboschool
roboschool_path=$(realpath ./roboschool)
mkdir -p bullet3/build
pushd    bullet3/build
cmake -DBUILD_SHARED_LIBS=ON -DUSE_DOUBLE_PRECISION=1 -DCMAKE_INSTALL_PREFIX:PATH=$roboschool_path/roboschool/cpp-household/bullet_local_install -DBUILD_CPU_DEMOS=OFF -DBUILD_BULLET2_DEMOS=OFF -DBUILD_EXTRAS=OFF  -DBUILD_UNIT_TESTS=OFF -DBUILD_CLSOCKET=OFF -DBUILD_ENET=OFF -DBUILD_OPENGL3_DEMOS=OFF ..
make -j"$(nproc)"
make install
popd

# Include out model to the search path
ln -vsr robot_models $roboschool_path/roboschool/models_robot/robot_models

# Download and patch YamaX model
wget -O robot_models/yamax.urdf https://github.com/Y-modify/YamaX/releases/download/v6.0/YamaX_v6.0.urdf

if type "nvidia-smi" > /dev/null 2>&1
then
  sed -i 's/tensorflow =/tensorflow-gpu =/' Pipfile
else
  echo "Using tensorflow without GPU support"
fi

# Roboschool installation needs to be done in virtualenv
if [ -f Pipfile.lock ]; then
  pipenv run pipenv install
else
  pipenv install
  pipenv install -e ./baselines
  pipenv run pipenv install -e ./roboschool
fi
