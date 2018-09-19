#!/usr/bin/env bash

ROBOSCHOOL_PATH=$(realpath ./roboschool)

if [ ! -d $ROBOSCHOOL_PATH ]; then
  git submodule update --init --recusrive
fi

mkdir -p bullet3/build
cd    bullet3/build
cmake -DBUILD_SHARED_LIBS=ON -DUSE_DOUBLE_PRECISION=1 -DCMAKE_INSTALL_PREFIX:PATH=$ROBOSCHOOL_PATH/roboschool/cpp-household/bullet_local_install -DBUILD_CPU_DEMOS=OFF -DBUILD_BULLET2_DEMOS=OFF -DBUILD_EXTRAS=OFF  -DBUILD_UNIT_TESTS=OFF -DBUILD_CLSOCKET=OFF -DBUILD_ENET=OFF -DBUILD_OPENGL3_DEMOS=OFF ..
make -j"$(nproc)"
make install

ln -vsr robot_models $ROBOSCHOOL_PATH/roboschool/models_robot/robot_models

wget -O robot_models/yamax.urdf https://github.com/Y-modify/YamaX/releases/download/4.0/YamaX_4.0.urdf

pipenv install
