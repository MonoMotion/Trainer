#!/bin/bash
# Deploy Script
# Required Environment Variables:
# $DEEPL2_DISCORD_CHANNEL
# $DEEPL2_DISCORD_TOKEN
# $TF_VAR_access_key
# $TF_VAR_secret_key

echo -e 'y\n' | ssh-keygen -t rsa -b 4096 -C "me@coord-e.com" -N '' -f ~/.ssh/terraform

cat << EOS > terraform.tfvars
key_name = "terraform"
public_key_path = "~/.ssh/terraform.pub"
EOS

cat << EOS > creds.sh
export DEEPL2_DISCORD_TOKEN='${DEEPL2_DISCORD_TOKEN}'
export DEEPL2_DISCORD_CHANNEL='${DEEPL2_DISCORD_CHANNEL}'
EOS

terraform init -input=false
terraform plan -out=tfplan -input=false
terraform apply -input=false tfplan

sleep 10 # Wait for instance

scp -o ConnectTimeout=120 -o StrictHostKeychecking=no -r -i ~/.ssh/terraform [!.]* ubuntu@$(terraform output ip):/home/ubuntu/deepl2-pybullet-locomotion

ssh -t -t -o StrictHostKeychecking=no -i ~/.ssh/terraform ubuntu@$(terraform output ip) << EOS
export DEBIAN_FRONTEND=noninteractive
cd /home/ubuntu/deepl2-pybullet-locomotion
sudo apt-get update
sudo apt-get install ffmpeg python3-pip python3-tk libffi-dev libopenmpi-dev libssl-dev psmisc curl git
sudo pip3 install pipenv
git clone https://github.com/openai/baselines --depth 1
sed -i -e 's/mujoco,atari,classic_control,robotics/classic_control/g' baselines/setup.py
pipenv install baselines/
pipenv install
. creds.sh
./run.sh
EOS
