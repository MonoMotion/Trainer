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
terraform apply -input=false tfplan || exit -1

cat terraform.tfstate

IP_ADDR=$(terraform output ip)

sleep 30 # Wait for instance

until ssh -o StrictHostKeychecking=no -i ~/.ssh/terraform ubuntu@$IP_ADDR 'mkdir -p /home/ubuntu/deepl2'
do
  sleep 1
done

scp -o StrictHostKeychecking=no -r -i ~/.ssh/terraform [!.]* ubuntu@$IP_ADDR:/home/ubuntu/deepl2

ssh -t -t -o StrictHostKeychecking=no -i ~/.ssh/terraform ubuntu@$IP_ADDR << EOS
export DEBIAN_FRONTEND=noninteractive
cd /home/ubuntu/deepl2
sudo -E apt-get update
sudo -E apt-get install -y -qq ffmpeg python3-pip python3-tk libffi-dev libopenmpi-dev libssl-dev psmisc curl git
sudo -E pip3 install pipenv
git clone https://github.com/openai/baselines --depth 1
sed -i -e 's/mujoco,atari,classic_control,robotics/classic_control/g' baselines/setup.py
pipenv install baselines/
pipenv install
. creds.sh
./run.sh
exit
EOS
