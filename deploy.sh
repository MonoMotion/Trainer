#!/bin/bash
# Deploy Script
# Required Environment Variables:
# $DEEPL2_DISCORD_CHANNEL
# $DEEPL2_DISCORD_TOKEN
# $DEEPL2_S3_BUCKET_NAME
# $TF_VAR_access_key
# $TF_VAR_secret_key
# Optional:
# $DEEPL2_ADDITIONAL_SSH_PUBKEY

echo -e 'y\n' | ssh-keygen -t rsa -b 4096 -C "me@coord-e.com" -N '' -f ~/.ssh/terraform

cat << EOS > terraform.tfvars
key_name = "terraform"
public_key_path = "~/.ssh/terraform.pub"
EOS

cat << EOS > creds.sh
export BUCKET_NAME='${DEEPL2_S3_BUCKET_NAME}'
export AWS_ACCESS_KEY_ID='${TF_VAR_access_key}'
export AWS_SECRET_ACCESS_KEY='${TF_VAR_secret_key}'
export DEEPL2_DISCORD_TOKEN='${DEEPL2_DISCORD_TOKEN}'
export DEEPL2_DISCORD_CHANNEL='${DEEPL2_DISCORD_CHANNEL}'
export DEEPL2_ADDITIONAL_SSH_PUBKEY='${DEEPL2_ADDITIONAL_SSH_PUBKEY}'
EOS

mkdir ~/.aws

cat << EOS > ~/.aws/credentials
[default]
aws_access_key_id = ${TF_VAR_access_key}
aws_secret_access_key = ${TF_VAR_secret_key}
EOS

terraform init -input=false
terraform plan -out=tfplan -input=false
terraform apply -input=false tfplan || exit -1

IP_ADDR=$(terraform output ip)

sleep 50 # Wait for instance

until ssh -o StrictHostKeychecking=no -i ~/.ssh/terraform ubuntu@$IP_ADDR 'mkdir -p /home/ubuntu/deepl2'
do
  ((cnt++)) && ((cnt==10)) && exit -1
  sleep 5
done

scp -o StrictHostKeychecking=no -r -i ~/.ssh/terraform [!.]* ubuntu@$IP_ADDR:/home/ubuntu/deepl2 || exit -1

ssh -t -t -o StrictHostKeychecking=no -i ~/.ssh/terraform ubuntu@$IP_ADDR << EOS || exit -1
export DEBIAN_FRONTEND=noninteractive \
&& cd /home/ubuntu/deepl2 \
&& sudo -E apt-get update \
&& sudo -E apt-get install -y -qq ffmpeg python3-pip python3-tk libffi-dev libopenmpi-dev libssl-dev psmisc curl git \
&& sudo -H pip3 install pipenv \
&& sudo -H pip3 install awscli \
&& git clone https://github.com/openai/baselines --depth 1 \
&& sed -i -e 's/mujoco,atari,classic_control,robotics/classic_control/g' baselines/setup.py \
&& pipenv install baselines/ \
&& pipenv install \
&& echo "Dependency installation succeeded" \
&& . creds.sh \
&& . run.sh \
&& echo "Deploy succeeded" \
&& echo $DEEPL2_ADDITIONAL_SSH_PUBKEY >> ~/.ssh/authorized_keys \
&& echo "SSH key injection succeeded" \
&& exit
EOS
