terraform {
  required_version = ">= 0.11.0"

  backend "s3" {
    bucket = "deepl2"
    key    = "terraform.tfstate"
    region = "us-west-2"
  }
}

provider "aws" {
  access_key = "${var.access_key}"
  secret_key = "${var.secret_key}"
  region     = "${var.region}"
}

data "aws_ami" "ubuntu_xenial" {
    most_recent = true

    filter {
        name   = "name"
        values = ["ubuntu/images/hvm-ssd/ubuntu-xenial-16.04-amd64-server-*"]
    }

    filter {
        name   = "virtualization-type"
        values = ["hvm"]
    }

    owners = ["099720109477"]
}

resource "aws_spot_instance_request" "yamax_learn_worker" {
  ami           = "${data.aws_ami.ubuntu_xenial.id}"
  spot_price    = "0.1"
  instance_type = "${var.instance_type}"
  key_name      = "${aws_key_pair.auth.id}"
  spot_type = "one-time"
  wait_for_fulfillment = true
  vpc_security_group_ids = ["${aws_security_group.default.id}"]
  associate_public_ip_address = true

  tags {
    Name = "YamaXLearnWorker"
  }
}

resource "aws_security_group" "default" {
  name        = "ssh_tensorboard_security_group"
  description = "Used in the terraform"

  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    from_port   = 6006
    to_port     = 6006
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port = 0
    to_port = 0
    protocol = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_key_pair" "auth" {
  key_name   = "${var.key_name}"
  public_key = "${file(var.public_key_path)}"
}

output "ip" {
  value = "${aws_spot_instance_request.yamax_learn_worker.public_ip}"
}
