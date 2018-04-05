provider "aws" {
  access_key = "${var.access_key}"
  secret_key = "${var.secret_key}"
  region     = "${var.region}"
}

resource "aws_spot_instance_request" "yamax_learn_worker" {
  ami           = "ami-4e79ed36"
  spot_price    = "0.1"
  instance_type = "${var.instance_type}"
  key_name      = "${aws_key_pair.auth.id}"
  spot_type = "one-time"
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
}

resource "aws_key_pair" "auth" {
  key_name   = "${var.key_name}"
  public_key = "${file(var.public_key_path)}"
}

output "ip" {
  value = "${aws_spot_instance_request.yamax_learn_worker.public_ip}"
}
