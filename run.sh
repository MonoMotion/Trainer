#!/bin/bash

nohup pipenv run python train.py --monitor monitor --save model -se 500 --monitor-video 50000 --timesteps 10000000 --tensorboard ./tblog --discord &
nohup pipenv run python discord_reporter.py &
