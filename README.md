# DeepL2.Trainer

An experiment to make [YamaX](https://y-modify.org/yamax) walk in the simulation environment using Bullet physics engine.

DeepL2 Project: [https://blog.y-modify.org/2018/01/04/deepl2-start/](https://blog.y-modify.org/2018/01/04/deepl2-start/)

## Try

### Preparation

```shell
# Clone this repo
git clone https://github.com/Y-modify/deepl2.git --recursive
cd deepl2

# Setup
./setup.sh
```

### Train

```shell
pipenv run train -i input.fom -r robot.urdf
```

### Preview the trained motion

```shell
pipenv run preview -i trained.fom -r robot.urdf
```
