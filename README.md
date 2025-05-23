# Tortoise and Hare Guidance

## Create Docker Environment
```
docker build -t thg:latest .
docker run -d -it --gpus=all --ipc=host thg:latest
```
## Setup

Run inside docker container.
```
git clone https://github.com/Tortoise-and-Hare-Guidance/THG
cd THG
python scripts/extract_coco30k.py  # Download coco 2014 30k dataset
cp configs/local_config.yaml.example configs/local_config.yaml  # Use example config (generate 30 images)
rm configs/models/sd3.5l.yaml  # Remove sd3.5l config for sd1.5 experiment
mkdir -p /data/db
(mongod > /dev/null 2>&1 &) ; disown  # Run mongodb background
```
## Run T2I task

```
accelerate launch main.py  # Run all experiments in configs/t2i

python scripts/make_configs_<model>.py  # generate config files for experiment
```

## Calculate error bounds

```
accelerate launch -m experiments.calc_m with configs/calc_m/sd1.5.yaml
accelerate launch -m experiments.calc_m with configs/calc_m/sd3.5l.yaml
```
