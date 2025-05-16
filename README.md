# Tortoise and Hare Guidance

## Measure quality metrics for a method
```
accelerate launch main.py  # Run all experiments in configs/t2i
accelerate launch -m experiments.t2i with configs/t2i/<config>.yaml  # Run one experiment

python scripts/make_configs_<model>.py  # generate config files for experiment
```

## Calculate cT, cH
```
accelerate launch -m experiments.calc_m with configs/calc_m/<model>.yaml
```

## Setup local config file for main.py
```
cp configs/local_config.yaml.example configs/local_config.yaml
```
