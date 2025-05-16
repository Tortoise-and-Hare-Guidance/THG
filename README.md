# thindiffusion

Example usage:
```
accelerate launch -m experiments.batch_t2i  # Run all experiments in configs/t2i
accelerate launch -m experiments.t2i with configs/t2i/cfg.yaml  # Run one experiment
```

## Calculate m

```
accelerate launch -m experiments.calc_m with configs/calc_m/sd3.5l.yaml
```

## Setup
```
cp configs/local_config.yaml.example configs/local_config.yaml
```