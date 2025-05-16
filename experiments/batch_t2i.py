from pathlib import Path

from accelerate import Accelerator
from sacred import Experiment

from experiments.t2i import t2i, setup_experiment
from utils import read_yaml, deep_update


accelerator = Accelerator()


def do_experiment(config, name):
    ex = Experiment(name, ingredients=[t2i])
    setup_experiment(ex)
    try:
        options = None if accelerator.is_main_process else {"--loglevel": "ERROR"}
        ex.run(config_updates=config, options=options)
    except Exception as e:
        print(f"Error running experiment: {e}")
        raise
    accelerator.print()
    accelerator.wait_for_everyone()


def main():
    for model_config in Path("configs/models").glob("**/*.yaml"):
        for path in Path("configs/t2i").glob("**/*.yaml"):
            t2i_config = read_yaml(path)
            deep_update(t2i_config, read_yaml(model_config))
            config = {"t2i": t2i_config}
            deep_update(config, read_yaml("configs/local_config.yaml"))
            name = f"{path.stem}_{model_config.stem}"
            if "path" not in config["t2i"]:
                config["t2i"]["path"] = f"images/{name}"
            do_experiment(config, name)


if __name__ == "__main__":
    main()
