import yaml
import os
import torch


def dump(filename, data):
    with open(filename, "w") as file:
        yaml.dump(data, file, default_flow_style=False)


def caching():
    intervals = [3, 5, 7]
    starts = [0, 9]
    for interval in intervals:
        for start in starts:
            config_data = {
                "solver_name": "CachingFlowMatchEulerSolver",
                "solver_kwargs": {
                    "guidance_scale": 3.5,
                    "caching_interval": interval,
                    "caching_start": start,
                },
                "num_inference_steps": 28,
            }

            output_file = os.path.join("configs/t2i", f"cache{start}_{interval}.yaml")
            dump(output_file, config_data)


def gaussian():
    intervals = [3, 5, 7]
    starts = [0, 9]
    radius_values = [2]

    for radius in radius_values:
        for interval in intervals:
            for start in starts:
                config_data = {
                    "solver_name": "GaussianCachingFlowMatchEulerSolver",
                    "solver_kwargs": {
                        "guidance_scale": 3.5,
                        "caching_interval": interval,
                        "caching_start": start,
                        "radius": radius,
                        "boost_factor": 1.2,
                        "i_0": 14,
                    },
                    "num_inference_steps": 28,
                }

                output_file = os.path.join(
                    "configs/t2i", f"g_cache{start}_{interval}_r{radius}.yaml"
                )
                dump(output_file, config_data)


def boost():
    intervals = [3, 5, 7]
    starts = [0, 9]
    factors = [1.2]
    for factor in factors:
        for interval in intervals:
            for start in starts:
                config_data = {
                    "solver_name": "BoostCachingFlowMatchEulerSolver",
                    "solver_kwargs": {
                        "guidance_scale": 3.5,
                        "caching_interval": interval,
                        "caching_start": start,
                        "boost_factor": 1.2,
                    },
                    "num_inference_steps": 28,
                }

                output_file = os.path.join(
                    "configs/t2i", f"b_cache{start}_{interval}_x{factor:.2f}.yaml"
                )
                dump(output_file, config_data)


def fft():
    intervals = [3, 5, 7]
    starts = [0, 9]
    radius_values = [10]
    for radius in radius_values:
        for interval in intervals:
            for start in starts:
                config_data = {
                    "solver_name": "FFTCachingFlowMatchEulerSolver",
                    "solver_kwargs": {
                        "guidance_scale": 3.5,
                        "caching_interval": interval,
                        "caching_start": start,
                        "radius": radius,
                        "boost_factor": 1.2,
                        "i_0": 14,
                    },
                    "num_inference_steps": 28,
                }

                output_file = os.path.join(
                    "configs/t2i", f"f_cache{start}_{interval}_r{radius}.yaml"
                )
                dump(output_file, config_data)


def heun():
    for steps in range(10, 17, 2):
        config_data = {
            "solver_name": "FlowMatchHeunSolver",
            "solver_kwargs": {
                "guidance_scale": 3.5,
            },
            "num_inference_steps": steps,
        }

        output_file = os.path.join("configs/t2i", f"heun{steps}.yaml")
        dump(output_file, config_data)


def euler():
    for steps in range(14, 29, 2):
        config_data = {
            "solver_name": "FlowMatchEulerSolver",
            "solver_kwargs": {
                "guidance_scale": 3.5,
            },
            "num_inference_steps": steps,
        }

        output_file = os.path.join("configs/t2i", f"euler{steps}.yaml")
        dump(output_file, config_data)


def limited():
    for steps in range(1, 10):
        config_data = {
            "solver_name": "LimitedFlowMatchEulerSolver",
            "solver_kwargs": {
                "guidance_scale": 3.5,
                "t_lo": steps,
                "t_hi": 28,
            },
            "num_inference_steps": steps,
        }

        output_file = os.path.join("configs/t2i", f"limited{steps}.yaml")
        dump(output_file, config_data)


def compute_macro_steps(cx, cg, rho):
    m = rho * cx / cg
    m = torch.clamp(m, min=1, max=28).to(torch.int32).tolist()
    res = []
    i = 0
    while i < 28:
        res.append(i)
        i += m[i]
    return res


def multirate(path: str):
    c = torch.load(path).cpu()
    cx = c.mean(0)[:, 0]
    cg = c.mean(0)[:, 1]

    rhos = [0.5 + i * 0.1 for i in range(16)]

    for rho in rhos:
        m = compute_macro_steps(cx, cg, rho)
        config_data = {
            "solver_name": "MultirateFlowMatchEulerSolver",
            "solver_kwargs": {
                "guidance_scale": 3.5,
                "macro_steps": m,
            },
            "num_inference_steps": 28,
        }
        output_file = os.path.join("configs/t2i", f"multirate_{rho:.1f}.yaml")
        dump(output_file, config_data)


def limited_back():
    for steps in range(5, 28, 2):
        config_data = {
            "solver_name": "LimitedFlowMatchEulerSolver",
            "solver_kwargs": {
                "guidance_scale": 3.5,
                "t_lo": 0,
                "t_hi": steps,
            },
            "num_inference_steps": 28,
        }

        output_file = os.path.join("configs/t2i", f"limited_back{steps}.yaml")
        dump(output_file, config_data)


def mlb(path):
    c = torch.load(path).cpu()
    cx = c.mean(0)[:, 0]
    cg = c.mean(0)[:, 1]

    back_steps = [15, 17, 19, 21, 23, 25, 27]
    rhos = [0.8, 0.9, 1.0, 1.1, 1.2, 1.3]

    for back_step in back_steps:
        for rho in rhos:
            m = compute_macro_steps(cx, cg, rho)
            config_data = {
                "solver_name": "LimitedBoostMultirateFlowMatchEulerSolver",
                "solver_kwargs": {
                    "guidance_scale": 3.5,
                    "macro_steps": m,
                    "t_lo": 0,
                    "t_hi": back_step,
                    "boost_factor": 1.2,
                },
                "num_inference_steps": 28,
            }
            output_file = os.path.join("configs/t2i", f"mlb_{back_step}_{rho:.1f}.yaml")
            dump(output_file, config_data)


def main():
    os.mkdir("configs/t2i")
    # caching()
    # gaussian()
    # boost()
    # fft()
    # heun()
    # euler()
    # limited()

    # multirate("images/calc_m_sd3.5l/denoising_output.pt")
    # limited_back()

    mlb("images/calc_m_sd3.5l/denoising_output.pt")


if __name__ == "__main__":
    main()
