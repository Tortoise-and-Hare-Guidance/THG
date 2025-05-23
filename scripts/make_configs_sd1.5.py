import yaml
import os
import torch


def dump(filename, data):
    with open(filename, "w") as file:
        yaml.dump(data, file, default_flow_style=False)


def caching():
    intervals = [11]
    starts = [17]
    for interval in intervals:
        for start in starts:
            config_data = {
                "solver_name": "CachingDDIMSolver",
                "solver_kwargs": {
                    "guidance_scale": 7.5,
                    "caching_interval": interval,
                    "caching_start": start,
                },
                "num_inference_steps": 50,
            }

            output_file = os.path.join("configs/t2i", f"cache{start}_{interval}.yaml")
            dump(output_file, config_data)


def gaussian():
    intervals = [3, 5, 7]
    starts = [0, 17]
    radius_values = [2]

    for radius in radius_values:
        for interval in intervals:
            for start in starts:
                config_data = {
                    "solver_name": "GaussianCachingDDIMSolver",
                    "solver_kwargs": {
                        "guidance_scale": 7.5,
                        "caching_interval": interval,
                        "caching_start": start,
                        "radius": radius,
                        "boost_factor": 1.2,
                        "i_0": 14,
                    },
                    "num_inference_steps": 50,
                }

                output_file = os.path.join(
                    "configs/t2i", f"g_cache{start}_{interval}_r{radius}.yaml"
                )
                dump(output_file, config_data)


def boost():
    intervals = [3, 5, 7]
    starts = [0, 17]
    factors = [1.2]
    for factor in factors:
        for interval in intervals:
            for start in starts:
                config_data = {
                    "solver_name": "BoostCachingDDIMSolver",
                    "solver_kwargs": {
                        "guidance_scale": 7.5,
                        "caching_interval": interval,
                        "caching_start": start,
                        "boost_factor": 1.2,
                    },
                    "num_inference_steps": 50,
                }

                output_file = os.path.join(
                    "configs/t2i", f"b_cache{start}_{interval}_x{factor:.2f}.yaml"
                )
                dump(output_file, config_data)


def fft():
    intervals = [11]
    starts = [17]
    radius_values = [10]
    for radius in radius_values:
        for interval in intervals:
            for start in starts:
                config_data = {
                    "solver_name": "FFTCachingDDIMSolver",
                    "solver_kwargs": {
                        "guidance_scale": 7.5,
                        "caching_interval": interval,
                        "caching_start": start,
                        "radius": radius,
                        "boost_factor": 1.2,
                        "i_0": 14,
                    },
                    "num_inference_steps": 50,
                }

                output_file = os.path.join(
                    "configs/t2i", f"f_cache{start}_{interval}_r{radius}.yaml"
                )
                dump(output_file, config_data)


def heun():
    for steps in range(12, 27, 2):
        config_data = {
            "solver_name": "HeunSolver",
            "solver_kwargs": {
                "guidance_scale": 7.5,
            },
            "num_inference_steps": steps,
        }

        output_file = os.path.join("configs/t2i", f"heun{steps}.yaml")
        dump(output_file, config_data)


def euler():
    for steps in range(25, 51, 5):
        config_data = {
            "solver_name": "EulerSolver",
            "solver_kwargs": {
                "guidance_scale": 7.5,
            },
            "num_inference_steps": steps,
        }

        output_file = os.path.join("configs/t2i", f"euler{steps}.yaml")
        dump(output_file, config_data)


def ddim():
    for steps in [35, 50]:
        config_data = {
            "solver_name": "DDIMSolver",
            "solver_kwargs": {
                "guidance_scale": 7.5,
            },
            "num_inference_steps": steps,
        }

        output_file = os.path.join("configs/t2i", f"ddim{steps}.yaml")
        dump(output_file, config_data)


def cfgpp():
    for steps in range(25, 51, 5):
        config_data = {
            "solver_name": "CFGPPDDIMSolver",
            "solver_kwargs": {
                "guidance_scale": 0.6,
            },
            "num_inference_steps": steps,
        }

        output_file = os.path.join("configs/t2i", f"cfgpp{steps}.yaml")
        dump(output_file, config_data)


def limited():
    for steps in range(1, 10):
        config_data = {
            "solver_name": "LimitedDDIMSolver",
            "solver_kwargs": {
                "guidance_scale": 7.5,
                "t_lo": steps,
                "t_hi": 50,
            },
            "num_inference_steps": 50,
        }

        output_file = os.path.join("configs/t2i", f"limited{steps}.yaml")
        dump(output_file, config_data)


def compute_macro_steps(cx, cg, rho):
    m = rho * cx / cg
    m = torch.clamp(m, min=1, max=50).to(torch.int32).tolist()
    res = []
    i = 0
    while i < 50:
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
            "solver_name": "MultirateDDIMSolver",
            "solver_kwargs": {
                "guidance_scale": 7.5,
                "macro_steps": m,
            },
            "num_inference_steps": 50,
        }
        output_file = os.path.join("configs/t2i", f"multirate_{rho:.1f}.yaml")
        dump(output_file, config_data)


def limited_back():
    for steps in range(20, 50, 2):
        config_data = {
            "solver_name": "LimitedDDIMSolver",
            "solver_kwargs": {
                "guidance_scale": 7.5,
                "t_lo": 0,
                "t_hi": steps,
            },
            "num_inference_steps": 50,
        }

        output_file = os.path.join("configs/t2i", f"limited_back{steps}.yaml")
        dump(output_file, config_data)


def mlb(path):
    c = torch.load(path).cpu()
    cx = c.mean(0)[:, 0]
    cg = c.mean(0)[:, 1]

    back_steps = list(range(34, 50, 2))
    rhos = [0.8, 0.9, 1.0, 1.1, 1.2, 1.3]

    for back_step in back_steps:
        for rho in rhos:
            m = compute_macro_steps(cx, cg, rho)
            config_data = {
                "solver_name": "LimitedBoostMultirateDDIMSolver",
                "solver_kwargs": {
                    "guidance_scale": 7.5,
                    "macro_steps": m,
                    "t_lo": 0,
                    "t_hi": back_step,
                    "boost_factor": 1.2,
                },
                "num_inference_steps": 50,
            }
            output_file = os.path.join("configs/t2i", f"mlb_{back_step}_{rho:.1f}.yaml")
            dump(output_file, config_data)


def mlb2(m):
    config_data = {
        "solver_name": "LimitedBoostMultirateDDIMSolver",
        "solver_kwargs": {
            "guidance_scale": 7.5,
            "macro_steps": m,
            "t_lo": 0,
            "t_hi": 38,
            "boost_factor": 1.2,
        },
        "num_inference_steps": 50,
    }
    output_file = os.path.join("configs/t2i", f"mlb_38_1.1.yaml")
    dump(output_file, config_data)


def main():
    os.mkdir("configs/t2i")
    caching()
    # gaussian()
    # boost()
    fft()
    # heun()
    # euler()
    ddim()
    # cfgpp()
    # limited()

    sd15_rho11 = [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        8,
        10,
        12,
        14,
        17,
        20,
        23,
        26,
        28,
        30,
        32,
        34,
        36,
        38,
        39,
        40,
        41,
        42,
        43,
        44,
        45,
        46,
        47,
        48,
        49,
    ]
    mlb2(sd15_rho11)

    # multirate("images/calc_m_sd1.5/denoising_output.pt")
    # limited_back()

    # mlb("images/calc_m_sd1.5/denoising_output.pt")


if __name__ == "__main__":
    main()
