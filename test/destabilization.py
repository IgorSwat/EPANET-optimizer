import numpy as np
import pandas as pd

from epyt import epanet


def destabilize_roughness(input_model_path: str, output_model_path: str,
                          noise_std_frac: float = 0.05,
                          min_roughness: float = 1e-6) -> None:
    try:
        network = epanet(input_model_path)
    except Exception as e:
        print(f"ERROR: {e}")
        exit(1)

    pipe_idxs    = network.getLinkPipeIndex()
    base_rough   = np.array(network.getLinkRoughnessCoeff(), dtype=float)

    sigma = noise_std_frac * base_rough
    noise = np.random.normal(loc=0.0, scale=sigma)

    new_rough = base_rough + noise
    new_rough[new_rough <= min_roughness] = min_roughness

    network.setLinkRoughnessCoeff(pipe_idxs, new_rough.tolist())

    network.saveInputFile(output_model_path)

    print(f"Perturbed model written to {output_model_path}")