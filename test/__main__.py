from .wso_test import wso_test, wso_vs_pso_test
from .destabilization import destabilize_roughness

import yaml

if __name__ == "__main__":
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    destabilize_roughness(config["paths"]["models"]["target"],
                          "output/destabilized.inp",
                          noise_std_frac=0.1,
                          min_roughness=0.001)
    # wso_test()
    # wso_vs_pso_test()