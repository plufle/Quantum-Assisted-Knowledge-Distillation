import json

import hydra
from omegaconf import OmegaConf

from qakd.train.trainer import train_student


@hydra.main(version_base=None, config_path="../../../configs", config_name="config")
def main(cfg):
    OmegaConf.resolve(cfg)
    result = train_student(cfg)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
