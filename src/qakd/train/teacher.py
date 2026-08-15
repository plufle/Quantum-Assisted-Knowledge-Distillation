import json

import hydra
from omegaconf import OmegaConf

from qakd.train.trainer import train_teacher


@hydra.main(version_base=None, config_path="../../../configs", config_name="config")
def main(cfg):
    OmegaConf.resolve(cfg)
    result = train_teacher(cfg)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
