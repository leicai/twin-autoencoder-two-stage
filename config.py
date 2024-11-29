import yaml
from easydict import EasyDict
from pathlib import Path
import argparse


def read_config(config_file):
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--model_file', type=str, default = config_file)
    args = parser.parse_args()
    cfg_from_yaml_file(args.model_file, cfg)
    return cfg


# read yaml files and merge to config(unique)
def cfg_from_yaml_file(cfg_file, config):
    # Merge incoming config
    with open(cfg.root_path/cfg_file, 'r', encoding='utf-8') as f:
        try:
            new_config = yaml.safe_load(f, Loader=yaml.FullLoader)
        except:
            new_config = yaml.safe_load(f)
        merge_new_config(config=config, new_config=new_config)
    return config


def merge_new_config(config, new_config):
    # This 'if' cycle will merge  '_BASE_CONFIG_' in yaml files
    if '_BASE_CONFIG_' in new_config:
        with open(cfg.root_path/new_config['_BASE_CONFIG_'], 'r', encoding='utf-8') as f:
            try:
                yaml_config = yaml.safe_load(f, Loader=yaml.FullLoader)
            except:
                yaml_config = yaml.safe_load(f)
        config.update(EasyDict(yaml_config))

    # This 'for' cycle will add 'config' some new property, iterate over the dictionary
    for key, val in new_config.items():
        # Determine whether val is a dict object
        # no: add directly to the dictionary
        if not isinstance(val, dict):
            config[key] = val
            continue
        # yes: recursive merge
        # this 'if' to judge 'key' whether is already in config
        if key not in config:
            config[key] = EasyDict()
        merge_new_config(config[key], val)
    return config


cfg = EasyDict()
cfg.root_path = (Path(__file__).resolve().parent / '../').resolve()
cfg.root_path_str = cfg.root_path.as_posix()
