"""Write training YAML without losing scientific-notation numeric types."""
import json
from pathlib import Path
import yaml
from common.configuration import load_yaml_config


def write_runtime_config(path, cfg):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    # JSON's 9e-06 is a string under the runtime's YAML 1.1 resolver.
    # SafeDumper emits 9.0e-06, retaining floats without modifying the loader.
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    reloaded = load_yaml_config(path)
    canonical = lambda value: json.dumps(value, sort_keys=True, allow_nan=False)
    if canonical(reloaded) != canonical(cfg):
        raise ValueError(f'training config changed on reload: {path}')
    for key in ('lr', 'weight_decay'):
        if type(reloaded[key]) is not float:
            raise TypeError(f'{key} must remain a float after training reload')
    return reloaded
