import json
from pathlib import Path
from typing import Union


class Config(object):
    """
    - initialize with a dictionary of config
    - define dependency rules rule_{key}(self, key, value) to:
        update self.cfg accordingly
    - provides get, pop and save functions
    """
    def __init__(self, dict_default: dict = None, name: str = 'config'):
        self.name = name
        self.cfg = dict_default if isinstance(dict_default, dict) else dict()

    def get(self) -> dict:
        return self.cfg

    def set(self, settings: dict) -> dict:
        """
        settings: {key: value to change}
        update self.cfg, add key and value pairs if not exists
        """
        for key, val in settings.items():
            fun = getattr(self, 'rule_{}'.format(key), None)
            if callable(fun):
                fun(key, val)
            else:
                self.cfg[key] = val
        return self.cfg

    def pop(self, keys: Union[str, list]):
        """remove keys from self.cfg regardless of whether it exists"""
        keys = [keys] if isinstance(keys, str) else keys
        for key in keys:
            self.cfg.pop(key, None)
        return self.cfg

    def get_str_cfg(self) -> dict:
        """return a copy of self.cfg where all the values are turned to str"""
        config = self.cfg.copy()
        for key, val in config.items():
            config[key] = str(val)
        return config

    def save(self, prefix: Union[str, list] = '.', filename: str = None) -> None:
        """save self.cfg to prefix/filename"""
        config = self.get_str_cfg()
        file = filename if filename is not None else 'cfg_{}.json'.format(self.name)
        path = Path(prefix, file)
        with path.open(mode='w', encoding='utf-8') as f:
            json.dump(config, f, indent=4, ensure_ascii=False)
