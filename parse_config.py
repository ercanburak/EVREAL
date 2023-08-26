class ConfigParser:
    def __init__(self, config):
        """
        Simplified version of https://github.com/victoresque/pytorch-template/blob/master/parse_config.py
        Used by torch.load for unpickling ConfigParser objects from model checkpoints
        Then we use init_obj to create respective model class
        """
        self._config = config

    def init_obj(self, name, module, *args, **kwargs):
        module_name = self[name]['type']
        module_args = dict(self[name]['args'])
        assert all([k not in module_args for k in kwargs]), 'Overwriting kwargs given in config file is not allowed'
        module_args.update(kwargs)
        return getattr(module, module_name)(*args, **module_args)

    def __getitem__(self, name):
        return self.config[name]

    @property
    def config(self):
        return self._config
