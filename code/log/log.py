import torch
import os

class DataLogger:
    def __init__(self, name, hyperparams, log_dir):
        self.name = name
        self.hyperparams = hyperparams
        self.data = {}
        self.log_dir = log_dir
        self.path = os.path.join(log_dir, name)

    def update(self, epoch, **kwargs):
        for key, value in kwargs.items():
            if key not in self.data:
                self.data[key] = {}
            self.data[key][epoch] = value

    def dump(self):
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        torch.save({
            'name': self.name,
            'hyperparams': self.hyperparams,
            'data': self.data
        }, self.path)

    def load(self):
        if os.path.exists(self.path):
            loaded_data = torch.load(self.path, weights_only=True)
            self.name = loaded_data['name']
            self.hyperparams = loaded_data['hyperparams']
            self.data = loaded_data['data']
        else:
            raise FileNotFoundError(f"{self.path} does not exist.")

