from src.UKDALE_Dataset import UKDALE
from src.Models import UNetNiLM
from src.Preprocess import undo_normalize, _quantile_signal

import numpy as np


import torch


class SEnergyM():
    def __init__(self, unet_path, appliances, First_seq):
        self.power = np.array(First_seq)

        self.appliances = ['washing_machine', 'dishwasher', 'kettle', 'fridge',
                            'microwave'] if appliances is None else appliances
        self.train_config = {
            'taus': torch.tensor([0.025, 0.1, 0.5, 0.9, 0.975]),
            'means': np.array([32.57149563, 17.5326421, 14.76560635, 38.22767168, 5.9921367, 122.83210198], dtype=np.float64),
            'stds': np.array([216.65610219, 185.18295596, 174.87227402, 49.19164206, 82.51916197, 362.31497472], dtype=np.float64)
        }

        self.device = torch.device("cpu")

        self.window_size = 100
        self.n_quantiles = 5

        unet = UNetNiLM(
            num_layers=5,
            features_start=8,
            n_channels=1,
            num_classes=len(self.appliances),
            pooling_size=16,
            window_size=self.window_size,
            num_quantiles=self.n_quantiles,
            dropout=0.1,
            d_model=128,
        )

        unet.load_state_dict(torch.load(unet_path, map_location=torch.device('cpu')))

        unet = unet.to(self.device)

        unet.eval()

        self.train_config['model'] = unet

        agg = _quantile_signal(self.power, 10)
        agg = (agg - self.train_config['means'][-1]) / self.train_config['stds'][-1]

        agg = torch.tensor(agg, dtype=torch.float32, device=self.device)

        agg = agg.unsqueeze(0)

        quantile_idx = len(self.train_config['taus']) // 2

        y, s = self.train_config['model'](agg)
        y = y.detach().squeeze()
        s = s.detach().squeeze()

        # undo normalize
        y = undo_normalize(y, self.train_config['means'][:-1], self.train_config['stds'][:-1])

        s[s >= 0] = 1
        s[s < 0] = 0

        y = s * y[:, quantile_idx, :]

        self.appliances_power = np.array(y)
        self.appliances_stats = np.array(s)
        
    def get_aggregated_power(self):
        return self.power
    def get_appliances_stats(self):
        return self.appliances_stats

    def get_appliances_power(self):
        return self.appliances_power

    def predict(self, agg_power):
        self.power = np.append(self.power, agg_power)

        seq = self.power[self.power.shape[0]-100:self.power.shape[0]]

        seq = _quantile_signal(seq, 10)
        seq = (seq - self.train_config['means'][-1]) / self.train_config['stds'][-1]

        seq = torch.tensor(seq, dtype=torch.float32, device=self.device)

        seq = seq.unsqueeze(0)

        quantile_idx = len(self.train_config['taus']) // 2  # from model_pl, line 125

        y, s = self.train_config['model'](seq)
        y = y.detach().squeeze()
        s = s.detach().squeeze()

        # undo normalize
        y = undo_normalize(y, self.train_config['means'][:-1], self.train_config['stds'][:-1])

        s[s >= 0] = 1
        s[s < 0] = 0

        y = s * y[:, quantile_idx, :]

        self.appliances_power = np.append(self.appliances_power, np.array([y[-1,:].tolist()]), axis=0)
        self.appliances_stats = np.append(self.appliances_stats, np.array([s[-1,:].tolist()]), axis=0)

        # return y[-1,:], s[-1,:]

unet_path = "./models/model.pth"
target_appliances = ['washing_machine', 'dishwasher', 'kettle', 'fridge', 'microwave']



