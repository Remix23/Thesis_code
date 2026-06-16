from torch import nn
from torch import cat
from torch import randn
import torch
import numpy as np

def findar_p (times_series, p):
    target = times_series[p:]
    lagged = np.array([times_series[i:-(p - i)] for i in range(p)]).T
    ones = np.ones((lagged.shape[0], 1))
    lagged = np.hstack((ones, lagged))
    ols = np.linalg.lstsq(lagged, target, rcond=None)
    return ols[0]

STATISTICS = {
    "mean": lambda sim_out: np.mean(sim_out),
    "std": lambda sim_out: np.std(sim_out),
    "yearly_corr": lambda sim_out: np.corrcoef(sim_out[:-4], sim_out[4:])[0, 1],
    "auto_corr_1" : lambda sim_out: np.corrcoef(sim_out[:-1], sim_out[1:])[0, 1],
    "auto_corr_2" : lambda sim_out: np.corrcoef(sim_out[:-2], sim_out[2:])[0, 1],
    "auto_corr_3" : lambda sim_out: np.corrcoef(sim_out[:-3], sim_out[3:])[0, 1],
    "ar1_coeff": lambda sim_out: findar_p(sim_out, 1)[1],
    "ar2_coeff": lambda sim_out: findar_p(sim_out, 2)[1],
    "min": lambda sim_out: np.min(sim_out),
    "max": lambda sim_out: np.max(sim_out),
    "skewness": lambda sim_out: np.mean((sim_out - np.mean(sim_out))**3) / np.std(sim_out)**3,
    "kurtosis": lambda sim_out: np.mean((sim_out - np.mean(sim_out))**4) / np.std(sim_out)**4,
    "quantile_25" : lambda sim_out: np.quantile(sim_out, .25),
    "quantile_50" : lambda sim_out: np.quantile(sim_out, .50),
    "quantile_75" : lambda sim_out: np.quantile(sim_out, .75),
    "recession_count" : lambda sim_out: np.sum(sim_out < 0),
    "sequential" : lambda sim_out: None,
}

implemented = ["elman", "gru"]

### with channel:
# log GDP 
# log diff GDP 
# detrended GDP

### every network gets:
# x: (NUM_CALIBRATION_DATES, N_RUNS, n_features, T_hist + 1)
class CNN_GDP(nn.Module):
	def __init__(self, stat_keys, in_channels = 3, 
			  conv_dims= [16, 32], summary_dims = 16, out_dims = 32, kernel_size = 3, pool_kernel_size = 1, **kwargs):
		super().__init__(**kwargs)
		num_summaries = len(stat_keys)
		self.stat_keys = stat_keys

		

		self.conv = nn.Sequential(
            nn.Conv1d(in_channels, 16, kernel_size=kernel_size, padding=kernel_size-1, dilation=1),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=kernel_size, padding=kernel_size-1, dilation=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(pool_kernel_size)
        )
		
		self.summary_net = nn.Sequential(
            nn.Linear(num_summaries, 16),
            nn.ReLU()
        )
		self.final = nn.Sequential(
            nn.Linear(32 + 16, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )

	def compute_summaries(self, x):
		summaries = np.array([STATISTICS[key](x) for key in self.stat_keys])
		return torch.tensor(summaries, dtype=torch.float32)

	def forward(self, x):
		x = x.reshape((x.shape[0], 3, -1)) # reshape to (batch_size, in_channels, sequence_length)
		# print(f"Input shape: {x.shape}")
		h1 = self.conv(x).squeeze(-1)

		log_gdp = x[:, 0, :]
		### log gdp in in first channel
		summaries = torch.zeros((log_gdp.shape[0], len(self.stat_keys)), dtype=torch.float32)
		for batch_id in range(log_gdp.shape[0]):
			summaries[batch_id, :] = self.compute_summaries(log_gdp[batch_id].numpy())

		summaries = self.summary_net(summaries)
		out = cat((h1, summaries), dim=1)
		
		return self.final(out)

class RNN(nn.Module):

	def __init__(self, input_dim, hidden_dim, num_layers, mlp_dims,
				 flavour="gru", hidden_out_dim=None, N=None, **kwargs):

		super(RNN, self).__init__(**kwargs)

		if not flavour in implemented:
			errmsg = "Kwarg 'flavour' must be in {0}".format(implemented)
			raise ValueError(errmsg)

		if flavour == "elman":	
			self.mod = nn.RNN(input_dim, hidden_dim, num_layers, batch_first=True)
		elif flavour == "gru":
			self.mod = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
		if isinstance(mlp_dims, int):
			output_dim = mlp_dims
			self._layers = []
			self.final = nn.Linear(hidden_dim, output_dim)
		elif isinstance(mlp_dims, list):
			self._layers = [nn.Linear(hidden_dim, mlp_dims[0])]
			for i in range(len(mlp_dims) - 2):
				self._layers.append(nn.Linear(mlp_dims[i], mlp_dims[i+1]))
			self.final = nn.Linear(mlp_dims[-2], mlp_dims[-1])
		self.relu = nn.ReLU()

	def forward(self, x):
		out, _ = self.mod(x)
		_x = out
		for layer in self._layers:
			_x = self.relu(layer(_x))
		return self.final(_x)
	
class SeqEmbedding(nn.Module):
	def __init__ (self, T, n_features, hidden_size, out_dim):
		super().__init__()
		self.T, self.n_features, self.hidden_size, self.out_dim = T, n_features, hidden_size, out_dim

		self.gru = nn.GRU(
			input_size=n_features,
			hidden_size = hidden_size,
			num_layers=2,
			batch_first=True,
		)

		self.head = nn.Linear(
			hidden_size, 
			out_dim,
		)

	def forward(self, x):
		# x is (CALIBRATION_DATES, N_RUNS, n_features, T_hist + 1)
		### take average of the runs
		x = x.reshape(-1, self.n_features, self.T) # (batch_size, T, n_features)
		x = x.permute(0, 2, 1) # (batch_size, n_features, T)
		_, h = self.gru(x)
		return self.head(h[-1])
	
class SimpleHierarchical(nn.Module):
	def __init__ (self, calibration_dates, T, n_features, hidden_size, out_dim):
		super().__init__()
		self.T, self.n_features, self.hidden_size, self.out_dim = T, n_features, hidden_size, out_dim
		self.n_calibration_dates = calibration_dates

		self.gru = nn.GRU(
			input_size=n_features,
			hidden_size = hidden_size,
			num_layers=2,
			batch_first=True,
		)

		self.head = nn.Linear(
			hidden_size, 
			out_dim,
		)

	def forward(self, x):
		# x is (CALIBRATION_DATES, N_RUNS, n_features, T_hist + 1)
		### take average of the runs
		x = x.reshape(-1, self.n_calibration_dates, self.n_features, self.T) # (batch_size, T, n_features)
	
		### compact to (batch_size, T, n_features)
		batch_size = x.shape[0]
		x = x.reshape(batch_size * self.n_calibration_dates, self.n_features, self.T) # (batch_size * n_calibration_dates, n_features, T)

		### for gru
		x = x.permute(0, 2, 1) # (batch_size * n_calibration_dates, T, n_features)
		
		_, h = self.gru(x) ### h: (1, batch_size * n_calibration_dates, hidden_size)
		
		h_out = h[-1].reshape(batch_size, self.n_calibration_dates, self.hidden_size) # (batch_size, n_calibration_dates * hidden_size)
		
		return self.head(h_out.mean(dim=1))
	
class Hierarchical(nn.Module):
	def __init__ (self, calibration_dates, T, n_features, hidden_sizes, out_dim):
		super().__init__()
		self.T, self.n_features, self.hidden_sizes, self.out_dim = T, n_features, hidden_sizes, out_dim
		self.n_calibration_dates = calibration_dates

		self.gru1 = nn.GRU(
			input_size=n_features,
			hidden_size = hidden_sizes[0],
			num_layers=2,
			batch_first=True,
		)

		self.gru2 = nn.GRU(
			input_size=hidden_sizes[0],
			hidden_size = hidden_sizes[1],
			num_layers=2,
			batch_first=True,
		)

		self.head = nn.Linear(
			hidden_sizes[1], 
			out_dim,
		)

	def forward(self, x):
		# x is (CALIBRATION_DATES, n_features, T_hist + 1)
		### take average of the runs
		x = x.reshape(-1, self.n_calibration_dates, self.n_features, self.T) # (batch_size, T, n_features)
	
		### compact to (batch_size, T, n_features)
		batch_size = x.shape[0]
		x = x.reshape(batch_size * self.n_calibration_dates, self.n_features, self.T) # (batch_size * n_calibration_dates, n_features, T)

		### for gru
		x = x.permute(0, 2, 1) # (batch_size * n_calibration_dates, T, n_features)
		
		_, h = self.gru1(x) ### h: (1, batch_size * n_calibration_dates, hidden_size)
		h = h[-1, : , :]
		h = h.reshape(batch_size, self.n_calibration_dates, self.hidden_sizes[0]) # (batch_size, n_calibration_dates, hidden_size)
		
		_, h_over_calibration =self.gru2(h) ### h_over_calibration: (1, batch_size, hidden_size)

		h_out = h_over_calibration[-1, :, :]
		
		return self.head(h_out)
	
if __name__ == "__main__":
	cnn = CNN_GDP(
		stat_keys=["mean", "std"], 
	)
	print(cnn.forward(randn(1, 3, 10)))

	seq = SeqEmbedding(
		T = 10, 
		n_features = 3,
		hidden_size = 64,
		out_dim = 16,
	)

	print(seq.forward(randn(1, 3, 10).flatten()))

	simple = SimpleHierarchical(
		calibration_dates = 5,
		T = 10,
		n_features = 3,
		hidden_size = 64,
		out_dim = 16,
	)
	print(simple.forward(randn(1, 5, 3, 10).flatten()))

	hierarchical = Hierarchical(
		calibration_dates = 5,
		T = 10,
		n_features = 3,
		hidden_sizes = [64, 32],
		out_dim = 16,
	)

	print(hierarchical.forward(randn(1, 5, 3, 10).flatten()))

	print("Num of trainable parameters:")
	print(sum(p.numel() for p in cnn.parameters() if p.requires_grad))
	print(sum(p.numel() for p in seq.parameters() if p.requires_grad))
	print(sum(p.numel() for p in simple.parameters() if p.requires_grad))
	print(sum(p.numel() for p in hierarchical.parameters() if p.requires_grad))
	
	