from torch import nn
from torch import cat
from torch import randn

implemented = ["elman", "gru"]

class MLP(nn.Module):
	def __init__(self, input_dim, hidden_dim, num_layers, mlp_dims, **kwargs):
		pass

### with channel:
# log GDP 
# log diff GDP 
# detrended GDP
class CNN_GDP(nn.Module):
    def __init__(self, num_summaries = 8,**kwargs):
        super().__init__(**kwargs)

        self.conv = nn.Sequential(
            nn.Conv1d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
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

    def forward(self, x):
        h1 = self.conv(x).squeeze(-1)
		# compute summaries
		# h2 = self.summary_net(x.mean(dim=-1))
		#h = cat([h1, h2], dim=-1)
        return self.final(h1)

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
	
if __name__ == "__main__":
	rnn = RNN(
		input_dim=10,
		flavour="elman",
		hidden_dim=32,
		num_layers=2,
		mlp_dims=12
	)
	print(rnn.forward(randn(1, 10)))
	print(rnn)