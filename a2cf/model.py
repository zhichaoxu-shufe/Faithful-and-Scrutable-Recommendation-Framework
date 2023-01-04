import torch
from collections import OrderedDict

def compute_vp(X_user, Y_item, F, epsilon):
	ln = softmax(X_user * Y_item / epsilon)
	vp = np.zeros(len(F[0]))
	for n in range(len(F)):
		vp += F[n] * ln[n]
	return vp

def tanh_modified(e, n):
	return (n * torch.exp(2 * e) + 1) / (torch.exp(2 * e) + 1)

class EmbeddingNet(torch.nn.Module):
	def __init__(self, X, Y, m, n, p, r):  # m: user num, n: item num, p: feature num, r: implicit feature num
		super(EmbeddingNet, self).__init__()
		self.X = X.float()
		self.Y = Y.float()
		self.U = torch.nn.Parameter(torch.FloatTensor(m, r).normal_(0., 0.01))
		self.I = torch.nn.Parameter(torch.FloatTensor(n, r).normal_(0., 0.01))
		self.F = torch.nn.Parameter(torch.FloatTensor(p, r).normal_(0., 0.01))
		h0_dim = 2 * r
		self.u_f_fc1 = torch.nn.Linear(h0_dim, h0_dim)
		self.relu = torch.nn.ReLU()
		self.u_f_fc2 = torch.nn.Linear(h0_dim, 1, bias=False)
		self.i_f_fc1 = torch.nn.Linear(h0_dim, h0_dim)
		self.i_f_fc2 = torch.nn.Linear(h0_dim, 1, bias=False)

		torch.nn.init.normal_(self.u_f_fc1.weight, 0, 0.01)
		torch.nn.init.normal_(self.u_f_fc2.weight, 0, 0.01)
		torch.nn.init.normal_(self.i_f_fc1.weight, 0, 0.01)
		torch.nn.init.normal_(self.i_f_fc2.weight, 0, 0.01)

	def forward(self, data_pair, mode):
		if mode == 'u_f':
			gt_score = torch.unsqueeze(self.X[data_pair[:, 0], data_pair[:, 1]], dim=1)
			user_embedding = self.U[data_pair[:, 0]]
			feature_embedding = self.F[data_pair[:, 1]]
			h0 = torch.cat((user_embedding, feature_embedding), dim=1)
			h1 = self.relu(self.u_f_fc1(h0)) + h0
			pre_score = tanh_modified(self.u_f_fc2(h1), 5)
		elif mode == 'i_f':
			gt_score = torch.unsqueeze(self.Y[data_pair[:, 0], data_pair[:, 1]], dim=1)
			item_embedding = self.I[data_pair[:, 0]]
			feature_embedding = self.F[data_pair[:, 1]]
			h0 = torch.cat((item_embedding, feature_embedding), dim=1)
			h1 = self.relu(self.i_f_fc1(h0)) + h0
			pre_score = tanh_modified(self.i_f_fc2(h1), 5)
		else:
			print('Err: wrong mode. Has to be u_f or i_f.')
			exit(1)
		return pre_score, gt_score


class RankingScoreNet(torch.nn.Module):
	def __init__(self, feature_length):
		super(RankingScoreNet, self).__init__()
		self.fc = torch.nn.Sequential(OrderedDict({
			'fc_1': torch.nn.Linear(feature_length*2, feature_length),
			'ReLU': torch.nn.ReLU(),
			'fc_2': torch.nn.Linear(feature_length, 1)
			}))

	def forward(self, A2CF_u_i_embedding):
		out = self.fc(A2CF_u_i_embedding)
		return out


class TwoWayNet(torch.nn.Module):
	def __init__(self, feature_length, alpha):
		super(TwoWayNet, self).__init__()
		self.fc1 = torch.nn.Sequential(OrderedDict({
			'fc_1': torch.nn.Linear(feature_length, feature_length),
			'ReLU': torch.nn.ReLU(),
			'fc_2': torch.nn.Linear(feature_length, 1)
			}))
		self.fc2 = torch.nn.Sequential(OrderedDict({
			'fc_1': torch.nn.Linear(feature_length, feature_length),
			'ReLU': torch.nn.ReLU(),
			'fc_2': torch.nn.Linear(feature_length, 1)
			}))
		self.alpha = alpha

	def forward(self, ui_product_embed, ui_vp_embed):
		out1 = self.fc1(ui_product_embed)
		out2 = self.fc2(ui_vp_embed)
		return self.alpha*out1 + (1-self.alpha)*out2

	# def output_aspect(self, ui_vp_embed):


class TupleNet(torch.nn.Module):
	def __init__(self, embedding_net):
		super(TupleNet, self).__init__()
		self.embedding_net = embedding_net

	def forward(self, u_p_embedding, u_n_embedding):
		p_score = self.embedding_net(u_p_embedding)
		n_score = self.embedding_net(u_n_embedding)
		return p_score, n_score
