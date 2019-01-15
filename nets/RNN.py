class GRU_Encoder(nn.Module):
	def __init__(self, num_cls):
		super(GRU_Encoder, self).__init__()
		self.hidden_dim = 64
		self.input_dim = 128
		self.output_dim = 128#num_cls#50 # class_num or feature dimension(for concat)
	
		self.GRU = nn.GRU(self.input_dim, self.hidden_dim, num_layers = 2,  batch_first = True, dropout=0.5)
		self.fc = nn.Sequential(
			nn.Linear(self.hidden_dim , self.output_dim),
			nn.ReLU(),
			#nn.Sigmoid()
		)
		self.fc2 = nn.Sequential(
			nn.Linear(self.output_dim, 40),
		)
		#utils.initialize_weights(self)

	def forward(self, feature):

		feature = feature.transpose(1,2)#dimension change : batch x time x dimension	
		x, hidden = self.GRU(feature)
		x = x.select(1, x.size(1)-1).contiguous()
		x = x.view(-1, self.hidden_dim)
		result = self.fc(x)
		result = self.fc2(result)
		return result


class LSTM(nn.Module):
	#I should divide one channel each and pass through LSTM and lst concat each channel(dim)
	def __init__(self, batch_size):
		super(LSTM, self).__init__()
		self.hidden_dim = 50
		self.embedding_dim = 32
		self.input_dim = 5
		self.output_dim = 10
		self.batch_size = batch_size

		self.embedding = nn.Embedding(self.input_dim, self.embedding_dim)
		self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim)
		self.hidden2label = nn.Linear(self.hidden_dim, self.output_dim)

		self.hidden = self.init_hidden()
	
	def init_hidden(self):
		h0 = Variable(torch.zeros(1, self.batch_size, self.hidden_dim).cuda())
		c0 = Variable(torch.zeros(1, self.batch_size, self.hidden_dim).cuda())
		return (h0, c0)

	def forward(self, sentence):
		pdb.set_trace()
		embeds = self.embedding(sentence)
		x = embeds.view(len(sentence), self.batch_size, -1)
		lstm_out, self.hidden = self.lstm(x, self.hidden)
		y  = self.hidden2label(lstm_out[-1])
		return y