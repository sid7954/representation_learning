import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np

D=10
hidden_H=50
H=3

def dataloader():
	f=open("data/train.tsv","r")
	g=open("data/test.tsv","r")
	x_train , y_train , x_test , y_test=[],[],[],[]
	for line in f:
		parts=line.strip("\n").split("\t")
		feature=[float(i) for i in parts[0].split()]
		label=float(parts[1])
		x_train.append(feature)
		y_train.append(label)

	for line in g:
		parts=line.strip("\n").split("\t")
		feature=[float(i) for i in parts[0].split()]
		label=float(parts[1])
		x_test.append(feature)
		y_test.append(label)

	return x_train , y_train , x_test , y_test

class Feedforward(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Feedforward, self).__init__()
        self.input_size = input_size
        self.hidden_size  = hidden_size
        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(self.hidden_size, self.hidden_size)
        #self.fc3 = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.fc4 = torch.nn.Linear(self.hidden_size, H)

    def forward(self, x):
        hidden = self.fc1(x)
        relu = self.relu(hidden)
        hidden = self.fc2(relu)
        relu = self.relu(hidden)
        #hidden = self.fc3(relu)
        #relu = self.relu(hidden)
        output = self.fc4(relu)
        return output

class F_network(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(F_network, self).__init__()
        self.input_size = input_size
        self.output_size  = output_size
        self.fc = torch.nn.Linear(self.input_size, self.output_size)

    def forward(self, x):
        output = self.fc(x)
        return output

def main():
	x_train , y_train , x_test , y_test=dataloader()

	max_num=1000
	x_train=x_train[0:max_num]
	y_train=y_train[0:max_num]
	print(len(x_train),len(x_test))

	x_train = torch.FloatTensor(x_train)
	y_train = torch.FloatTensor(y_train)
	x_test = torch.FloatTensor(x_test)
	y_test = torch.FloatTensor(y_test)

	all_weights=np.zeros([50,3253])
	all_losses=[]

	for iter in range(50):
		print(iter)
		random.seed(iter)

		model = Feedforward(D,hidden_H)
		f_model = F_network(H,1)
		criterion = torch.nn.MSELoss()
		params2 = list(f_model.parameters()) + list(model.parameters())
		optimizer2 = torch.optim.SGD(params2, lr=0.002, momentum=0.9)
		#optimizer2 = torch.optim.Adam(params2, lr=0.0001)

		model.train()
		f_model.train()
		epoch = 20000
		for epoch in range(epoch):
			optimizer2.zero_grad()
			y_pred = f_model(model(x_train))
			loss = criterion(y_pred.squeeze(), y_train)
			# if (epoch % 100) == 0:
			# 	print('Epoch {}: train loss: {}'.format(epoch, loss.item()))
			loss.backward()
			optimizer2.step()
			
			# if (epoch % 100) == 0:
			# 	model.eval()
			# 	f_model.eval()
			# 	y_pred = f_model(model(x_test))
			# 	loss = criterion(y_pred.squeeze(), y_test)
			# 	print('Classification loss: {}'.format(loss.item()))
			# 	model.train()
			# 	f_model.train()
		model.eval()
		f_model.eval()
		y_pred = f_model(model(x_test))
		loss = criterion(y_pred.squeeze(), y_test)
		#print('Classification loss: {}'.format(loss.item()))
		all_losses.append(loss)

		data_dict = model.state_dict()
		weights=[]
		weights.extend(data_dict['fc1.weight'].numpy().flatten())
		weights.extend(data_dict['fc1.bias'].numpy().flatten())
		weights.extend(data_dict['fc2.weight'].numpy().flatten())
		weights.extend(data_dict['fc2.bias'].numpy().flatten())
		weights.extend(data_dict['fc4.weight'].numpy().flatten())
		weights.extend(data_dict['fc4.bias'].numpy().flatten())
		all_weights[iter]=weights

	np.save('end_to_end.npy',all_weights)
	all_losses= np.asarray(all_losses)
	np.savetxt('end_to_end_loss.csv', all_losses, delimiter='\n')

if __name__ == "__main__":
	main()	