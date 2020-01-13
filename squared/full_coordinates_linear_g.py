import torch
import torch.nn as nn
import torch.nn.functional as F

D=10
H=100

def dataloader():
	f=open("data/train.tsv","r")
	g=open("data/test.tsv","r")
	x_train , y_train , x_test , y_test=[],[],[],[]
	x_D_train , x_D_test=[],[]
	for line in f:
		parts=line.strip("\n").split("\t")
		feature=[float(i) for i in parts[0].split()]
		x_D_train.append(feature[-1])
		feature=feature[:-1]
		label=float(parts[1])
		x_train.append(feature)
		y_train.append(label)

	for line in g:
		parts=line.strip("\n").split("\t")
		feature=[float(i) for i in parts[0].split()]
		x_D_test.append(feature[-1])
		feature=feature[:-1]
		label=float(parts[1])
		x_test.append(feature)
		y_test.append(label)

	return x_train , y_train , x_test , y_test, x_D_train , x_D_test

class Feedforward(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Feedforward, self).__init__()
        self.input_size = input_size
        self.hidden_size  = hidden_size
        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.fc3 = torch.nn.Linear(self.hidden_size, 2)

    def forward(self, x):
        hidden = self.fc1(x)
        relu = self.relu(hidden)
        hidden = self.fc2(relu)
        relu = self.relu(hidden)
        output = self.fc3(relu)
        return output

class G_network(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(G_network, self).__init__()
        self.input_size = input_size
        self.output_size  = output_size
        self.fc = torch.nn.Linear(self.input_size, self.output_size)

    def forward(self, x):
        output = self.fc(x)
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
	x_train , y_train , x_test , y_test, x_D_train , x_D_test=dataloader()

	max_num=50
	x_train=x_train[0:max_num]
	y_train=y_train[0:max_num]
	x_D_train=x_D_train[0:max_num]
	print(len(x_train),len(x_test))
	
	x_train = torch.FloatTensor(x_train)
	y_train = torch.FloatTensor(y_train)
	x_test = torch.FloatTensor(x_test)
	y_test = torch.FloatTensor(y_test)
	x_D_train = torch.FloatTensor(x_D_train)
	x_D_test = torch.FloatTensor(x_D_test)

	model = Feedforward(D,H)
	g_model = G_network(2,1)
	f_model = F_network(2,1)
	criterion = torch.nn.MSELoss()
	params = list(model.parameters()) + list(g_model.parameters())
	optimizer = torch.optim.SGD(params, lr=0.1, momentum=0.9)
	params2 = list(f_model.parameters()) + list(model.parameters())
	optimizer2 = torch.optim.SGD(params2, lr=0.01, momentum=0.9)
	pad= nn.ConstantPad1d((0, 1), 0)

	model.eval()
	g_model.eval()
	y_pred = g_model(model(pad(x_test)))
	loss = criterion(y_pred.squeeze(), x_D_test)
	print('Representation loss: {}'.format(loss.item()))
	
	model.train()
	g_model.train()
	epoch = 3000
	for epoch in range(epoch):
		optimizer.zero_grad()
		y_pred = g_model(model(pad(x_train)))
		loss = criterion(y_pred.squeeze(), x_D_train)
		if (epoch % 300) == 0:
			print('Epoch {}: train loss: {}'.format(epoch, loss.item()))
		loss.backward()
		optimizer.step()
		
		if (epoch % 300) == 0:
			model.eval()
			g_model.eval()
			y_pred = g_model(model(pad(x_test)))
			loss = criterion(y_pred.squeeze(), x_D_test)
			print('Representation loss: {}'.format(loss.item()))
			model.train()
			g_model.train()

	for param in model.parameters():
		param.requires_grad = False

	model.eval()
	f_model.eval()
	y_pred = f_model(model(torch.cat((x_test,torch.unsqueeze(x_D_test,1)),-1)))
	loss = criterion(y_pred.squeeze(), y_test)
	print('Classification loss: {}'.format(loss.item()))
	
	model.train()
	f_model.train()
	epoch = 1500
	for epoch in range(epoch):
		optimizer2.zero_grad()
		y_pred = f_model(model(torch.cat((x_train,torch.unsqueeze(x_D_train,1)),-1)))
		loss = criterion(y_pred.squeeze(), y_train)
		if (epoch % 300) == 0:
			print('Epoch {}: train loss: {}'.format(epoch, loss.item()))
		loss.backward()
		optimizer2.step()
		
		if (epoch % 300) == 0:
			model.eval()
			f_model.eval()
			y_pred = f_model(model(torch.cat((x_test,torch.unsqueeze(x_D_test,1)),-1)))
			loss = criterion(y_pred.squeeze(), y_test)
			print('Classification loss: {}'.format(loss.item()))
			model.train()
			f_model.train()

if __name__ == "__main__":
	main()	