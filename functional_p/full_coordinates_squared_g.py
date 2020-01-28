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

def mask_data(x_train, x_test):
	mask_train, mask_test = [], []
	for i in range(len(x_train)):
		label=x_train[i][D-1]
		x_train[i][D-1]=0
		mask_train.append((x_train[i].copy(),label))
		x_train[i][D-1]=label
		label=x_train[i][0]
		x_train[i][0]=0
		mask_train.append((x_train[i].copy(),label))
		x_train[i][0]=label

	for i in range(len(x_test)):
		label=x_test[i][D-1]
		x_test[i][D-1]=0
		mask_test.append((x_test[i].copy(),label))
		x_test[i][D-1]=label
		label=x_test[i][0]
		x_test[i][0]=0
		mask_test.append((x_test[i].copy(),label))
		x_test[i][0]=label

	random.shuffle(mask_train)
	random.shuffle(mask_test)

	mask_x_train , mask_y_train , mask_x_test , mask_y_test=[],[],[],[]

	for i in range(len(mask_train)):
		mask_x_train.append(mask_train[i][0])
		mask_y_train.append(mask_train[i][1])

	for i in range(len(mask_test)):
		mask_x_test.append(mask_test[i][0])
		mask_y_test.append(mask_test[i][1])

	return mask_x_train, mask_x_test, mask_y_train, mask_y_test


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
	x_train , y_train , x_test , y_test=dataloader()

	max_num=1000
	x_train=x_train[0:max_num]
	y_train=y_train[0:max_num]
	print(len(x_train),len(x_test))


	mask_x_train, mask_x_test, mask_y_train, mask_y_test=mask_data(x_train, x_test)
	
	mask_x_train = torch.FloatTensor(mask_x_train)
	mask_y_train = torch.FloatTensor(mask_y_train)
	mask_x_test = torch.FloatTensor(mask_x_test)
	mask_y_test = torch.FloatTensor(mask_y_test)


	x_train = torch.FloatTensor(x_train)
	y_train = torch.FloatTensor(y_train)
	x_test = torch.FloatTensor(x_test)
	y_test = torch.FloatTensor(y_test)

	model = Feedforward(D,hidden_H)
	g_model = G_network(H,1)
	f_model = F_network(H,1)
	criterion = torch.nn.MSELoss()
	params = list(model.parameters()) + list(g_model.parameters())
	optimizer = torch.optim.SGD(params, lr=0.02, momentum=0.9)
	

	model.eval()
	g_model.eval()
	y_pred = model(mask_x_test)
	y_pred = g_model(torch.mul(y_pred, y_pred))
	loss = criterion(y_pred.squeeze(), mask_y_test)
	print('Representation loss: {}'.format(loss.item()))
	
	model.train()
	g_model.train()
	epoch = 20000
	for epoch in range(epoch):
		optimizer.zero_grad()
		y_pred = model(mask_x_test)
		y_pred = g_model(torch.mul(y_pred, y_pred))
		loss = criterion(y_pred.squeeze(), mask_y_train)
		if (epoch % 100) == 0:
			print('Epoch {}: train loss: {}'.format(epoch, loss.item()))
		loss.backward()
		optimizer.step()
		
		if (epoch % 100) == 0:
			model.eval()
			g_model.eval()
			y_pred = model(mask_x_test)
			y_pred = g_model(torch.mul(y_pred, y_pred))
			loss = criterion(y_pred.squeeze(), mask_y_test)
			print('Representation loss: {}'.format(loss.item()))
			model.train()
			g_model.train()

	model.eval()
	f_model.eval()
	y_pred = f_model(model(x_test))
	loss = criterion(y_pred.squeeze(), y_test)
	print('Classification loss: {}'.format(loss.item()))

	params2 = list(f_model.parameters()) + list(model.parameters())
	optimizer2 = torch.optim.SGD(params2, lr=0.002, momentum=0.9)

	#for param in model.parameters():
	#	param.requires_grad = False

	model.train()
	f_model.train()
	epoch = 20000
	for epoch in range(epoch):
		optimizer2.zero_grad()
		y_pred = f_model(model(x_train))
		loss = criterion(y_pred.squeeze(), y_train)
		if (epoch % 100) == 0:
			print('Epoch {}: train loss: {}'.format(epoch, loss.item()))
		loss.backward()
		optimizer2.step()
		
		if (epoch % 100) == 0:
			model.eval()
			f_model.eval()
			y_pred = f_model(model(x_test))
			loss = criterion(y_pred.squeeze(), y_test)
			print('Classification loss: {}'.format(loss.item()))
			model.train()
			f_model.train()

if __name__ == "__main__":
	main()	

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import random

# D=10
# hidden_H=50
# H=3

# def dataloader():
# 	f=open("data/train.tsv","r")
# 	g=open("data/test.tsv","r")
# 	x_train , y_train , x_test , y_test=[],[],[],[]
# 	for line in f:
# 		parts=line.strip("\n").split("\t")
# 		feature=[float(i) for i in parts[0].split()]
# 		label=float(parts[1])
# 		x_train.append(feature)
# 		y_train.append(label)

# 	for line in g:
# 		parts=line.strip("\n").split("\t")
# 		feature=[float(i) for i in parts[0].split()]
# 		label=float(parts[1])
# 		x_test.append(feature)
# 		y_test.append(label)

# 	return x_train , y_train , x_test , y_test

# def mask_data(x_train, x_test):
# 	mask_train, mask_test = [], []
# 	for i in range(len(x_train)):
# 		label=x_train[i][D-1]
# 		x_train[i][D-1]=0
# 		mask_train.append((x_train[i].copy(),label))
# 		x_train[i][D-1]=label
# 		label=x_train[i][0]
# 		x_train[i][0]=0
# 		mask_train.append((x_train[i].copy(),label))
# 		x_train[i][0]=label

# 	for i in range(len(x_test)):
# 		label=x_test[i][D-1]
# 		x_test[i][D-1]=0
# 		mask_test.append((x_test[i].copy(),label))
# 		x_test[i][D-1]=label
# 		label=x_test[i][0]
# 		x_test[i][0]=0
# 		mask_test.append((x_test[i].copy(),label))
# 		x_test[i][0]=label

# 	random.shuffle(mask_train)
# 	random.shuffle(mask_test)

# 	mask_x_train , mask_y_train , mask_x_test , mask_y_test=[],[],[],[]

# 	for i in range(len(mask_train)):
# 		mask_x_train.append(mask_train[i][0])
# 		mask_y_train.append(mask_train[i][1])

# 	for i in range(len(mask_test)):
# 		mask_x_test.append(mask_test[i][0])
# 		mask_y_test.append(mask_test[i][1])

# 	return mask_x_train, mask_x_test, mask_y_train, mask_y_test


# class Feedforward(torch.nn.Module):
#     def __init__(self, input_size, hidden_size):
#         super(Feedforward, self).__init__()
#         self.input_size = input_size
#         self.hidden_size  = hidden_size
#         self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
#         self.relu = torch.nn.ReLU()
#         self.fc2 = torch.nn.Linear(self.hidden_size, self.hidden_size)
#         #self.fc3 = torch.nn.Linear(self.hidden_size, self.hidden_size)
#         self.fc4 = torch.nn.Linear(self.hidden_size, H)

#     def forward(self, x):
#         hidden = self.fc1(x)
#         relu = self.relu(hidden)
#         hidden = self.fc2(relu)
#         relu = self.relu(hidden)
#         #hidden = self.fc3(relu)
#         #relu = self.relu(hidden)
#         output = self.fc4(relu)
#         return output

# class F_network(torch.nn.Module):
#     def __init__(self, input_size, output_size):
#         super(F_network, self).__init__()
#         self.input_size = input_size
#         self.output_size  = output_size
#         self.fc = torch.nn.Linear(self.input_size, self.output_size)

#     def forward(self, x):
#         output = self.fc(x)
#         return output

# def main():
# 	x_train , y_train , x_test , y_test=dataloader()

# 	max_num=1000
# 	x_train=x_train[0:max_num]
# 	y_train=y_train[0:max_num]
# 	print(len(x_train),len(x_test))


# 	mask_x_train, mask_x_test, mask_y_train, mask_y_test=mask_data(x_train, x_test)
	
# 	mask_x_train = torch.FloatTensor(mask_x_train)
# 	mask_y_train = torch.FloatTensor(mask_y_train)
# 	mask_x_test = torch.FloatTensor(mask_x_test)
# 	mask_y_test = torch.FloatTensor(mask_y_test)


# 	x_train = torch.FloatTensor(x_train)
# 	y_train = torch.FloatTensor(y_train)
# 	x_test = torch.FloatTensor(x_test)
# 	y_test = torch.FloatTensor(y_test)

# 	model = Feedforward(D,hidden_H)
# 	f_model = F_network(H,1)
# 	criterion = torch.nn.MSELoss()
# 	params = list(model.parameters())
# 	optimizer = torch.optim.SGD(params, lr=0.02, momentum=0.9)
	

# 	model.eval()
# 	y_pred = model(mask_x_test)
# 	y_pred= torch.sum(torch.mul(y_pred, y_pred),1)	
# 	loss = criterion(y_pred.squeeze(), mask_y_test)
# 	print('Representation loss: {}'.format(loss.item()))
	
# 	model.train()
# 	epoch = 20000
# 	for epoch in range(epoch):
# 		optimizer.zero_grad()
# 		y_pred = model(mask_x_train)
# 		y_pred= torch.sum(torch.mul(y_pred, y_pred),1)	
# 		loss = criterion(y_pred.squeeze(), mask_y_train)
# 		if (epoch % 100) == 0:
# 			print('Epoch {}: train loss: {}'.format(epoch, loss.item()))
# 		loss.backward()
# 		optimizer.step()
		
# 		if (epoch % 100) == 0:
# 			model.eval()
# 			y_pred = model(mask_x_test)
# 			y_pred= torch.sum(torch.mul(y_pred, y_pred),1)	
# 			loss = criterion(y_pred.squeeze(), mask_y_test)
# 			print('Representation loss: {}'.format(loss.item()))
# 			model.train()

# 	model.eval()
# 	f_model.eval()
# 	y_pred = f_model(model(x_test))
# 	loss = criterion(y_pred.squeeze(), y_test)
# 	print('Classification loss: {}'.format(loss.item()))

# 	params2 = list(f_model.parameters()) + list(model.parameters())
# 	optimizer2 = torch.optim.SGD(params2, lr=0.02, momentum=0.9)

# 	for param in model.parameters():
# 		param.requires_grad = False

# 	model.train()
# 	f_model.train()
# 	epoch = 20000
# 	for epoch in range(epoch):
# 		optimizer2.zero_grad()
# 		y_pred = f_model(model(x_train))
# 		loss = criterion(y_pred.squeeze(), y_train)
# 		if (epoch % 100) == 0:
# 			print('Epoch {}: train loss: {}'.format(epoch, loss.item()))
# 		loss.backward()
# 		optimizer2.step()
		
# 		if (epoch % 100) == 0:
# 			model.eval()
# 			f_model.eval()
# 			y_pred = f_model(model(x_test))
# 			loss = criterion(y_pred.squeeze(), y_test)
# 			print('Classification loss: {}'.format(loss.item()))
# 			model.train()
# 			f_model.train()

# if __name__ == "__main__":
# 	main()	
