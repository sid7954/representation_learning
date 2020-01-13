import torch
import torch.nn as nn
import torch.nn.functional as F

D=10
H=300

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
        self.fc4 = torch.nn.Linear(self.hidden_size, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        hidden = self.fc1(x)
        relu = self.relu(hidden)
        hidden = self.fc2(relu)
        relu = self.relu(hidden)
        output = self.fc4(relu)
        #output = torch.nn.functional.softmax(output,dim=1)
        output=self.sigmoid(output)
        return output

def main():
	x_train , y_train , x_test , y_test=dataloader()

	x_train=x_train[0:]
	y_train=y_train[0:]
	print(len(x_train),len(x_test))
	
	x_train = torch.FloatTensor(x_train)
	y_train = torch.FloatTensor(y_train)
	x_test = torch.FloatTensor(x_test)
	y_test = torch.FloatTensor(y_test)

	model = Feedforward(D,H)
	criterion = torch.nn.BCELoss()
	params = list(model.parameters())
	optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9)

	model.eval()
	y_pred = model(x_test)
	y_pred = (y_pred > 0.5).float()
	total = y_test.size(0)
	correct = (y_pred.squeeze() == y_test).sum().item()
	print('Accuracy: %f %%' % (100 * correct / total))

	model.train()
	epoch = 1500
	for epoch in range(epoch):
		optimizer.zero_grad()
		y_pred = model(x_train)
		loss = criterion(y_pred.squeeze(), y_train)
		print('Epoch {}: train loss: {}'.format(epoch, loss.item()))
		loss.backward()
		optimizer.step()
		
		model.eval()
		y_pred = model(x_test)
		y_pred = (y_pred > 0.5).float()
		total = y_test.size(0)
		correct = (y_pred.squeeze() == y_test).sum().item()
		print('Accuracy: %f %%' % (100 * correct / total))
		model.train()


if __name__ == "__main__":
	main()	