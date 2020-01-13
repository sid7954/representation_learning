import torch
import torch.nn as nn
import torch.nn.functional as F

D=10
H=300
H_mask=150


def dataloader():
	f=open("data/train.tsv","r")
	g=open("data/test.tsv","r")
	x_train , y_train , x_test , y_test=[],[],[],[]
	for line in f:
		parts=line.strip("\n").split("\t")
		feature=[float(i) for i in parts[0].split()]
		feature=feature[:-1]
		label=float(parts[1])
		x_train.append(feature)
		y_train.append(label)

	for line in g:
		parts=line.strip("\n").split("\t")
		feature=[float(i) for i in parts[0].split()]
		feature=feature[:-1]
		label=float(parts[1])
		x_test.append(feature)
		y_test.append(label)

	return x_train , y_train , x_test , y_test

class MaskPredictor(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MaskPredictor, self).__init__()
        self.input_size = input_size
        self.hidden_size  = hidden_size
        self.fc1 = torch.nn.Linear(self.input_size-1, self.hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(self.hidden_size, 1)

    def forward(self, x):
        hidden = self.fc1(x)
        relu = self.relu(hidden)
        output = self.fc2(relu)
        return output

class Feedforward(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Feedforward, self).__init__()
        self.input_size = input_size
        self.hidden_size  = hidden_size
        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(self.hidden_size, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        hidden = self.fc1(x)
        relu = self.relu(hidden)
        output = self.fc2(relu)
        #output = torch.nn.functional.softmax(output,dim=1)
        output=self.sigmoid(output)
        return output

def main():
	x_train , y_train , x_test , y_test=dataloader()

	x_train=x_train[0:15]
	y_train=y_train[0:15]
	print(len(x_train),len(x_test))
	
	x_train = torch.FloatTensor(x_train)
	y_train = torch.FloatTensor(y_train)
	x_test = torch.FloatTensor(x_test)
	y_test = torch.FloatTensor(y_test)

	masked_model= MaskPredictor(D,H_mask)	
	model = Feedforward(D,H)
	criterion = torch.nn.BCELoss()
	params = list(masked_model.parameters()) + list(model.parameters())
	optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9)

	model.eval()
	x_mask=masked_model(x_test)
	x=torch.cat((x_test,x_mask),-1)
	y_pred = model(x)
	y_pred = (y_pred > 0.5).float()
	total = y_test.size(0)
	correct = (y_pred.squeeze() == y_test).sum().item()
	print('Accuracy: %f %%' % (100 * correct / total))

	model.train()
	epoch = 3500
	max_accuracy=0
	best_epoch=0
	for epoch in range(epoch):
		optimizer.zero_grad()
		x_mask=masked_model(x_train)
		x=torch.cat((x_train,x_mask),-1)
		y_pred = model(x)
		loss = criterion(y_pred.squeeze(), y_train)
		if (epoch % 10)==0:
			print('Epoch {}: train loss: {}'.format(epoch, loss.item()))
		loss.backward()
		optimizer.step()
		
		if (epoch % 10)==0:
			model.eval()
			x_mask=masked_model(x_test)
			x=torch.cat((x_test,x_mask),-1)
			y_pred = model(x)
			y_pred = (y_pred > 0.5).float()
			total = y_test.size(0)
			correct = (y_pred.squeeze() == y_test).sum().item()
			accuracy= (100 * correct / total)
			if accuracy > max_accuracy:
				max_accuracy = accuracy
				best_epoch = epoch
			#print('Accuracy: %f %%' % (100 * correct / total))
			model.train()
	print(max_accuracy, best_epoch)


if __name__ == "__main__":
	main()	