import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from os import path


inputFile = path.dirname(__file__)
labelFile = path.dirname(__file__)

class fileReader():
    def __init__(self):
        self.inputFile = open(path.join(inputFile, "./assignment_data/inputs.txt"),"r", encoding="utf-8")
        self.labelFile = open(path.join(labelFile, "./assignment_data/labels.txt"),"r", encoding="utf-8")
        self.inputArr = np.empty((2000, 1, 2352), dtype=np.float32)    #2000 lines, 2352 variables per line
        self.labelArr = np.empty((2000), dtype=np.int64)            #0-9 range

    def readFromFile(self):
        count = 0
        for inputLine in self.inputFile:
            inputList = inputLine.split(" ")
            for i in range(len(inputList)):
                self.inputArr[count][0][i] = inputList[i]

            count+=1

        count = 0
        for labelLine in self.labelFile:
            self.labelArr[count] = labelLine
            count+=1

    def toStringInputs(self):
        return self.inputArr, self.labelArr

inputReader = fileReader()
inputReader.readFromFile()
npinputs, nplabels = inputReader.toStringInputs()
inputs = torch.from_numpy(npinputs)
labels = torch.from_numpy(nplabels)

torch.manual_seed(0)

batch_size = 2000
classes = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(2352, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
        x = torch.sigmoid(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)
epochs = 4
print("Training")

for epoch in range(epochs):  # loop over the dataset multiple times
    net.train()
    for i in range(batch_size):
        epochInput = inputs[i]
        epochLabel = labels[i]

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        epochOutput = net(epochInput)
        epochOutput = epochOutput[-1, :]
        loss = criterion(epochOutput, epochLabel)
        loss.backward()
        optimizer.step()

print('Finished Training with', epochs, 'epochs')

output = net(inputs)
newOutput = torch.tensor(np.empty((2000), dtype=np.int8))
for count, i in enumerate(output):
    argmax = torch.argmax(i)
    newOutput[count] = argmax
    output[count] = output[argmax]

print("Final output", newOutput, newOutput.size())


#Testing accuracy
correct = 0

for i in range(batch_size):
    if(newOutput[i]==labels[i]):
        correct += 1

print("model accuracy:", round(correct/batch_size, 3))


#save model 
PATH = "state_dict_model.pt"
torch.save(net.state_dict(), PATH)
print("saved")