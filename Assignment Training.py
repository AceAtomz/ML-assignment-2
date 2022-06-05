import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import sys
from os import path

inputFile = path.dirname(__file__)
labelFile = path.dirname(__file__)

class fileReader():
    def __init__(self):
        self.inputFile = open(path.join(inputFile, "./assignment_data/inputs.txt"),"r", encoding="utf-8")
        self.labelFile = open(path.join(labelFile, "./assignment_data/labels.txt"),"r", encoding="utf-8")
        self.inputArr = np.empty((2000, 2352), dtype=np.float64)    #2000 lines, 2352 variables per line
        self.labelArr = np.empty((2000), dtype=np.int8)         #0-9 range

    def readFromFile(self):
        count = 0
        for inputLine in self.inputFile:
            inputList = inputLine.split(" ")
            for i in range(len(inputList)):
                self.inputArr[count][i], inputList[i]

            count+=1

        count = 0
        for labelLine in self.labelFile:
            self.labelArr[count] = labelLine
            count+=1

    def toStringInputs(self):
        return self.inputArr, self.labelArr
        

#----------------------------------------------------------------------------------------------------------
#TODO: ins and outs
#sys.stdout.write(str(num))
def main():
    #load stdin
    np.loadtxt(sys.stdin)
    for line in sys.stdin:
        if (line == ""):
            break

        # Write stdout
        output = line
        sys.stdout.write(str(output))
#main()
#----------------------------------------------------------------------------------------------------------

inputReader = fileReader()
inputReader.readFromFile()
inputs, labels = inputReader.toStringInputs()

torch.manual_seed(0)

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = labels.size
classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # an affine operation: y = Wx + b
        self.fc1 = torch.nn.Sequential(nn.Linear(1024, 120))
        with torch.no_grad():
            self.fc1[0].weight[0, 0] = 2.
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.learning_rate = 0.01

    def forward(self, x):
        x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

output = net(inputs)
print("First output", output)
print("Training")

for epoch in range(2):  # loop over the dataset multiple times
    # zero the parameter gradients
    optimizer.zero_grad()

    # forward + backward + optimize
    outputs = net(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    # print statistics
    print(loss.item())

print('Finished Training')

output = net(inputs)
print("Final output", output)
loss = criterion(output, labels)
print("Final loss", loss.item())