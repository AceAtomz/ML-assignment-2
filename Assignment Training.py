import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
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
        print(self.inputArr)
        print(self.labelArr)

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

#inputReader = fileReader()
#inputReader.readFromFile()
#inputReader.toStringInputs()

torch.manual_seed(0)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        self.conv1 = nn.Linear(5, 6)
        self.conv2 = nn.Linear(6, 6)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.learning_rate = 0.01

    def forward(self, x):
        x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
        #x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()
params = list(net.parameters()) #10 params

input = torch.randn(1, 1, 32, 32)
output = net(input)
print("first output", output)

target = torch.randn(10)  # a dummy target, for example
target = target.view(1, -1)  # make it the same shape as output
criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=0.01)

for i in range(10):
     # in your training loop:
    optimizer.zero_grad()   # zero the gradient buffers
    output = net(input)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()    # Does the update

print("final output", output)
print("final loss", loss.item())