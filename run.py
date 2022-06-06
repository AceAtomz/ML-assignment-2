import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
from os import path

PATH = "state_dict_model.pt"

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(2352, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = Net()

model.load_state_dict(torch.load(PATH))
model.eval()

#inputFile = path.dirname(__file__)
#inputFile = open(path.join(inputFile, "./assignment_data/inputs.txt"),"r", encoding="utf-8")

def main():
    #load stdin
    test_data = np.loadtxt(sys.stdin)
    #test_data = np.loadtxt(inputFile)

    for data_point in test_data:
        input = torch.tensor(np.array([data_point], dtype=np.float32))
        output = model(input)
        prediction = torch.argmax(output)

        # Write stdout
        sys.stdout.write(str(prediction.item()))
        
main()