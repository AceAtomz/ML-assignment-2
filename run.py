import torch
import numpy as np
import sys

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = torch.nn.Linear(2352, 120)
        self.fc2 = torch.nn.Linear(120, 84)
        self.fc3 = torch.nn.Linear(84, 10)

    def forward(self, x):
        x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def main():
    PATH = "state_dict_model.pt"
    model = Net()
    model.load_state_dict(torch.load(PATH))
    model.eval()

    test_data = np.loadtxt(sys.stdin)
    shape = test_data.shape
    if(shape != (2352,)):
        for data_point in test_data:
            input = torch.tensor(np.array([data_point], dtype=np.float32))
            output = model(input)
            prediction = torch.argmax(output)
            sys.stdout.write(str(prediction.item()))
    else:
        input = torch.tensor(np.array([test_data], dtype=np.float32))
        output = model(input)
        prediction = torch.argmax(output)
        sys.stdout.write(str(prediction.item()))

if (__name__ == '__main__'):
    main()