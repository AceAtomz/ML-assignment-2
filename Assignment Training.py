import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
import math
import sys
from os import path

inputFile = path.dirname(__file__)
labelFile = path.dirname(__file__)


class NeuralNet():
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
        # for i in self.inputArr:
        #     print(self.inputArr[i])
        #     print(self.labelArr[i])


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

nn = NeuralNet()
#nn.readFromFile()
#nn.toStringInputs()

import torch
print(torch.cuda.is_available())