import json
import math
import numpy as np
from tqdm import tqdm

class QuantNum:
    def __init__(self, max_num):
        self.max_num = max_num
        self.centroids = [] # max_num - 1 centroids

    def train(self, data):
        data = sorted(data)
        lowest_sep = data[math.ceil(len(data) / (10 * self.max_num))]
        highest_sep = data[-math.ceil(len(data) / (10 * self.max_num))]
        self.centroids = np.linspace(lowest_sep, highest_sep, self.max_num - 1).tolist()

    def save(self, filename):
        with open(filename, 'w') as file:
            json.dump(self.centroids, file)

    def load(self, filename):
        with open(filename, 'r') as file:
            self.centroids = json.load(file)

    def evaluate(self, number):
        # Find the nearest centroid
        for i in range(len(self.centroids)):
            if number < self.centroids[i]:
                return i
        return len(self.centroids)
    

def _test():
    # Example Usage
    quantizer = QuantNum(10)
    data = np.random.randn(100000)  # Example data
    quantizer.train(data)
    print(quantizer.centroids)
    quant_data = [quantizer.evaluate(d) for d in data]
    from collections import Counter
    print(sorted(Counter(quant_data).items()))


if __name__ == "__main__":
    _test()