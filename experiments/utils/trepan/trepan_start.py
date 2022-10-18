import pandas as pd
from trepan_generation import *
import pickle
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch import tensor, from_numpy
#{'criterion': 'entropy', 'max_depth': 400, 'max_features': 'auto', 'min_samples_leaf': 3, 'min_samples_split': 5}
nome = 'adult'
blackbox = 'nn'
class MyNet(nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.layers = 0

        self.lin1 = torch.nn.Linear(self.num_features, 150)
        self.lin2 = torch.nn.Linear(50, 50)
        self.lin3 = torch.nn.Linear(50, 50)

        self.lin4 = torch.nn.Linear(150, 150)

        self.lin5 = torch.nn.Linear(50, 50)
        self.lin6 = torch.nn.Linear(50, 50)
        self.lin10 = torch.nn.Linear(150, self.num_classes)

        self.prelu = nn.PReLU()
        self.dropout = nn.Dropout(0.25)

    def forward(self, xin):
        self.layers = 0

        x = F.relu(self.lin1(xin))
        self.layers += 1

        # x = F.relu(self.lin2(x))
        # self.layers += 1
        for y in range(8):
            x = F.relu(self.lin4(x))
            self.layers += 1

        x = self.dropout(x)

        x = F.relu(self.lin10(x))
        self.layers += 1
        return x
train_set = pd.read_csv('../data/adult_original_train_set.csv')
test_set = pd.read_csv('../data/adult_original_test_set.csv')
train_label = pd.read_csv('../data/adult_original_train_label.csv')
test_label = pd.read_csv('../data/adult_original_test_label.csv')
train_set.pop('Unnamed: 0')
test_set.pop('Unnamed: 0')
bb = pickle.load(open('../blackbox/'+blackbox+'_adult_original.sav', 'rb'))
generator = TrePanGenerator()
gen = generator.generate(train_set.values, oracle=bb, size = 70000)
print(gen, gen.shape)
labels = gen[:, -1]
gen = np.delete(gen, -1, axis=1)
print('alb', labels, labels.shape)
gen = pd.DataFrame(gen)
labels = pd.DataFrame(labels)
#filename = '../data/diva_trepan_dt.sav'
#f = open(filename,'wb')
#pickle.dump(gen,f)
gen.to_csv('../data/adult_trepan_dt_data.csv')
labels.to_csv('../data/adult_trepan_dt_labels.csv')

