from torch import tanh
from torch.nn import Module, Linear, Dropout
from torch.nn.init import xavier_normal_
from torch.nn.functional import softmax


class Net(Module):
    def __init__(self):
        super().__init__()
        # for xavier initialization
        self.fc1 = Linear(236, 128)
        xavier_normal_(self.fc1.weight)
        self.drop = Dropout(0.3)
        self.fc4 = Linear(128, 30)
        xavier_normal_(self.fc4.weight)

    def forward(self, x):
        # Pass the input tensor through each of our operations
        x = tanh(self.fc1(x))
        x = self.fc4(self.drop(x))
        return softmax(x, dim=1)

