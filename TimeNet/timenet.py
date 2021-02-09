import torch
import torch.optim as optim
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class TimeNet(nn.Module):
    def __init__(self):
        super(TimeNet, self).__init__()

        self.FEATURES = 1
        self.HIDDEN_SIZE = 60
        self.NUM_LAYERS = 3
        self.DROPOUT = 0.4

        self.gru_encoder = nn.GRU(self.FEATURES, self.HIDDEN_SIZE, self.NUM_LAYERS, batch_first=True,
                                  dropout=self.DROPOUT)

        self.gru_cell1 = nn.GRUCell(self.FEATURES, self.HIDDEN_SIZE)
        self.gru_cell2 = nn.GRUCell(self.HIDDEN_SIZE, self.HIDDEN_SIZE)
        self.gru_cell3 = nn.GRUCell(self.HIDDEN_SIZE, self.HIDDEN_SIZE)

        self.linear = nn.Linear(self.HIDDEN_SIZE, self.FEATURES)
        self.dropout = nn.Dropout(self.DROPOUT)

    def forward(self, inputs, outputs):
        seq_length = inputs.size(1)
        outputs_reversed = outputs.transpose(0, 1)

        encoded_seq = self.gru_encoder(inputs)
        encoded_vec = encoded_seq[1]  # [3 x 200 x 32] (hlt)
        # print(encoded_vec[0].shape) # num_layers, batch, hidden size

        preds = []
        de_hidden_1 = encoded_vec[2]
        de_hidden_2 = encoded_vec[1]
        de_hidden_3 = encoded_vec[0]

        # f1 = outputs_reversed[0]
        # de_hidden_1 = self.gru_cell1(f1, de_hidden_1)      # [200 x 32]
        # de_hidden_2 = self.gru_cell2(self.dropout(de_hidden_1), de_hidden_2)      # [200 x 32]
        # de_hidden_3 = self.gru_cell3(self.dropout(de_hidden_2), de_hidden_3)      # [200 x 32]
        # preds += [self.linear(self.dropout(de_hidden_3))]    # [200 x 1]

        for idx in range(seq_length):
            de_hidden_1 = self.gru_cell1(outputs_reversed[idx], de_hidden_1)  # [200 x 32]
            de_hidden_2 = self.gru_cell2(self.dropout(de_hidden_1), de_hidden_2)  # [200 x 32]
            de_hidden_3 = self.gru_cell3(self.dropout(de_hidden_2), de_hidden_3)  # [200 x 32]
            preds += [self.linear(self.dropout(de_hidden_3))]  # [200 x 1]

        preds = torch.stack(preds)  # [2000 x 200 x 1]
        predicted = preds.transpose(0, 1)  # [200 x 2000 x 1]

        return predicted, encoded_vec.transpose(0, 1)


if __name__ == '__main__':
    net = TimeNet()
    # print(net)

    test = torch.rand((2, 5, 1), requires_grad=False)  # batch, seq length, features
    test_reversed = np.fliplr(test).copy()
    test_reversed = Variable(torch.from_numpy(test_reversed), requires_grad=False)

    print(test.size(), test_reversed.size())
    out, encoded_vec = net(test, test_reversed)
    # print(test)
    # print(test_reversed)
    print(out.shape,encoded_vec.shape)
    # print(encoded_vec.shape)