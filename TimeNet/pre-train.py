import pandas
import numpy
import torch
import torch.optim as optim
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import sys
import code
import os
import random
from sklearn.preprocessing import normalize
from timenet import TimeNet
import math

# PARAMS:
CUDA = True
EPOCHS = 3000
BATCH_STEP_SIZE = 256


print('preparing data')
datasetTrain = []
datasetTest = []
Train_datasets=[
    'ItalyPowerDemand',
    'SonyAIBORobotSurface',
    'FacesUCR',
    'Gun Point',
    'WordSynonyms',
    'Lightning7',
    'DiatomSizeReduction',
    'Ham',
    'ShapeletSim',
    'SonyAIBORobotSurfaceII',
    'TwoLeadECG',
    'Plane',
    'ArrowHead',
    'ToeSegmentation1',
    'ToeSegmentation2',
    'OSULeaf',
    'Fish',
    'ShapesAll'
]
# availableDatasets = [
#     "synthetic_control",
#     "PhalangesOutlinesCorrect",
#     "DistalPhalanxOutlineAgeGroup",
#     "DistalPhalanxOutlineCorrect",
#     "DistalPhalanxTW",
#     "MiddlePhalanxOutlineAgeGroup",
#     "MiddlePhalanxOutlineCorrect",
#     "MiddlePhalanxTW",
#     "ProximalPhalanxOutlineAgeGroup",
#     "ProximalPhalanxOutlineCorrect",
#     "ProximalPhalanxTW",
#     "ElectricDevices",
#     "MedicalImages",
#     "Swedish_Leaf",
#     "Two Patterns",
#     "ECG5000",
#     "ECGFiveDays",
#     "Wafer",
#     "ChlorineConcentration",
#     "Adiac",
#     "Strawberry",
#     "Cricket_X",
#     "Cricket_Y",
#     "Cricket_Z",
#     "uWaveGestureLibrary_X",
#     "uWaveGestureLibrary_Y",
#     "uWaveGestureLibrary_Z",
#     "yoga",
#     "FordA",
#     "FordB",
# ]

Test_datasets=[
    'MoteStrain',
    'Trace',
    'Herring',
    'CBF',
    'Symbols',
    'Earthquakes'
]
data_dir='../UCRArchive_2018'
for directory, subdirectories, files in os.walk(data_dir):
    for file in files:
        filename = os.path.join(directory, file)
        if any(datasetName in file for datasetName in Train_datasets):
            filedata = pandas.read_csv(filename, header=None, sep='	', ).values[:,1:]
            normalize(filedata, norm='l2')
            # local_dataset = []
            # for i in range(filedata.shape[0]):
            #     local_dataset.append( numpy.expand_dims(filedata[i], axis=2) )
            datasetTrain.append(filedata[:,:,numpy.newaxis])

        if any(datasetName in file for datasetName in Test_datasets):
            filedata = pandas.read_csv(filename, header=None, sep='	', ).values[:,1:]
            normalize(filedata, norm='l2')
            # local_dataset = []
            # for i in range(filedata.shape[0]):
            #     local_dataset.append(numpy.expand_dims(filedata[i], axis=2) )
            datasetTest.append(filedata[:,:,numpy.newaxis])

net = TimeNet()
net.double()
# net.load_state_dict(torch.load('models/model-sae3-checkpoint.pt'))
if CUDA: net.cuda()

criterion = nn.MSELoss()
learning_rate = 0.006
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

lowest_loss_per_epoch = 1000

lossArr=numpy.array([])

total_iterations=0
if not os.path.exists('./models'):
    os.makedirs('./models')
for i in range(EPOCHS):
    loss_to_show = 0
    losses_per_epoch = numpy.array([])
    random.shuffle(datasetTrain)

    for file in range(len(datasetTrain)):
        current_dataset = datasetTrain[file]
        setps_per_epoch = math.ceil(len(current_dataset) / BATCH_STEP_SIZE)


        for j in range(setps_per_epoch):
            j = j * BATCH_STEP_SIZE
            starts_from = j
            ends_at = min(j + BATCH_STEP_SIZE, len(current_dataset))

            local_batch_x = Variable( torch.DoubleTensor( current_dataset[starts_from : ends_at] ), requires_grad=False )
            if CUDA: local_batch_x = local_batch_x.cuda()

            optimizer.zero_grad()
            local_batch_x_reversed = local_batch_x.detach().cpu().numpy() if CUDA else local_batch_x.detach().numpy()
            local_batch_x_reversed = numpy.flip(local_batch_x_reversed, axis=1).copy()
            local_batch_x_reversed = Variable( torch.from_numpy(local_batch_x_reversed).double(), requires_grad=False )
            if CUDA: local_batch_x_reversed = local_batch_x_reversed.cuda()

            predicted,embeding_ = net(local_batch_x, local_batch_x_reversed)

            optimizer.zero_grad()
            loss = criterion(predicted, local_batch_x_reversed)

            loss_to_show = loss.detach().cpu().item() if CUDA else loss.detach().item()
            lossArr = numpy.append(lossArr, [loss_to_show], axis=0)
            losses_per_epoch = numpy.append(losses_per_epoch, [loss_to_show], axis=0)

            print("epoch: %s, total_i: %s , current_length: %s, file: %s/%s, step: %s/%s, loss: %s" %
                  (i, total_iterations ,current_dataset[0].shape[0],file + 1, len(datasetTrain), ends_at, len(current_dataset), loss_to_show))
            loss.backward()
            optimizer.step()
            total_iterations += 1

    loss_per_epoch = numpy.average(losses_per_epoch)
    print('current_loss: %s, lowest_loss: %s' % (loss_per_epoch, lowest_loss_per_epoch))
    if loss_per_epoch < lowest_loss_per_epoch:
        lowest_loss_per_epoch = loss_per_epoch
        if CUDA:
            torch.save(net.cpu().state_dict(), 'models/model-sae3-lowestlossperepoch-%s-epoch-%s.pt' % (lowest_loss_per_epoch, i))
            net.cuda()
        else:
            torch.save(net.state_dict(), 'models/model-sae3-lowestlossperepoch-%s-epoch-%s.pt' % (lowest_loss_per_epoch, i))
    # torch.save(net.state_dict(), 'models/model-sae3-checkpoint.pt')
    numpy.savetxt('loss3.csv', lossArr, delimiter=',')

if CUDA:
    net.cpu()
torch.save(net.state_dict(), 'models/model-sae3.pt')

# code.interact(local=locals())
