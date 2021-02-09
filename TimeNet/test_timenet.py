import pandas
import numpy
import torch
import torch.optim as optim
from torch.autograd import Variable
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize
import torch.nn as nn
import torch.nn.functional as F
import sys
from tqdm import tqdm
import code
import time
import os
import numpy as np
from torch.utils.data import Dataset,DataLoader
from sklearn.metrics import accuracy_score,f1_score
# PARAMS:
CUDA =True
EPOCHS = 500
BATCH_STEP_SIZE = 64
num_workers=8


class model(nn.Module):
    def __init__(self,nb_classes):
        super(model, self).__init__()

        self.FEATURES = 1
        self.HIDDEN_SIZE = 60
        self.DEPTH = 3
        self.DROPOUT = 0.4

        self.gru_e = nn.GRU(self.FEATURES, self.HIDDEN_SIZE, self.DEPTH, batch_first=True, dropout=self.DROPOUT)

        # self.gru_cell1 = nn.GRUCell(self.FEATURES, self.HIDDEN_SIZE)
        # self.gru_cell2 = nn.GRUCell(self.HIDDEN_SIZE, self.HIDDEN_SIZE)
        # self.gru_cell3 = nn.GRUCell(self.HIDDEN_SIZE, self.HIDDEN_SIZE)

        self.linear = nn.Linear(self.HIDDEN_SIZE, nb_classes)
        # self.dropout = nn.Dropout(p=self.DROPOUT)


    def forward(self, inputs, outputs):
        seq_length = inputs.size()[1]
        outputs_transposed = outputs.transpose(0,1)

        encoded_seq = self.gru_e(inputs)
        encoded_vec = encoded_seq[0]     # [3 x 200 x 60] (bld)

        return self.linear(encoded_vec).permute(0,2,1)

print('preparing data')
datasetTrain = []
datasetTest = []
dataset_names=[#'Adiac', 'ArrowHead', 'Beef', 'BeetleFly', 'BirdChicken', 'Car', 'CBF', 'ChlorineConcentration', 'CinCECGTorso', 'Coffee',
               'Computers', 'CricketX', 'CricketY', 'CricketZ', 'DiatomSizeReduction', 'DistalPhalanxOutlineAgeGroup', 'DistalPhalanxOutlineCorrect', 'DistalPhalanxTW', 'Earthquakes', 'ECG200', 'ECG5000', 'ECGFiveDays', 'ElectricDevices', 'FaceAll', 'FaceFour', 'FacesUCR', 'FiftyWords', 'Fish', 'FordA', 'FordB', 'GunPoint', 'Ham', 'HandOutlines', 'Haptics', 'Herring', 'InlineSkate', 'InsectWingbeatSound', 'ItalyPowerDemand', 'LargeKitchenAppliances', 'Lightning2', 'Lightning7', 'Mallat', 'Meat', 'MedicalImages', 'MiddlePhalanxOutlineAgeGroup', 'MiddlePhalanxOutlineCorrect', 'MiddlePhalanxTW', 'MoteStrain', 'NonInvasiveFetalECGThorax1', 'NonInvasiveFetalECGThorax2', 'OliveOil', 'OSULeaf', 'PhalangesOutlinesCorrect', 'Phoneme', 'Plane', 'ProximalPhalanxOutlineAgeGroup', 'ProximalPhalanxOutlineCorrect', 'ProximalPhalanxTW', 'RefrigerationDevices', 'ScreenType', 'ShapeletSim', 'ShapesAll', 'SmallKitchenAppliances', 'SonyAIBORobotSurface1', 'SonyAIBORobotSurface2', 'StarLightCurves', 'Strawberry', 'SwedishLeaf', 'Symbols', 'SyntheticControl', 'ToeSegmentation1', 'ToeSegmentation2', 'Trace', 'TwoLeadECG', 'TwoPatterns', 'UWaveGestureLibraryAll', 'UWaveGestureLibraryX', 'UWaveGestureLibraryY', 'UWaveGestureLibraryZ', 'Wafer', 'Wine', 'WordSynonyms', 'Worms', 'WormsTwoClass', 'Yoga', 'ACSF1', 'AllGestureWiimoteX', 'AllGestureWiimoteY', 'AllGestureWiimoteZ', 'BME', 'Chinatown', 'Crop', 'DodgerLoopDay', 'DodgerLoopGame', 'DodgerLoopWeekend', 'EOGHorizontalSignal', 'EOGVerticalSignal', 'EthanolLevel', 'FreezerRegularTrain', 'FreezerSmallTrain', 'Fungi', 'GestureMidAirD1', 'GestureMidAirD2', 'GestureMidAirD3', 'GesturePebbleZ1', 'GesturePebbleZ2', 'GunPointAgeSpan', 'GunPointMaleVersusFemale', 'GunPointOldVersusYoung', 'HouseTwenty', 'InsectEPGRegularTrain', 'InsectEPGSmallTrain', 'MelbournePedestrian', 'MixedShapesRegularTrain', 'MixedShapesSmallTrain', 'PickupGestureWiimoteZ', 'PigAirwayPressure', 'PigArtPressure', 'PigCVP', 'PLAID', 'PowerCons', 'Rock', 'SemgHandGenderCh2', 'SemgHandMovementCh2', 'SemgHandSubjectCh2', 'ShakeGestureWiimoteZ', 'SmoothSubspace', 'UMD']

class SingleTSDataset(Dataset):
    def __init__(self,data,is_mask=True):
        super(SingleTSDataset, self).__init__()
        self.ts_id2label = np.array([[i for i in range(data.shape[0])], data[:, 0].tolist()]).T.astype(np.int)
        self.label = torch.from_numpy(data[:, 0])
        self.data = torch.from_numpy(data[:, 1:])
        self.len, self.seq_len = self.data.shape
    def __getitem__(self, item):
        return self.data[item].unsqueeze(0).type(torch.float64),self.label[item].type(torch.LongTensor)
    def __len__(self):
        return self.len

data_dir='../UCRArchive_2018'
out_dir= 'timenet_result/'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
timenet_result={}
timenet_result['Name']=[]
timenet_result['val_acc']=[]
for i in range(len(dataset_names)):
    datasetname= dataset_names[i]
    train_np= pandas.read_csv(os.path.join(data_dir,datasetname,datasetname+'_TRAIN.tsv'),
                              sep='	',
                              header=None).to_numpy()

    test_np= pandas.read_csv(os.path.join(data_dir,datasetname,datasetname+'_TEST.tsv'),
                             sep='	',
                             header=None).to_numpy()

    train_np[np.isnan(train_np)]= 0
    test_np[np.isnan(test_np)]=0

    nclass = len(set(train_np[:, 0]))
    seq_len = train_np.shape[1] - 1

    BATCH_STEP_SIZE= 128000//seq_len
    print(BATCH_STEP_SIZE)
    # label mapping to 0~n-1
    label2id = {}
    for i, key in enumerate(set(train_np[:, 0])):
        label2id[key] = i
    train_np[:, 0] = np.array([label2id[key] for key in train_np[:, 0]])
    test_np[:, 0] = np.array([label2id[key] for key in test_np[:, 0]])

    train_dataset = SingleTSDataset(train_np)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_STEP_SIZE, shuffle=True,
                                               num_workers=8, pin_memory=True)

    test_dataset = SingleTSDataset(test_np)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_STEP_SIZE, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    device = torch.device('cuda:{}'.format(0))
    net = model(nclass)
    net.double()
    # net.load_state_dict(torch.load('models/model-sae3-lowestlossperepoch-0.0111093344694-epoch-119.pt'))
    # net.eval()
    if CUDA:
        net.to(device)
    encoder_dic = torch.load('./models/model-sae3-lowestlossperepoch-0.017289805623443802-epoch-1009.pt')
    encoder_dic = encoder_dic['state_dict'] if 'state_dict' in encoder_dic else encoder_dic
    net_dict = net.state_dict()
    encoder_dic = {k: v for k, v in encoder_dic.items() if
                   (k in net_dict) and encoder_dic[k].shape == net_dict[k].shape}
    net_dict.update(encoder_dic)
    net.load_state_dict(net_dict, strict=False)

    optimizer = optim.Adam(net.parameters(), lr=0.01, betas=(0.5, 0.999), weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=50, verbose=True,
                                                     min_lr=0.0001)
    best_accuracy = 1e-9
    for epoch in range(EPOCHS):
        net.train()
        running_loss=0.0
        start=time.time()
        train_pred=[]
        train_gt=[]
        for i, (xs,labels) in tqdm (enumerate(train_loader,0)):
            optimizer.zero_grad()
            if CUDA:
                xs,labels= xs.to(device),labels.to(device)
            local_batch_x_reversed = xs.detach().cpu().numpy() \
                if CUDA else xs.detach().numpy()
            local_batch_x_reversed = numpy.flip(local_batch_x_reversed, axis=1).copy()
            local_batch_x_reversed = Variable(torch.from_numpy(local_batch_x_reversed).double(), requires_grad=False)
            if CUDA: local_batch_x_reversed = local_batch_x_reversed.to(device)
            predicted = net(xs.permute(0,2,1), local_batch_x_reversed.permute(0,2,1))[:,:,-1]
            loss = criterion(predicted, labels)
            # loss = criterion(predicted, labels.unsqueeze(-1).repeat(1, predicted.shape[-1]))
            loss.backward()
            optimizer.step()
            running_loss+=loss.item()
            train_pred.append(predicted.argmax(-1).detach().to('cpu').numpy())
            train_gt.append(labels.detach().to('cpu').numpy())
        train_pred= np.concatenate(train_pred,axis=0)
        train_gt= np.concatenate(train_gt,axis=0)

        acc = accuracy_score(y_true=train_gt, y_pred=train_pred)
        running_loss = running_loss / (i + 1)
        print('train: [%4d/ %5d] loss: %.6f, acc: %.6f,  time: %f s' %
              (epoch + 1, i, running_loss, acc, time.time() - start))

        net.eval()
        val_loss = 0.0
        pred = []
        gt = []
        for i, (xs, labels) in enumerate(test_loader, 0):
            if CUDA:
                xs, labels = xs.to(device), labels.to(device)
            local_batch_x_reversed = xs.detach().cpu().numpy() \
                if CUDA else xs.detach().numpy()
            local_batch_x_reversed = numpy.flip(local_batch_x_reversed, axis=1).copy()
            local_batch_x_reversed = Variable(torch.from_numpy(local_batch_x_reversed).double(), requires_grad=False)
            if CUDA: local_batch_x_reversed = local_batch_x_reversed.to(device)
            predicted = net(xs.permute(0,2,1), local_batch_x_reversed.permute(0,2,1))[:,:,-1]
            loss = criterion(predicted, labels)
            # loss = criterion(predicted, labels.unsqueeze(-1).repeat(1, predicted.shape[-1]))

            val_loss += loss.item()
            pred.append(predicted.argmax(-1).detach().to('cpu').numpy())
            gt.append(labels.detach().to('cpu').numpy())

        gt = np.concatenate(gt, axis=0)
        pred = np.concatenate(pred, axis=0)
        val_loss=val_loss/(i+1)
        acc_val= accuracy_score(y_true=gt,y_pred=pred) #sum(gt==pred)/gt.shape[0]
        print('test: [%4d/ %4d] loss: %.6f accuracy %.6f' %
              (epoch + 1, EPOCHS, val_loss, acc_val))

        scheduler.step(acc_val)
        if best_accuracy <acc_val:
            best_accuracy =acc_val
    timenet_result['Name'].append(datasetname)
    timenet_result['val_acc'].append(best_accuracy)

    df_result = pandas.DataFrame(timenet_result)
    df_result.to_csv('Result_result.csv')