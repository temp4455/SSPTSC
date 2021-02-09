import numpy as np
import torch
import os
import torch.nn as nn
from tensorboardX import SummaryWriter
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from torchvision import datasets,transforms
from torch.autograd import Variable
import json
from pyprobar import bar, probar
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow, imsave
import time
import itertools
import pickle
# import imageio
import random
from sklearn.metrics import average_precision_score,f1_score
import pandas as pd
from glob import glob
from config_dtw import cmd_parameter_fine_tuning,Parameter
# from model_mask_Batchnorm import TSEncoder,TSClassifcation
from model_dtw_longformer import TSEncoderDTWED as LongformerTSEncoderDTWED
from model_dtw import TSEncoderDTWED
from torch.utils.data import DataLoader,Dataset
from sklearn.metrics import accuracy_score,f1_score
from sklearn.model_selection import train_test_split
import logging


class SingleTSDatasetDWT(Dataset):
    def __init__(self,data):
        super(SingleTSDatasetDWT, self).__init__()
        self.ts_id2label = np.array([[i for i in range(data.shape[0])], (data[:, 0] - 1).tolist()]).T.astype(np.int)
        self.ts_id = torch.tensor([i for i in range(data.shape[0])])
        self.data = torch.from_numpy(data[:, 1:])
        self.label = torch.from_numpy(data[:, 0])
        self.len, self.seq_len = self.data.shape

    def __getitem__(self, item):
        return  self.data[item].unsqueeze(0).type(torch.float32),self.ts_id[item].type(torch.long)
        # return out_data.unsqueeze(0).type(torch.float32),self.label[item].type(torch.LongTensor)-1
    def __len__(self):
        return self.len

def save_checkpoint(states, is_best, output_dir, loss,filename='checkpoint.pth.tar'):
    torch.save(states,os.path.join(output_dir,filename))
    if is_best and 'state_dict' in states:
        torch.save(states['state_dict'],os.path.join(output_dir,'model_best_{}.pth.tar'.format(loss)))

if __name__=='__main__':
    # for data
    # param=Parameter()
    param=cmd_parameter_fine_tuning()
    param.learning_rate=0.0001
    # config = json.load(open(os.path.join(os.path.dirname(param.model_path),'config.json'),'r'))
    # param.update(config)
    writer = SummaryWriter(param.out_dir)
    path=param.data_dir
    files = glob(os.path.join(path, '*_TRAIN.tsv'))
    train_np = pd.read_csv(files[0], sep='	', header=None).to_numpy()#pd.concat([pd.read_csv(file, sep='	', header=None) for file in files], axis=0, ignore_index=True)
    test_np= pd.read_csv(glob(os.path.join(path,'*_TEST.tsv'))[0], sep='	', header=None).to_numpy()
    data_all= np.concatenate([train_np,test_np],axis=0)

    if (train_np.shape[1] - 1) > 200:
        param.seq_len = ((train_np.shape[1] - 1) // param.attention_window) * param.attention_window
        data_all=data_all[:, :1 + param.seq_len]
    param.seq_len = train_np.shape[1] - 1

    param.nclass= len(set(data_all[:,0]))
    # train_np_split,val_np_split= train_test_split(data_all,test_size=0.3,random_state=42)

    similar_matrix= np.load(param.similar_path)
    train_dataset= SingleTSDatasetDWT(data_all)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=param.train_batch_size, shuffle=True, num_workers=param.num_workers, pin_memory=True,drop_last=True)

    test_dataset= SingleTSDatasetDWT(data_all)
    test_loader= DataLoader(test_dataset, batch_size= param.val_batch_size, shuffle= True,num_workers=param.num_workers, pin_memory=True,drop_last=True)


    if not os.path.exists(param.out_dir):
        os.makedirs(param.out_dir)
    json.dump(param.to_dict(),open(os.path.join(param.out_dir,'config.json'),'w'),indent=4)


    device= torch.device('cuda:{}'.format(param.start_cuda_id)) \
        if param.gpu_nums>0 and torch.cuda.is_available()  else torch.device('cpu')
    # model= TSClassifcation(param.nclass,param.global_index_num,param.attention_window)
    if param.seq_len >200:
        print('long former ts encoder')
        model= LongformerTSEncoderDTWED(param.global_index_num,param.attention_window)
    else:
        print('ts encoder')
        model= TSEncoderDTWED(param.global_index_num,param.attention_window)
    if len(param.model_path) :
        encoder_dic = torch.load(param.model_path)
        encoder_dic = encoder_dic['state_dict'] if 'state_dict' in encoder_dic else encoder_dic
        model_dict= model.state_dict()
        encoder_dic= {k:v for k,v in encoder_dic.items() if (k in model_dict) and encoder_dic[k].shape==model_dict[k].shape}
        model_dict.update(encoder_dic)
        model.load_state_dict(model_dict, strict=False)

    if param.gpu_nums>1 and torch.cuda.device_count()>=param.gpu_nums :
        model=nn.DataParallel(model,device_ids=[i for i in range(param.start_cuda_id,param.start_cuda_id+param.gpu_nums)])
    model.to(device)

    optimizer= optim.Adam(model.parameters(),lr=param.learning_rate,betas=(0.5,0.999))
    # criterion = nn.MSELoss()
    criterion = nn.CrossEntropyLoss()
    scheduler= optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',patience=10,verbose=True,min_lr=1e-6)

    book = np.ones((param.train_batch_size, param.train_batch_size))
    index1 = np.arange(param.train_batch_size)
    book[index1, index1] = 0

    book1 = np.ones((param.val_batch_size, param.val_batch_size))
    index1 = np.arange(param.val_batch_size)
    book1[index1, index1] = 0

    # training
    best_loss = 1e9
    for epoch in range(param.max_epoch):
        model.train()
        running_loss=0.0
        start=time.time()
        for i,(xs,ts_id) in tqdm(enumerate(train_loader,0)):
            optimizer.zero_grad()
            xs=xs.to(device)
            ts_id=ts_id.numpy()
            temp_similar_matrix= similar_matrix[ts_id,:][:,ts_id]
            temp_similar_matrix = temp_similar_matrix[book == 1].reshape(param.train_batch_size, param.train_batch_size - 1)
            label = torch.tensor(temp_similar_matrix.argmin(axis=1), dtype=torch.long).to(device)


            output,_= model(xs)
            loss = criterion(output,label)
            # print(loss)
            loss.backward()
            optimizer.step()
            # print(loss)
            running_loss+=loss.item()
        running_loss=running_loss / (i+1)
        print('train: [%4d/ %5d] loss: %.6f,   time: %f s' %
              (epoch + 1, i, running_loss, time.time()-start))

        if epoch and epoch % param.val_interval == 0:
            model.eval()
            val_loss=0.0
            for i ,(xs,ts_id) in enumerate(test_loader,0):
                xs= xs.to(device)
                ts_id = ts_id.numpy()
                temp_similar_matrix = similar_matrix[ts_id, :][:, ts_id]
                temp_similar_matrix=temp_similar_matrix[book1==1].reshape(param.val_batch_size, param.val_batch_size - 1)
                label= torch.tensor(temp_similar_matrix.argmin(axis=1),dtype=torch.long).to(device)
                output,outputx = model(xs )
                loss= criterion(output,label)
                # if loss.detach().cpu().item() > 100:
                #     print(label,output)
                #     print(loss)
                val_loss += loss.item()

            val_loss=val_loss/(i+1)

            print('test: [%4d/ %4d] loss: %.6f ' %
                  (epoch + 1, param.max_epoch, val_loss))

            writer.add_scalars(os.path.join(param.out_dir,'loss'),{'train loss':running_loss,'val loss':val_loss},epoch)

            scheduler.step(val_loss)
            if best_loss > val_loss:
                best_loss = val_loss
                model_save_name='checkpoint_epoch{:03d}_accuracy{:0.4f}.pth.tar'.format(epoch,val_loss)
                save_checkpoint({
                    'epoch':epoch+1,
                    'state_dict':model.module.state_dict() if isinstance(model,nn.DataParallel) else model.state_dict(),
                    'perf':val_loss,
                    'optimizer': optimizer.state_dict(),
                },True,param.out_dir,val_loss,model_save_name)

    writer.close()


