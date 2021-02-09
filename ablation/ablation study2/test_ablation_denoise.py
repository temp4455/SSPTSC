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
from config_ablation_denoise import cmd_parameter_fine_tuning,Parameter
# from model_mask_Batchnorm import TSEncoder,TSClassifcation
from model_ablation_denoise_lf import TSEncoder as LongformerTSEncoder
from model_ablation_denoise import TSEncoder
from torch.utils.data import DataLoader,Dataset
from sklearn.metrics import accuracy_score,f1_score
from sklearn.model_selection import train_test_split
import logging

def awgn(x, snr=10, seed=7):
    np.random.seed(seed)
    # torch.random.seed()  
    snr = 10 ** (snr / 10.0)
    xpower = torch.sum(x ** 2) / len(x)
    npower = xpower / snr
    noise = torch.tensor(np.random.randn(len(x))* np.sqrt(npower.numpy()))
    # noise = torch.random.ra(len(x)) * torch.sqrt(npower)
    return x + noise

class SingleTSDatasetDenoise(Dataset):
    def __init__(self,data,beta=10,is_denoise=False):
        super(SingleTSDatasetDenoise, self).__init__()
        self.ts_id2label = np.array([[i for i in range(data.shape[0])], (data[:, 0] - 1).tolist()]).T.astype(np.int)
        self.data = torch.from_numpy(data[:, 1:])
        self.label = torch.from_numpy(data[:, 0])
        self.len, self.seq_len = self.data.shape
        self.beta = beta
    def __getitem__(self, item):
        out_data=self.data[item].clone()
        mask = torch.ones(self.seq_len)
        nums = self.seq_len // 100 * self.beta
        for i in range(nums):
            start = random.randint(0, self.seq_len - 1)
            # single_len = random.randint(self.beta, self.beta+10)
            single_len =1
            out_data[start:start+single_len] = awgn(out_data[start:start+single_len])
            mask[start:start + single_len] = 0
        return  out_data.unsqueeze(0).type(torch.float32),mask.unsqueeze(0).type(torch.float32),self.data[item].type(torch.float32)
        # return out_data.unsqueeze(0).type(torch.float32),self.label[item].type(torch.LongTensor)-1
    def __len__(self):
        return self.len

def save_checkpoint(states, is_best, output_dir, loss,filename='checkpoint.pth.tar'):
    torch.save(states,os.path.join(output_dir,filename))
    if is_best and 'state_dict' in states:
        torch.save(states['state_dict'],os.path.join(output_dir,'model_best_{}.pth.tar'.format(loss)))

if __name__=='__main__':
    param=cmd_parameter_fine_tuning()
    writer = SummaryWriter(param.out_dir)
    path=param.data_dir
    train_np = pd.read_csv(glob(os.path.join(path, '*_TRAIN.tsv'))[0], sep='	', header=None).to_numpy()#pd.concat([pd.read_csv(file, sep='	', header=None) for file in files], axis=0, ignore_index=True)
    test_np= pd.read_csv(glob(os.path.join(path,'*_TEST.tsv'))[0], sep='	', header=None).to_numpy()
    data_all= np.concatenate([train_np,test_np],axis=0)

    # deal nan
    data_all[np.isnan(data_all)] = 0

    if (train_np.shape[1]-1)>200:
        param.seq_len=(( train_np.shape[1]-1)//param.attention_window)*param.attention_window
        data_all=data_all[:,:1+param.seq_len]
    param.seq_len= train_np.shape[1]-1

    param.nclass = len(set(data_all[:, 0]))
    # pre process
    data_all[:, 1:] = (data_all[:, 1:] - data_all[:, 1:].min()) / (data_all[:, 1:].max() - data_all[:, 1:].min()) * (
                param.nclass - 1)


    train_np_split,val_np_split= train_test_split(data_all,test_size=0.3,random_state=42)

    train_dataset= SingleTSDatasetDenoise(train_np_split,param.beta)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=param.train_batch_size, shuffle=True, num_workers=param.num_workers, pin_memory=True)

    test_dataset= SingleTSDatasetDenoise(val_np_split,param.beta)
    test_loader= DataLoader(test_dataset, batch_size= param.val_batch_size, shuffle= False,num_workers=param.num_workers, pin_memory=True)


    if not os.path.exists(param.out_dir):
        os.makedirs(param.out_dir)
    json.dump(param.to_dict(),open(os.path.join(param.out_dir,'config.json'),'w'),indent=4)


    device= torch.device('cuda:{}'.format(param.start_cuda_id)) \
        if param.gpu_nums>0 and torch.cuda.is_available()  else torch.device('cpu')
    # model= TSClassifcation(param.nclass,param.global_index_num,param.attention_window)
    if param.seq_len >200:
        model= LongformerTSEncoder(param.global_index_num,param.attention_window)
    else:
        model =TSEncoder(param.global_index_num,param.attention_window)
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

    optimizer= optim.Adam(model.parameters(),lr=param.learning_rate,betas=(0.5,0.999),weight_decay=0.01)
    criterion = nn.MSELoss()
    scheduler= optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=0.5,patience=50,verbose=True,min_lr=1e-6)

    # training
    best_loss = 1e9
    for epoch in range(param.max_epoch):
        model.train()
        running_loss=0.0
        start=time.time()
        for i,(xs,mask,label) in tqdm(enumerate(train_loader,0)):
            optimizer.zero_grad()
            xs,label=xs.to(device),label.to(device)
            # return :[batch,seq_len]
            output = model(xs)
            loss = criterion(output,label)
            # loss= criterion(out,xs.unsqueeze(-1).repeat(1,out1.shape[-1]))+criterion(out2,labels)
            # loss= criterion(out2,labels)
            loss.backward()
            optimizer.step()
            running_loss+=loss.item()
        running_loss=running_loss / (i+1)
        print('train: [%4d/ %5d] loss: %.6f,   time: %f s' %
              (epoch + 1, i, running_loss, time.time()-start))

        if epoch and epoch % param.val_interval == 0:
            model.eval()
            val_loss=0.0
            for i ,(xs,masks,label) in enumerate(test_loader,0):
                xs, label = xs.to(device), label.to(device)
                output = model(xs )
                loss= criterion(output,label)
                # loss = criterion(out1,labels.unsqueeze(-1).repeat(1,out1.shape[-1]))+criterion(out2,labels)
                # loss= criterion(out2,labels)
                # loss = loss1+loss2
                val_loss += loss.item()

            val_loss=val_loss/(i+1)

            print('test: [%4d/ %4d] loss: %.6f ' %
                  (epoch + 1, param.max_epoch, val_loss))
            # writer.add_scalar()
            # writer.add_scalars(os.path.join(param.out_dir,'accuracy'),{'train accuracy':acc,'val accuracy': acc_val},epoch)
            writer.add_scalars(os.path.join(param.out_dir,'loss'),{'train loss':running_loss,'val loss':val_loss},epoch)

            scheduler.step(val_loss)
            if optimizer.state_dict()['param_groups'][0]['lr'] < 1e-5:
                break
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


