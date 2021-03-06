import numpy as np
import torch
import os
import torch.nn as nn
from tensorboardX import SummaryWriter
import torch.optim as optim
from tqdm import tqdm
import json
import time
# import imageio
import pandas as pd
from glob import glob
from original_fine.config_original import cmd_parameter_original
# from model_mask_Batchnorm import TSEncoder,TSClassifcation
from original_fine.model_original import TSClassifcation
from original_fine.model_original_lf import TSClassifcation as LongformerTSClassifcation
from torch.utils.data import DataLoader,Dataset
from sklearn.metrics import accuracy_score


class SingleTSDataset(Dataset):
    def __init__(self,data,is_mask=True):
        super(SingleTSDataset, self).__init__()
        self.ts_id2label = np.array([[i for i in range(data.shape[0])], (data[:, 0] - 1).tolist()]).T.astype(np.int)
        self.data = torch.from_numpy(data[:, 1:])
        self.label = torch.from_numpy(data[:, 0])
        self.len, self.seq_len = self.data.shape
    def __getitem__(self, item):
        return self.data[item].unsqueeze(0).type(torch.float32),self.label[item].type(torch.LongTensor)-1
    def __len__(self):
        return self.len

def save_checkpoint(states, is_best, output_dir, loss,filename='checkpoint.pth.tar'):
    torch.save(states,os.path.join(output_dir,filename))
    if is_best and 'state_dict' in states:
        torch.save(states['state_dict'],os.path.join(output_dir,'model_best_{}.pth.tar'.format(loss)))

if __name__=='__main__':
    # for data
    # param=Parameter()
    param=cmd_parameter_original()
    writer = SummaryWriter(param.out_dir)


    dataset_names=  os.listdir(param.data_dir)
    for dataset_name in dataset_names:
        print('train the dataset {}'.format(dataset_name))
        t_path= os.path.join(param.data_dir,dataset_name)
        train_np = pd.read_csv(glob(os.path.join(t_path, '*_TRAIN.tsv'))[0], sep='	', header=None).to_numpy() #pd.concat([pd.read_csv(file, sep='	', header=None) for file in files], axis=0, ignore_index=True)
        param.nclass= len(set(train_np[:,0]))
        test_np= pd.read_csv(glob(os.path.join(t_path,'*_TEST.tsv'))[0], sep='	', header=None).to_numpy()

        if (train_np.shape[1] - 1) > 200:
            param.seq_len = ((train_np.shape[1] - 1) // 2) * 2
            train_np=train_np[:, :1 + param.seq_len]
            test_np=test_np[:, :1 + param.seq_len]
        param.seq_len = train_np.shape[1] - 1

        # label mapping to 0~n-1
        label2id = {}
        for i, key in enumerate(set(train_np[:, 0])):
            label2id[key] = i
        train_np[:, 0] = np.array([label2id[key] for key in train_np[:, 0]])
        test_np[:, 0] = np.array([label2id[key] for key in test_np[:, 0]])

        train_dataset= SingleTSDataset(train_np)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=param.train_batch_size, shuffle=True, num_workers=param.num_workers, pin_memory=True)

        test_dataset= SingleTSDataset(test_np)
        test_loader= DataLoader(test_dataset, batch_size= param.val_batch_size, shuffle= False,num_workers=param.num_workers, pin_memory=True)


        if not os.path.exists(os.path.join(param.out_dir,dataset_name)):
            os.makedirs(os.path.join(param.out_dir,dataset_name))
        json.dump(param.to_dict(),open(os.path.join(os.path.join(param.out_dir,dataset_name),'config.json'),'w'),indent=4)


        device= torch.device('cuda:{}'.format(param.start_cuda_id)) \
            if param.gpu_nums>0 and torch.cuda.is_available()  else torch.device('cpu')

        if param.seq_len > 200:
            model = LongformerTSClassifcation(param.nclass, param.global_index_num, param.attention_window)
        else:
            model = TSClassifcation(param.nclass, param.global_index_num, param.attention_window)

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
        criterion= nn.CrossEntropyLoss()

        scheduler= optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='max',patience=20,verbose=True,min_lr=1e-7)

        # training
        best_accuracy = 1e-9
        for epoch in range(param.max_epoch):
            model.train()
            running_loss=0.0
            start=time.time()
            train_pred = []
            train_gt = []
            for i,(xs,labels) in tqdm(enumerate(train_loader,0)):
                optimizer.zero_grad()
                xs,labels=xs.to(device),labels.to(device)
                out1,out2= model(xs)
                loss= criterion(out1,labels.unsqueeze(-1).repeat(1,out1.shape[-1]))+criterion(out2,labels)
                # loss= criterion(out2,labels)
                loss.backward()
                optimizer.step()
                running_loss+=loss.item()
                train_pred.append(out2.argmax(dim=-1).detach().to('cpu').numpy())
                train_gt.append(labels.detach().to('cpu').numpy())
                # if i and i % param.show_interval==0:
            train_pred = np.concatenate(train_pred, axis=0)
            train_gt = np.concatenate(train_gt, axis=0)

            acc = accuracy_score(y_true=train_gt,y_pred=train_pred)
            running_loss=running_loss / (i+1)
            # print(book)
            # print(train_dataset.ts_id2label[:,1])
            # print((book==train_dataset.ts_id2label[:,1]).sum())
            # print(book.shape)
            print('train: [%4d/ %5d] loss: %.6f, acc: %.6f,  time: %f s' %
                  (epoch + 1, i,running_loss ,acc,time.time()-start))
            # start=time.time()

            if epoch and epoch % param.val_interval==0:
                model.eval()
                val_loss=0.0
                pred = []
                gt = []
                for i ,(xs,labels) in enumerate(test_loader,0):
                    xs, labels = xs.to(device), labels.to(device)
                    out1,out2 = model(xs )
                    # print(out1.shape,out2.shape)
                    # print(labels.unsqueeze(-1).repeat(1,out1.shape[-1]).shape,labels.shape)
                    loss = criterion(out1,labels.unsqueeze(-1).repeat(1,out1.shape[-1]))+criterion(out2,labels)
                    # loss= criterion(out2,labels)
                    # loss = loss1+loss2
                    val_loss += loss.item()
                    pred.append(out2.argmax(dim=-1).detach().to('cpu').numpy())
                    gt.append(labels.detach().to('cpu').numpy())
                gt = np.concatenate(gt, axis=0)
                pred = np.concatenate(pred, axis=0)
                val_loss=val_loss/(i+1)
                acc_val= accuracy_score(y_true=gt,y_pred=pred) #sum(gt==pred)/gt.shape[0]
                print('test: [%4d/ %4d] loss: %.6f accuracy %.6f' %
                      (epoch + 1, param.max_epoch, val_loss, acc_val))
                writer.add_scalars(os.path.join(param.out_dir,dataset_name,'accuracy'),{'train accuracy':acc,'val accuracy': acc_val},epoch)
                writer.add_scalars(os.path.join(param.out_dir,dataset_name,'loss'),{'train loss':running_loss,'val loss':val_loss},epoch)

                scheduler.step(acc_val)
                if optimizer.state_dict()['param_groups'][0]['lr'] <1e-6:
                    break
                if best_accuracy < acc_val:
                    best_accuracy = acc_val
                    model_save_name='checkpoint_epoch{:03d}_accuracy{:0.4f}.pth.tar'.format(epoch,acc_val)
                    save_checkpoint({
                        'epoch':epoch+1,
                        'state_dict':model.module.state_dict() if isinstance(model,nn.DataParallel) else model.state_dict(),
                        'perf':val_loss,
                        'optimizer': optimizer.state_dict(),
                    },True,os.path.join(param.out_dir,dataset_name),acc_val,model_save_name)

        writer.close()


