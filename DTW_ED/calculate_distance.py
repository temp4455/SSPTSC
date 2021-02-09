from cydtw import dtw as cydtw
# from cdtw import pydtw
import numpy as np
import os
from glob import glob
from tqdm import tqdm
import pandas as pd
import cupy as cp

import multiprocessing
def dtw(x1,x2):
    assert len(x1.shape)==len(x2.shape), "the shape of input should be the same"
    if len(x1.shape)==1:
        return cydtw(x1,x2)
    x1_l,x2_l= x1.shape[0], x2.shape[0]
    res= np.zeros((x1_l,x2_l))
    for i in range(x1_l):
        for j in range(x2_l):
            res[i][j]= cydtw(x1[i].reshape((-1, 1)),x2[j].reshape((-1, 1)))
    return res

class multi_dtw():
    def __init__(self,x1,x2,process_num=10):
        self.x1=x1
        self.x2=x2
        self.process_num= process_num
    def child_process(self,start):
        # print(start)
        x1,x2= self.x1[start:start+1],self.x2
        return dtw(x1,x2)
    def get_result(self):
        samples=[i for i in range(self.x1.shape[0])]
        pool= multiprocessing.Pool(self.process_num)
        res=pool.map_async(self.child_process,samples).get()
        res=np.concatenate(res,axis=0)
        return res

def ed_single(x1,x2):
    # assert x1.shape==x2.shape, "the shape of input should be the same"
    return np.sqrt(np.sum(np.square(x1-x2),axis=1))
def ed_all(all_data):
    all_data=cp.array(all_data)
    res_ed = cp.zeros((all_data.shape[0], all_data.shape[0]))
    for i in range(all_data.shape[0]):
        res_ed[i] = ed_single(all_data[i], all_data)
    return cp.asnumpy(res_ed)#.asnumpy()
def dtw_batch(x1,x2):
    inf=10000000
    assert len(x1.shape)==len(x2.shape), "the shape of input should be the same"
    x1_num,x1_l = x1.shape
    x2_num,x2_l = x2.shape
    res= np.zeros((x1_num,x2_num))
    for i in range(x1_num):
        temp_m = np.ones((x2_num, x1_l + 1, x2_l+ 1)) * inf
        temp_m[:,0,0]=0
        for j in range(1,x1_l+1):
            for k in range(1,x2_l+1):
                temp_m[:,j,k]=np.min(np.array([temp_m[:,j-1,k],temp_m[:,j,k-1],temp_m[:,j-1,k-1]]),axis=0)+abs(x1[i,j-1]-x2[:,k-1])
        res[i,:]= (temp_m[:,-1,-1])
    return res


def calculate_all(out_dir="./similar_result",data_dir=''):
    assert len(data_dir)>0 , 'the length of data_dir should not be 0'
    dataset_names= os.listdir(data_dir)
    # ready calculate
    ready_cal_names = list(set([os.path.basename(path).split('_')[0] for path in glob('./similar_result/*.npy')]))
    ready_cal_names+=['Missing_value_and_variable_length_datasets_adjusted']
    dataset_names=[v for v in dataset_names if v not in ready_cal_names]

    for dataset_name in tqdm(dataset_names):
        print('start calculate the dataset {}'.format(dataset_name))
        train_file_name= dataset_name+'_TRAIN.tsv'
        test_file_name = dataset_name + '_TEST.tsv'
        train=pd.read_csv(os.path.join(data_dir,dataset_name,train_file_name),
                          sep='\t', header=None, index_col=None).to_numpy()[:,1:].astype(np.float)

        test=pd.read_csv(os.path.join(data_dir,dataset_name,test_file_name),
                          sep='\t', header=None, index_col=None).to_numpy()[:,1:].astype(np.float)

        all_data= np.concatenate([train,test],axis=0)
        # res_dtw= dtw(all_data,all_data)
        tool_dtw=multi_dtw(all_data,all_data,20)
        res_dtw=tool_dtw.get_result()
        res_ed= ed_all(all_data)

        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        np.save(os.path.join(out_dir,dataset_name+'_DTW.npy'), res_dtw)
        np.save(os.path.join(out_dir,dataset_name+'_ED.npy'), res_ed)


if __name__=='__main__':
    # s_y1 = np.array([[1, 2, 3, 4]])
    # s_y2 = np.array([[1, 2, 2, 3]])
    # res = dtw_batch(s_y1, s_y2)
    calculate_all(out_dir="./similar_result",
                  data_dir='../UCRArchive_2018'
                  )

