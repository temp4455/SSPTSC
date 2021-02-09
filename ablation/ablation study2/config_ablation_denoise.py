from easydict import EasyDict
import time
import argparse
class Parameter():
    def __init__(self):
        self.batch_size= 256
        self.seq_len=100
        self.in_channels=1
        self.learning_rate=0.01
        # self.data_dir='/home/***/PycharmProjects/***/UCRArchive_2018/StarLightCurves'
        self.data_dir='/media/***/externdisk/python_workspace/***/UCRArchive_2018/StarLightCurves'
        # self.use_cuda= True
        self.start_cuda_id= 0
        self.gpu_nums=1
        self.num_workers=8
        self.max_epoch=100
        self.out_dir='./out_'+time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        self.show_interval = 1
        self.val_interval =5
        self.nclass=None
        self.attention_window= 20
        self.global_index_num = 100
    def to_dict(self):
        return self.__dict__
    def update(self,dic):
        self.__dict__.update(dic)



def cmd_parameter_pre_training():
    param = Parameter()
    parser= argparse.ArgumentParser(description='Train the mask encoder!')

    parser.add_argument('--batch_size',default=512, type=int,help='the batch size of the input data')
    parser.add_argument('--data_dir',default='/media/***/externdisk/python_workspace/***/UCRArchive_2018/StarLightCurves',type=str, help='the path for input data')
    # parser.add_argument('--use_cuda',default=1,type=int, help='1 define True, 0 define False, default 1')
    parser.add_argument('--max_epoch',default=1000,type=int, help='max epoch')
    parser.add_argument('--out_dir',default='./out_'+time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),type=str)
    parser.add_argument('--show_interval',default=1,type=int)
    parser.add_argument('--val_interval',default=5,type=int)
    args=parser.parse_args()
    param.batch_size= args.batch_size
    param.data_dir= args.data_dir
    param.use_cuda= args.use_cuda
    param.max_epoch= args.max_epoch
    param.out_dir= args.out_dir
    param.show_interval= args.show_interval
    param.val_interval= args.val_interval

    return param


def cmd_parameter_fine_tuning():
    param = Parameter()
    parser = argparse.ArgumentParser(description='Fine tuning the classification!')

    parser.add_argument('--train_batch_size', default=128, type=int, help='the batch size of the input data in training')
    parser.add_argument('--val_batch_size', default=32, type=int, help='the batch size of the input data in val')

    parser.add_argument('--data_dir',
                        default='/media/***/externdisk/python_workspace/***/UCRArchive_2018/StarLightCurves', type=str,
                        help='the path for input data')
    parser.add_argument('--model_path',default='',type=str,help='the path of the pre train model ')
    # parser.add_argument('--use_cuda', default=1, type=int, help='1 define True, 0 define False, default 1')
    parser.add_argument('--start_cuda_id', default=0, type=int, help='the start id of gpu')
    parser.add_argument('--gpu_nums', default=1, type=int, help='the gpu nums that using, zeros define as not use gpu')

    parser.add_argument('--max_epoch', default=1000, type=int, help='max epoch')
    parser.add_argument('--out_dir', default='./fine_tuning_classify_' + time.strftime("%Y%m%d%H%M%S", time.localtime()), type=str)
    parser.add_argument('--show_interval', default=1, type=int)
    parser.add_argument('--val_interval', default=5, type=int)
    parser.add_argument('--seq_len', default=128, type=int)
    parser.add_argument('--attention_window', default=20, type=int)
    parser.add_argument('--global_index_num', default=100, type=int)
    parser.add_argument('--beta', default=10, type=int,help='The proportion of noise added')
    parser.add_argument('--lr', default=0.01, type=float, help='The learning rate')
    args = parser.parse_args()
    param.train_batch_size = args.train_batch_size
    param.val_batch_size = args.val_batch_size

    param.model_path= args.model_path
    param.data_dir = args.data_dir
    # param.use_cuda = args.use_cuda
    param.start_cuda_id=args.start_cuda_id
    param.gpu_nums= args.gpu_nums
    param.max_epoch = args.max_epoch
    param.out_dir = args.out_dir
    param.show_interval = args.show_interval
    param.val_interval = args.val_interval
    param.attention_window= args.attention_window
    param.global_index_num =args.global_index_num
    param.beta = args.beta
    param.learning_rate= args.lr
    # param.seq_len=args.seq_len
    return param


if __name__=='__main__':
    print(param.to_dict())

