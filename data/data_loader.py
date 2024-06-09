import os
import sys

import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler

import torch
from torch.utils.data import Dataset, DataLoader
# from sklearn.preprocessing import StandardScaler

from utils.tools import StandardScaler
from utils.timefeatures import time_features

import warnings
warnings.filterwarnings('ignore')


class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path, flag='train', size=None, 
                 features='S', data_path='ETTh1.csv', 
                 target='OT', scale=True, inverse=False, timeenc=0, freq='h', cols=None,args = None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24*4*4
            self.label_len = 24*4
            self.pred_len = 24*4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train':0, 'val':1, 'test':2}
        self.set_type = type_map[flag]
        
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        
        self.root_path = root_path
        self.data_path = data_path
        self.args = args
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12*30*24 - self.seq_len, 12*30*24+4*30*24 - self.seq_len]
        border2s = [12*30*24, 12*30*24+4*30*24, 12*30*24+8*30*24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        
        if self.features=='M' or self.features=='MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features=='S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
            
        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
    
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len 
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = np.concatenate([self.data_x[r_begin:r_begin+self.label_len], self.data_y[r_begin+self.label_len:r_end]], 0)
        else:
            seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark
    
    def __len__(self):
        return len(self.data_x) - self.seq_len- self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class Dataset_ETT_minute(Dataset):
    def __init__(self, root_path, flag='train', size=None, 
                 features='S', data_path='ETTm1.csv', 
                 target='OT', scale=True, inverse=False, timeenc=0, freq='t', cols=None,args = None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24*4*4
            self.label_len = 24*4
            self.pred_len = 24*4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train':0, 'val':1, 'test':2}
        self.set_type = type_map[flag]
        
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        
        self.root_path = root_path
        self.data_path = data_path
        self.args = args
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        # print(df_raw)
        # sys.exit()
        # 这里一脸蒙蔽
        border1s = [0, 12*30*24*4 - self.seq_len, 12*30*24*4+4*30*24*4 - self.seq_len]
        border2s = [12*30*24*4, 12*30*24*4+4*30*24*4, 12*30*24*4+8*30*24*4]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        # print(border1s)
        # print(border2s)
        
        if self.features=='M' or self.features=='MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features=='S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
            
        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)
        
        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

        # print(data_stamp)
        # sys.exit()
    
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = np.concatenate([self.data_x[r_begin:r_begin+self.label_len], self.data_y[r_begin+self.label_len:r_end]], 0)
        else:
            seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark
    
    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


# 训练数据的时候用到,制作train、vali、test数据集
class Dataset_Custom(Dataset):
    # 在exp_informer.py中传值进来，覆盖掉这些默认参数
    def __init__(self, root_path, flag='train', size=None, 
                 features='S', data_path='原始值-日均价.csv',
                 target='OT', scale=True, inverse=True, timeenc=0, freq='d', cols=None,args=None):
        # size [seq_len, label_len, pred_len]
        # 假如这个size没有定义，那么此刻将进行序列的维度的初始化
        if size == None:
            self.seq_len = 24*4*4
            self.label_len = 24*4
            self.pred_len = 24*4
        else:
            # 分别为输入encoder的序列长度、输入decoder中原属数据长度，预测长度
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train':0, 'val':1, 'test':2}
        self.set_type = type_map[flag]
        
        self.features = features
        self.target = target
        # 是否进行标准化，默认为true
        self.scale = scale
        self.inverse = inverse
        # 时间特征编码  args.embed, help='时间特征编码，选项：[timeF, fixed, learned]' ，默认为timeF
        #  这是注释   timeenc = 0 if args.embed!='timeF' else 1，默认为 1
        self.timeenc = timeenc
        # 时间特征编码的频率，就是进行特征工程的时候时间粒度选取多少，
        # '选项（options）:[s:secondly, t:minutely, h:hourly, d:daily, b:工作日（business days）, w:weekly, m:monthly], '
        self.freq = freq
        # cols实际上好像没传进来，为None
        self.cols=cols
        self.root_path = root_path
        self.data_path = data_path
        self.args = args
        self.__read_data__()

    # 读取数据并且完成特征工程
    def __read_data__(self):
        self.scaler = StandardScaler()
        # pandas读取数据,数据格式如下所示
        # print(self.data_path)
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        '''
        数据格式：date是必须存在的列
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        # cols = list(df_raw.columns); 
        # cols作为除了时间date以及zuizhongtarget之外的输入特征，以单特征预测来看，cols为空
        if self.cols:
            cols=self.cols.copy()
            cols.remove(self.target)
        else:
            # print(list(df_raw.columns))
            # 以单特征来看，cols为空
            cols = list(df_raw.columns); 
            # print('cols',cols);
            cols.remove('self.target'); print(cols);cols.remove('date')
        df_raw = df_raw[['date']+cols+[self.target]]
        # print(df_raw)

        # 数据集大小: 训练集：测试集：验证集   7:2:1，要大改
        num_train = int(len(df_raw)*self.args.train_proportion)
        num_test = int(len(df_raw)*self.args.test_proportion)
        num_vali = len(df_raw) - num_train - num_test

        # 只是拿来划分数据的时候用的:是为了筛选分割数据集
        # border1s与border2s要对应着看，如第一列，分别表示train的开始以及结束
        border1s = [0, num_train-self.seq_len, len(df_raw)-num_test-self.seq_len]
        border2s = [num_train, num_train+num_vali, len(df_raw)]
        # print(border1s)
        # print(border2s)
        # self.set_type为0或1或2，分别对应train、vali、test
        # border用于指出数据的边界
        # train：0；vali：num_train-self.seq_len；test：len(df_raw)-num_test-self.seq_len
        border1 = border1s[self.set_type]
        # train：num_train；vali： num_train+num_vali；test： len(df_raw)
        border2 = border2s[self.set_type]
        # print(border1)
        # print(border2)

        """
        M和MS都是多变量特征标志。
        S是单变量特征标志。
        """
        if self.features=='M' or self.features=='MS':
            # 取出非时间非target的所有列
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        # 单维特征target也即输入的特征
        elif self.features=='S':
            df_data = df_raw[[self.target]]

        # 是否进行标准化，默认为true
        # 注意，无论是训练还是vali还是test，标准化的时候都是以train的标准来进行的
        if self.scale:
            # 【0：num_train】，训练数据标准化，这边要自己改？？？
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
        # 获取数据的数量，分别在train的时候取出train的范围，vali的时候取出vali的范围，test的时候取出test的范围
        # 取出日期
        df_stamp = df_raw[['date']][border1:border2]
        # 转换时间类型
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        # 使用utils包里的timefeatures模块对时间操作
        # timeenc 时间特征编码  args.embed, help='时间特征编码，选项：[timeF, fixed, learned]' ，默认为timeF
        # timeenc 这是注释   timeenc = 0 if args.embed!='timeF' else 1，默认为 1

        # freq 时间特征编码的频率，就是进行特征工程的时候时间粒度选取多少，
        # freq '选项（options）:[s:secondly, t:minutely, h:hourly, d:daily, b:工作日（business days）, w:weekly, m:monthly], '
      
        data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)
        # print(data_stamp)

        # print(border1)
        # print(border2)
        # 获取数据的数量，分别在train的时候取出train的范围，vali的时候取出vali的范围，test的时候取出test的范围
        # 此处，data是经过标准化的 df_data
        self.data_x = data[border1:border2]
        # print(self.data_x)
        # 若要逆标准化，则这边 标签y直接用未经过标准化的值
        if self.inverse:
            # df_data就是target，且未经过标准化
            self.data_y = df_data.values[border1:border2]
        else:
            # 若无需逆标准化，则 标签y用经过标准化的
            self.data_y = data[border1:border2]
        # 时间信息
        self.data_stamp = data_stamp

        # print(self.data_stamp)
        # sys.exit()
    
    def __getitem__(self, index):
        # encoder的输入
        s_begin = index
        s_end = s_begin + self.seq_len
        # decoder的输入开始
        r_begin = s_end - self.label_len
        # decoder的输入结束
        r_end = r_begin + self.label_len + self.pred_len

        # 获取输入序列x
        seq_x = self.data_x[s_begin:s_end]
        # 假如会进行逆标准化
        if self.inverse:
            # self.data_x[r_begin:r_begin+self.label_len] 输入特征的 decoder输入中的真实值部分，此处已经经过标准化
            # self.data_y[r_begin+self.label_len:r_end]   target的 decoder输入中的要预测的部分，此处未经过标准化
            seq_y = np.concatenate([self.data_x[r_begin:r_begin+self.label_len], self.data_y[r_begin+self.label_len:r_end]], 0)
        else:
            # 不进行逆向标准化，经过标准化的y
            seq_y = self.data_y[r_begin:r_end]
        # 获取时间戳
        seq_x_mark = self.data_stamp[s_begin:s_end]
        # 获取时间戳
        seq_y_mark = self.data_stamp[r_begin:r_end]
        
        print("seq_x.shape=",seq_x.shape)
        print("seq_x=",seq_x)
        print("seq_x_mark.shape=",seq_x_mark.shape)
        print("seq_x_mark=",seq_x_mark)
        print("seq_y.shape=",seq_y.shape)
        print("seq_y=",seq_y)
        print("seq_y_mark.shape=",seq_y_mark.shape)
        print("seq_y_mark=",seq_y_mark)
        
        return seq_x, seq_y, seq_x_mark, seq_y_mark
    
    def __len__(self):
        return len(self.data_x) - self.seq_len- self.pred_len + 1

    # 将标准化后的数据 逆向转换回原来的样子
    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)











class Dataset_Pred(Dataset):
    def __init__(self, root_path, flag='pred', size=None, 
                 features='S', data_path='ETTh1.csv', 
                 target='OT', scale=True, inverse=True, timeenc=0, freq='15min', cols=None,args=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24*4*4
            self.label_len = 24*4
            self.pred_len = 24*4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['pred']
        
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols=cols
        self.root_path = root_path
        self.data_path = data_path
        self.pred_date = []
        self.args = args
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        # print("数据集划分情况：",len(df_raw),int(len(df_raw)*0.7),int(len(df_raw)*0.1),int(len(df_raw)*0.2),)
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        if self.cols:
            cols=self.cols.copy()
            cols.remove(self.target)
        else:
            cols = list(df_raw.columns); cols.remove(self.target); cols.remove('date')
        df_raw = df_raw[['date']+cols+[self.target]]

        # border1：数据集长度-序列长度(序列长度就是滑动窗口长度)
        border1 = len(df_raw)-self.seq_len
        border2 = len(df_raw)

        # 取特征数目
        if self.features=='M' or self.features=='MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features=='S':
            df_data = df_raw[[self.target]]
        # 标准化
        if self.scale:
            # ？？？我怀疑pred的时候不需要fit，直接transform
            self.scaler.fit(df_data.values)
            data = self.scaler.transform(df_data.values)
        # 如果不标准化，直接转化为数组
        else:
            data = df_data.values
        # 未知
        tmp_stamp = df_raw[['date']][border1:border2]
        # 转换类型
        tmp_stamp['date'] = pd.to_datetime(tmp_stamp.date)
        # ？？？生成时间：生成从date的最后一个时间点~预测长度+1（pred_len+1）【这里使用了延迟预测，即多预测了一个】
        pred_dates = pd.date_range(tmp_stamp.date.values[-1], periods=self.pred_len+1, freq=self.freq)
        # print("date_range产生的pred_dates结果：",pred_dates)
        # 时间的处理和转化
        df_stamp = pd.DataFrame(columns = ['date'])
        df_stamp.date = list(tmp_stamp.date.values) + list(pred_dates[1:])
        data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq[-1:])

        self.data_x = data[border1:border2]
        # 将标准化后的数据转回来
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
        # print("date处理的最后的data_stam：",self.data_stamp)

        self.pred_date = pred_dates.tolist()

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = self.data_x[r_begin:r_begin+self.label_len]
        else:
            seq_y = self.data_y[r_begin:r_begin+self.label_len]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark
    
    def __len__(self):
        return len(self.data_x) - self.seq_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)












# # 训练数据的时候用到,制作train、vali、test数据集
# class Dataset_Custom(Dataset):
#     # 在exp_informer.py中传值进来，覆盖掉这些默认参数
#     # start，end是后来加的，用于描述该数据从几号索引取到几号索引（0-1600，1600为最长电池长度，不会超过他）
#     def __init__(self, root_path, data_path, flag='train', size=None, 
#                  features='S',timeenc=0,args=None):
#         # size： [seq_len, label_len, pred_len]
#         # 最大范围，写死了
#         self.start=0
#         self.end=1600
#         # 分别为输入encoder的序列长度、输入decoder中原属数据长度，预测长度
#         self.seq_len = size[0]
#         self.label_len = size[1]
#         self.pred_len = size[2]
#         self.root_path = root_path
#         self.data_path = data_path
        
#         # 读取最小值
#         # print('aaa',self.root_path)
#         rawdataMin = pd.read_excel((os.path.join(self.root_path,
#                                           self.data_path)))
#         # 去除序号2
#         rawdataMin=rawdataMin.drop(rawdataMin.columns[2], axis=1)
#         # 选取训练数据
#         rawdataMin=rawdataMin.iloc[:,0:26]
#         rawdataMin=rawdataMin.values[self.start:self.end,:].astype(float)
        
#         # init
#         assert flag in ['train', 'test', 'val']
#         type_map = {'train':0, 'val':1, 'test':2}
#         # 0，表示train
#         self.set_type = type_map[flag]
        
#         # features为 S，表示单值预测
#         self.features = features
       
#         # 时间特征编码  args.embed, help='时间特征编码，选项：[timeF, fixed, learned]' ，默认为timeF
#         #  这是注释   timeenc = 0 if args.embed!='timeF' else 1，默认为 1
#         self.timeenc = timeenc
#         # 时间特征编码的频率，就是进行特征工程的时候时间粒度选取多少，
#         # '选项（options）:[s:secondly, t:minutely, h:hourly, d:daily, b:工作日（business days）, w:weekly, m:monthly], '
        
        
#         self.args = args
        
#         # 获取表格中所有列名（训练数据的）
#         self.dataTrain = pd.read_excel((os.path.join(self.root_path,
#                                           self.data_path)))
#         # 去除序号2
#         self.dataTrain=self.dataTrain.drop(self.dataTrain.columns[2], axis=1)
#         self.dataTrain=self.dataTrain.iloc[:,0:26]
#         # self.scalerData 与 self.scalerI   为了把归一化的步骤传到外面
#         self.encoderList, self.decoderList, self.scalerDataMin, self.lenListSum = self.__getsamples(rawdataMin)
        

#     def __getsamples(self, rawdataMin):                
#         lenList=[]
#         lenListSum=0
#         # QDFlatten=[]
#         MinFlatten=[]
#         # VarFlatten=[]
#         # 归一化及复原(26列)
#         for j,col in enumerate(self.dataTrain.columns):
#             # print('col',col)
#             # 每列的长度
#             len=(np.array(self.dataTrain.iloc[:,j].dropna())).shape[0]
#             lenList.append(len)
#             lenListSum=lenListSum+len
#             for i in range(len):
#                 # QDFlatten.append((rawdataQD[i,j]))
#                 MinFlatten.append(rawdataMin[i,j])
#                 # VarFlatten.append(rawdataVar[i,j])
#         # print('111',QDFlatten)
#         # print('sample_Sum', lenListSum - 26*(self.seq_len + self.pred_len - 1))
#         # print('lenList',lenList[0])
#         # 变成归一化接受的形式
#         # QDFlatten=np.array(QDFlatten).reshape(-1,1)
#         MinFlatten=np.array(MinFlatten).reshape(-1,1)
#         # VarFlatten=np.array(VarFlatten).reshape(-1,1)
        
#         # 归一化
#         # scalerDataQD = MinMaxScaler()
#         # scalerDataQD = scalerDataQD.fit(QDFlatten) 
#         # rawdataQD = scalerDataQD.transform(QDFlatten)
#         # rawdataQD = QDFlatten
#         # print('11',rawdataQD[793])      
#         scalerDataMin = MinMaxScaler()
#         scalerDataMin = scalerDataMin.fit(MinFlatten) 
#         # rawdataMin = scalerDataMin.transform(MinFlatten)
#         rawdataMin = MinFlatten
        
#         # scalerDataVar = MinMaxScaler()
#         # scalerDataVar = scalerDataVar.fit(VarFlatten) 
#         # rawdataVar = scalerDataVar.transform(VarFlatten)
     
            
#         # 还原成原来的格式  先
#         # QDNew=[]
#         MinNew=[]
#         # VarNew=[]
#         lenTemp=0
#         for i in range(26):
#             # print(i)
#             # QDNewTemp=[]
#             MinNewTemp=[]
#             # VarNewTemp=[]
          
#             for j in range(lenList[i]):
#                 # print(rawdataQD.shape)
#                 # QDNewTemp.append(rawdataQD[j+lenTemp][0])
#                 MinNewTemp.append(rawdataMin[j+lenTemp][0])
#                 # VarNewTemp.append(rawdataVar[j+lenTemp][0])
#                 # QDNewTemp.append(rawdataQD[j][0])
#                 # MinNewTemp.append(rawdataMin[j][0])
#                 # VarNewTemp.append(rawdataVar[j][0])
                
#             lenTemp=lenTemp+lenList[i]
#             # QDNewTemp=np.array(QDNewTemp)
#             MinNewTemp=np.array(MinNewTemp)
#             # VarNewTemp=np.array(VarNewTemp)
#             # QDNew.append(QDNewTemp)   
#             MinNew.append(MinNewTemp)   
#             # VarNew.append(VarNewTemp)   
        
#         # QDNew=np.array(QDNew,dtype = object)    
#         MinNew=np.array(MinNew,dtype = object)    
#         # VarNew=np.array(VarNew,dtype = object) 
#         # print("MinNew",MinNew.size())
#         # print("MinNew",MinNew)
            

#         # XAll为43列合并后的，XPre为每一列的
#         XAll=[]
#         YAll=[]
#         # 最后跑列
#         for j,col in enumerate(self.dataTrain.columns):
#             # XPre为每种充电方案的，XAll为42种充电方案汇总的
#             # 减10是为了算出 sample_num
#             sample_num=lenList[j] - self.seq_len - self.pred_len + 1
#             # (sample-num,3,1,10)，第二个参数 1 表示channel为 1
#             # X是encoder的输入
#             XPre = torch.zeros((sample_num, self.seq_len,1))
#             # Y是decoder的输入
#             YPre = torch.zeros((sample_num, self.label_len + self.pred_len,1))
#             # YPre = torch.zeros((sample_num, 1))
           
#             # IArray=np.array([I[j,0]]*self.sample_num).reshape(-1,1)
#             # print('IArray',IArray)
#             # 200条原始数据的话，sample_num为190
#             for i in range(sample_num):
#                 # encoder的输入开始
#                 s_begin = i
#                 # encoder的输入结束
#                 s_end = s_begin + self.seq_len
#                 # decoder的输入开始
#                 r_begin = s_end - self.label_len
#                 # decoder的输入结束
#                 r_end = r_begin + self.label_len + self.pred_len

#                 # 获取输入序列x
#                 # seq_x = self.data_x[s_begin:s_end]
#                 startX = i
#                 # end从10到200
#                 endX = i + self.seq_len
#                 # result=zip(QDNew[j][start:end], MinNew[j][start:end], VarNew[j][start:end])
#                 # j是第几列，start和end是起始以及终止的行数
#                 result_x=np.vstack(( MinNew[j][s_begin:s_end].reshape((self.seq_len,1))))
#                 result_y=np.vstack(( MinNew[j][r_begin:r_end].reshape((self.label_len+self.pred_len,1))))
                
#                 # 第一个参数 1 表示channel为 1 
#                 # XPre的shape为(sample_num,1,1,seq_len)
#                 XPre[i, :, :] = torch.from_numpy(np.array(list(result_x)))
#                 # YPre的shape为(sampe_num,1,1,label_len+pred_len)
#                 YPre[i, :, :] = torch.from_numpy(np.array(list(result_y)))
                
#             XAll.append(XPre)
#             YAll.append(YPre)
            
#         # 一字排开，变成竖的
        
     
#         # XAll.shape为（sample_num,seq_len,1）
#         XAll=torch.cat(XAll,dim=0).reshape(-1,self.seq_len,1).double()
#         YAll=torch.cat(YAll,dim=0).reshape(-1,self.label_len+self.pred_len,1).double()
#         # YAll=torch.stack(YAll).reshape(-1,1,1).double()
#         # YAll=torch.cat(YAll,dim=0).reshape(-1,1,1).double()
#         # print('YAll.shape=',YAll.shape)
#         # print('XAll',XAll.shape)
#         # print("XAll853",XAll[853])
#         # print("XAll854",XAll[854])
#         # print("XAll855",XAll[855])
#         # print('YAll',YAll.shape)
#         # print("YAll853",YAll[853])
#         # print("YAll854",YAll[854])
#         # print("YAll855",YAll[855])       
#         return (XAll, YAll, scalerDataMin, lenListSum) 
    
#     # def __getitem__(self, index):
#     #     # encoder的输入
#     #     s_begin = index
#     #     s_end = s_begin + self.seq_len
#     #     # decoder的输入开始
#     #     r_begin = s_end - self.label_len
#     #     # decoder的输入结束
#     #     r_end = r_begin + self.label_len + self.pred_len

#     #     # 获取输入序列x
#     #     seq_x = self.data_x[s_begin:s_end]
#     #     # 假如会进行逆标准化
#     #     if self.inverse:
#     #         # self.data_x[r_begin:r_begin+self.label_len] 输入特征的 decoder输入中的真实值部分，此处已经经过标准化
#     #         # self.data_y[r_begin+self.label_len:r_end]   target的 decoder输入中的要预测的部分，此处未经过标准化
#     #         seq_y = np.concatenate([self.data_x[r_begin:r_begin+self.label_len], self.data_y[r_begin+self.label_len:r_end]], 0)
#     #     else:
#     #         # 不进行逆向标准化，经过标准化的y
#     #         seq_y = self.data_y[r_begin:r_end]
#     #     # 获取带有掩码的输入序列x
#     #     seq_x_mark = self.data_stamp[s_begin:s_end]
#     #     # 获取带有掩码的输入序列x
#     #     seq_y_mark = self.data_stamp[r_begin:r_end]
#     #     return seq_x, seq_y, seq_x_mark, seq_y_mark
    
#     def __len__(self):
#         return self.lenListSum - 26*(self.seq_len + self.pred_len - 1)
    
#      # 外部使用【idx】来获取，idx的max值即上面的__len__
#     def __getitem__(self, idx):
#         seq_x=self.encoderList[idx, :, :]
#         seq_y=self.decoderList[idx, :, :]
#         # ？？？此处的seq_x_mark和seq_y_mark是假的，暂时不用，等要加上全局时间embedding的时候再加上
#         # 获取带有掩码的输入序列x
#         seq_x_mark = torch.zeros(1)
#         # 获取带有掩码的输入序列x
#         seq_y_mark = torch.zeros(1)
#         # print('idx',idx)
#         if(idx>-1 and idx==0):
#             print(idx)
#             print('seq_x',seq_x)
#             print('seq_y',seq_y)
        
        
#         return seq_x, seq_y,seq_x_mark, seq_y_mark