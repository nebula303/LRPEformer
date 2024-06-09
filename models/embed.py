# LSTMEmbed.py
import torch
import torch.nn as nn
import torch.nn.functional as F

import math

# 1.数据embedding，将输入维度映射成 d_model维（原文中是7维映射到512维度）
class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__>='1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model, 
                                    kernel_size=3, padding=padding, padding_mode='circular')
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight,mode='fan_in',nonlinearity='leaky_relu')

    def forward(self, x):
        # print("wordOrigin",x.shape)
        # print("wordOrigin",x)
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1,2)
        # print("word32",x.shape)
        # print("word32",x)
        return x

# 2.1 位置编码（0-99），即encoder输入的长度，decoder同理
class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # print("pe1",self.pe.shape)
        # print("pe1",self.pe)
        # print("pe1.1",self.pe[:, :x.size(1)].shape)
        # print("pe1.1",self.pe[:, :x.size(1)])
        
        return self.pe[:, :x.size(1)]
    

class CustomLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(CustomLSTM, self).__init__()
        
        # LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        # Fully connected layer to map LSTM output to the desired output size
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # Forward pass through LSTM
        out, _ = self.lstm(x)
        
        # Map LSTM output to the desired output size
        out = self.fc(out)
        
        return out




class  DataEmbedding(nn.Module):
    # 输入【64，100，1】，输出【64，100，16】（【batch，seq，d_model】）
    # c_in可能是encoder的输入特征数，也可能是decoder的输入特征数，具体看外面传,单特征就是 1，c_in就是lstm中的 input_size
    def __init__(self, c_in, d_model, lstm_hidden_size, lstm_num_layers, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding, self).__init__()

        # output_size先设置得和d_model一样，后期看要不要调整
        self.customLSTM = CustomLSTM(c_in, lstm_hidden_size, lstm_num_layers, c_in)
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        # self.position_embedding = PositionalEmbedding(d_model=d_model)
        
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        # 三个embedding直接相加得到模型输入，分别为数据embedding，位置embedding以及时间embedding
        # x = self.value_embedding(x) + self.position_embedding(x) + self.temporal_embedding(x_mark)
        # 暂时先删除时间embedding？？？
        
        # x = self.value_embedding(x) + self.position_embedding(x) + self.position_embeddingAll(x_mark)
        # 去除 1-100的位置embedding
        # 类似resnet
        x = self.customLSTM(x) + x
        # positionEmbedding在之后加
        x = self.value_embedding(x) 
        # print('x.size()',x.size())
        
        return self.dropout(x)