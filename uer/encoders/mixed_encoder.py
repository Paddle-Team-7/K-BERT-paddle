# -*- encoding:utf-8 -*-
#import torch
#import torch.nn as nn
import paddle
import paddle.nn as nn


class RcnnEncoder(nn.Layer):
    def __init__(self, args):
        super(RcnnEncoder, self).__init__()

        self.emb_size = args.emb_size
        self.hidden_size= args.hidden_size
        self.kernel_size = args.kernel_size
        self.layers_num = args.layers_num

        self.rnn = nn.LSTM(input_size=args.emb_size,
                           hidden_size=args.hidden_size,
                           num_layers=args.layers_num,
                           dropout=args.dropout,
                           time_major=False)

        self.drop = nn.Dropout(args.dropout)

        self.conv_1 = nn.Conv2D(1, args.hidden_size, (args.kernel_size, args.emb_size))
        self.conv = nn.LayerList([nn.Conv2D(args.hidden_size, args.hidden_size, (args.kernel_size, 1)) \
            for _ in range(args.layers_num-1)])

    def forward(self, emb, seg):
        batch_size, seq_len, _ = emb.size()

        hidden = self.init_hidden(batch_size, emb.device)
        output, hidden = self.rnn(emb, hidden) 
        output = self.drop(output)

        
        padding = paddle.zeros([batch_size, self.kernel_size-1, self.emb_size])
        hidden = paddle.concat([padding, output], dim=1).unsqueeze(1) # batch_size, 1, seq_length+width-1, emb_size
        hidden = self.conv_1(hidden)
        padding =  paddle.zeros([batch_size, self.hidden_size, self.kernel_size-1, 1])
        hidden = paddle.concat([padding, hidden], axis=2)
        for i, conv_i in enumerate(self.conv):
            hidden = conv_i(hidden)
            hidden = paddle.concat([padding, hidden], axis=2)
        hidden = hidden[:,:,self.kernel_size-1:,:]
        output = hidden.transpose(1,2).contiguous().view(batch_size, seq_len, self.hidden_size)

        return output

    def init_hidden(self, batch_size, device):
        return (paddle.zeros([self.layers_num, batch_size, self.hidden_size]),
                paddle.zeros([self.layers_num, batch_size, self.hidden_size]))


class CrnnEncoder(nn.Layer):
    def __init__(self, args):
        super(CrnnEncoder, self).__init__()

        self.emb_size = args.emb_size
        self.hidden_size= args.hidden_size
        self.kernel_size = args.kernel_size
        self.layers_num = args.layers_num

        self.conv_1 = nn.Conv2D(1, args.hidden_size, (args.kernel_size, args.emb_size))
        self.conv = nn.LayerList([nn.Conv2D(args.hidden_size, args.hidden_size, (args.kernel_size, 1)) \
            for _ in range(args.layers_num-1)])


        self.rnn = nn.LSTM(input_size=args.emb_size,
                           hidden_size=args.hidden_size,
                           num_layers=args.layers_num,
                           dropout=args.dropout,
                           time_major=False)
        self.drop = nn.Dropout(args.dropout)

    def forward(self, emb, seg):
        batch_size, seq_len, _ = emb.size()
        padding = paddle.zeros([batch_size, self.kernel_size-1, self.emb_size])
        emb = paddle.concat([padding, emb], axis=1).unsqueeze(1) # batch_size, 1, seq_length+width-1, emb_size
        hidden = self.conv_1(emb)
        padding =  paddle.zeros([batch_size, self.hidden_size, self.kernel_size-1, 1])
        hidden = paddle.concat([padding, hidden], axis=2)
        for i, conv_i in enumerate(self.conv):
            hidden = conv_i(hidden)
            hidden = paddle.concat([padding, hidden], axis=2)
        hidden = hidden[:,:,self.kernel_size-1:,:]
        output = hidden.transpose(1,2).contiguous().view(batch_size, seq_len, self.hidden_size)

        hidden = self.init_hidden(batch_size, emb.device)
        output, hidden = self.rnn(output, hidden) 
        output = self.drop(output)

        return output

    def init_hidden(self, batch_size, device):
        return (paddle.zeros([self.layers_num, batch_size, self.hidden_size]),
                paddle.zeros([self.layers_num, batch_size, self.hidden_size]))

