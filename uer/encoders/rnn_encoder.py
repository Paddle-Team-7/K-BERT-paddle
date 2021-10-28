# -*- encoding:utf-8 -*-
import paddle
import paddle.nn as nn


class LstmEncoder(nn.Layer):
    def __init__(self, args):
        super(LstmEncoder, self).__init__()

        self.bidirectional = args.bidirectional
        if self.bidirectional:
            assert args.hidden_size % 2 == 0 
            self.hidden_size= args.hidden_size // 2
        else:
            self.hidden_size= args.hidden_size
        
        self.layers_num = args.layers_num

        if self.bidirectional==True:
            xd ='bidirectional'
        else:
            xd = 'forward'

        self.rnn = nn.LSTM(input_size=args.emb_size,
                           hidden_size=self.hidden_size,
                           num_layers=args.layers_num,
                           dropout=args.dropout,
                           time_major=False,
                           direction=xd)

        self.drop = nn.Dropout(args.dropout)

    def forward(self, emb, seg):
        hidden = self.init_hidden(emb.size(0), emb.device)
        output, hidden = self.rnn(emb, hidden) 
        output = self.drop(output) 
        return output

    def init_hidden(self, batch_size, device):
        if self.bidirectional:
            return (paddle.zeros([self.layers_num*2, batch_size, self.hidden_size]),
                    paddle.zeros([self.layers_num*2, batch_size, self.hidden_size]))
        else:
            return (paddle.zeros([self.layers_num, batch_size, self.hidden_size]),
                    paddle.zeros([self.layers_num, batch_size, self.hidden_size]))


class GruEncoder(nn.Layer):
    def __init__(self, args):
        super(GruEncoder, self).__init__()

        self.bidirectional = args.bidirectional
        if self.bidirectional:
            assert args.hidden_size % 2 == 0 
            self.hidden_size= args.hidden_size // 2
        else:
            self.hidden_size= args.hidden_size

        self.layers_num = args.layers_num

        if self.bidirectional==True:
            xd ='bidirectional'
        else:
            xd = 'forward'

        self.rnn = nn.GRU(input_size=args.emb_size,
                           hidden_size=self.hidden_size,
                           num_layers=args.layers_num,
                           dropout=args.dropout,
                           time_major=False,
                           direction=xd)

        self.drop = nn.Dropout(args.dropout)

    def forward(self, emb, seg):
        hidden = self.init_hidden(emb.size(0), emb.device)
        output, hidden = self.rnn(emb, hidden) 
        output = self.drop(output) 
        return output

    def init_hidden(self, batch_size, device):
        if self.bidirectional:
            return paddle.zeros([self.layers_num*2, batch_size, self.hidden_size])
        else:
            return paddle.zeros([self.layers_num, batch_size, self.hidden_size])
