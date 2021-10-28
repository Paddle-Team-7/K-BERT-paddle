#import torch
#import torch.nn as nn
import paddle
import paddle.nn as nn


class LstmSubencoder(nn.Layer):
    def __init__(self, args, vocab_size):
        super(LstmSubencoder, self).__init__()
        self.hidden_size= args.emb_size
        self.layers_num = args.sub_layers_num

        self.embedding_layer = nn.Embedding(vocab_size, args.emb_size)
        self.rnn = nn.LSTM(input_size=args.emb_size,
                           hidden_size=self.hidden_size,
                           num_layers=self.layers_num,
                           dropout=args.dropout,
                           time_major=False)

    def forward(self, ids):
        batch_size, _ = ids.size() # batch_size, max_length
        hidden = (paddle.zeros([self.layers_num, batch_size, self.hidden_size]),
                  paddle.zeros([self.layers_num, batch_size, self.hidden_size]))
        emb = self.embedding_layer(ids)
        output, hidden = self.rnn(emb, hidden)
        output = output.mean(1)
        return output
