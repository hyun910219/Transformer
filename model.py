import math
import numpy as np
import torch
import torch.nn as nn

input_data = [[0,1,2], [1,2,3]]
output_data = [[1,2,3],[2,3,4]]

batch_size = 2
max_word_length = 3
d_model = 5
d_ff = 2048
encoder_voca_size = 5
decoder_voca_size = 5
number_of_encoder_layers = 6
number_of_decoder_layers = 6
e_h = 5
d_h1 = 5
d_h2 = 5

class embedding(nn.Module):
    def __init__(self, voca_size, dimension):
        super(embedding, self).__init__()
        self.embed = nn.Embedding(voca_size, dimension)

def positional_encoding(x):
    dimension = x.shape[-1]
    position = x.shape[-2]
    batches = x.shape[-3]
    pe = np.array([[[pos / 10000**(2*i) for i in range(dimension)] for pos in range(position)] for batch in range(batches)])
    pe[:, :, 0::2] = np.sin(pe[:, :, 0::2])
    pe[:, :, 1::2] = np.cos(pe[:, :, 1::2])
    return torch.as_tensor(torch.from_numpy(pe), dtype=torch.float32) + x

def scaled_dot_product_attention(q, k, v, mask=False):
    matmul_qk = torch.matmul(q, k.transpose(1,2))
    scale = matmul_qk / math.sqrt(d_model)
    if mask == True:
        for i in range(scale.shape[1]):
            for j in range(scale.shape[2]):
                if i >= j:
                    pass
                else:
                    scale[:, i, j] = -math.inf
        mask_opt = scale
    else:
        mask_opt = scale
    softmax = torch.softmax(mask_opt, dim=2)
    matmul_softv = torch.matmul(softmax, v)
    return matmul_softv

def add_norm(x1, x2):
    return x1 + x2

class multi_head_attention(nn.Module):
    def __init__(self, h):
        super(multi_head_attention, self).__init__()
        self.h = h
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask):
        Q = self.q_linear(q).view(batch_size, max_word_length, -1, self.h)
        K = self.k_linear(k).view(batch_size, max_word_length, -1, self.h)
        V = self.v_linear(v).view(batch_size, max_word_length, -1, self.h)
        head = []
        for i in range(self.h):
            head.append(scaled_dot_product_attention(Q[:, :, :, i], K[:, :, :, i], V[:, :, :, i], mask))
        concat = torch.cat(head, -1)
        linear = self.out_linear(concat)
        return linear

class feed_forward(nn.Module):
    def __init__(self):
        super(feed_forward, self).__init__()
        self.feed_linear1 = nn.Linear(d_model, d_ff)
        self.feed_linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.feed_linear2(torch.nn.ReLU()(self.feed_linear1(x)))

class encoder_layer(nn.Module):
    def __init__(self, mha_h):
        super(encoder_layer, self).__init__()
        self.mha_h = mha_h
        self.multi_head_attention = multi_head_attention(self.mha_h)
        self.feed_forward = feed_forward()

    def forward(self, x):
        sub_layer_x = self.multi_head_attention(q=x, k=x, v=x, mask=False)
        x = add_norm(x, sub_layer_x)
        sub_layer_x = self.feed_forward(x)
        x = add_norm(x, sub_layer_x)
        return x

class decoder_layer(nn.Module):
    def __init__(self, mha_h, mmha_h):
        super(decoder_layer, self).__init__()
        self.mha_h = mha_h
        self.mmha_h = mmha_h
        self.masked_multi_head_attention = multi_head_attention(self.mmha_h)
        self.multi_head_attention = multi_head_attention(self.mha_h)
        self.feed_forward = feed_forward()

    def forward(self, x, encoder_input):
        sub_layer_x = self.masked_multi_head_attention(q=x, k=x, v=x, mask=True)
        x = add_norm(x, sub_layer_x)
        sub_layer_x = self.multi_head_attention(q=x, k=encoder_input, v=encoder_input, mask=False)
        x = add_norm(x, sub_layer_x)
        sub_layer_x = self.feed_forward(x)
        x = add_norm(x, sub_layer_x)
        return x

class transformer(nn.Module):
    def __init__(self):
        super(transformer, self).__init__()
        self.input_embedding = embedding(encoder_voca_size, d_model)
        self.output_embedding = embedding(decoder_voca_size, d_model)
        self.encode_layers = nn.ModuleList([encoder_layer(mha_h=e_h) for i in range(number_of_encoder_layers)])
        self.decode_layers = nn.ModuleList([decoder_layer(mha_h=d_h1, mmha_h=d_h2) for i in range(number_of_decoder_layers)])

    def forward(self, inputs, outputs):
        inputs = self.input_embedding.embed(torch.tensor(inputs))
        inputs = positional_encoding((inputs))
        for i in range(number_of_encoder_layers):
            inputs = self.encode_layers[i](inputs)

        outputs = self.output_embedding.embed(torch.tensor(outputs))
        outputs = positional_encoding((outputs))
        # for i in range(number_of_decoder_layers):
        #     outputs = self.decode_layers[i](outputs, inputs)
        return outputs

model = transformer()
model(input_data, output_data)
