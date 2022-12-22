#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#from IPython.display import display, HTML
#display(HTML("<style>.container { width:100% !important; }</style>"))


# **GPU 사용 설정**

# In[ ]:


#%env CUDA_VISIBLE_DEVICES=0,1,2,3
#%env CUDA_VISIBLE_DEVICES=1
#%env CUDA_VISIBLE_DEVICES=""
get_ipython().run_line_magic('env', 'CUBLAS_WORKSPACE_CONFIG=:4096:8')


# **Dataset 지정**

# In[ ]:


dataset_name = "final"  # final dataset 
#dataset_name = "original"  # originial dataset in the paper
#dataset_name = "ko"        # helper dataset
#dataset_name = "1cycle"     # 1cycle dataset

if dataset_name == "original":
    EMBEDDING_FILE = "data/embeddings.pkl"
else:
    EMBEDDING_FILE = f"data/embeddings_{dataset_name}.pkl"


# **Random seed**

# In[ ]:


import numpy as np
import torch
import random

random_seed = 42

random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
torch.use_deterministic_algorithms(True, warn_only=True)
torch.backends.cudnn.deterministic = True


# **Model 정의**

# In[ ]:


#from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import *
import numpy as np
from torch.nn.parameter import Parameter
import pickle
from typing import Optional

class layer_normalization(nn.Module):

    def __init__(self, features, epsilon=1e-8): # original implemenation epsilon value
    #def __init__(self, features, epsilon=1e-5):
    #def __init__(self, features, epsilon=1e-12):
        '''Applies layer normalization.
        Args:
          epsilon: A floating number. A very small number for preventing ZeroDivision Error.
        '''
        super(layer_normalization, self).__init__()
        self.layernorm = nn.LayerNorm(features, eps=epsilon)

    def forward(self, x):
        return self.layernorm(x)
    
class multihead_attention(nn.Module):

    def __init__(self, num_units, num_heads=8, dropout_rate=0, causality=False):
        '''Applies multihead attention.
        Args:
            num_units: A scalar. Attention size.
            dropout_rate: A floating point number.
            causality: Boolean. If true, units that reference the future are masked.
            num_heads: An int. Number of heads.
        '''
        super(multihead_attention, self).__init__()
        self.num_units = num_units
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.causality = causality
        self.Q_proj = nn.Sequential(nn.Linear(self.num_units, self.num_units), nn.ReLU())
        self.K_proj = nn.Sequential(nn.Linear(self.num_units, self.num_units), nn.ReLU())
        self.V_proj = nn.Sequential(nn.Linear(self.num_units, self.num_units), nn.ReLU())
        
        # tensorflow compatible initializer
        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                torch.nn.init.zeros_(m.bias)
        self.Q_proj.apply(init_weights)
        self.K_proj.apply(init_weights)
        self.V_proj.apply(init_weights)

        self.output_dropout = nn.Dropout(p=self.dropout_rate)

        self.normalization = layer_normalization(self.num_units)

    def forward(self, queries, keys, values):
        # keys, values: same shape of [N, T_k, C_k]
        # queries: A 3d Variable with shape of [N, T_q, C_q]
        queries_skip = queries
        
        # Normalize (new position)
        #queries = self.normalization(queries)  # (N, T_q, C)
        #keys = self.normalization(keys)  # (N, T_q, C)
        #values = self.normalization(values)  # (N, T_q, C)

        # Linear projections
        Q = self.Q_proj(queries)  # (N, T_q, C)
        #print('Q shape:', Q.shape)
        K = self.K_proj(keys)  # (N, T_q, C)
        V = self.V_proj(values)  # (N, T_q, C)

        # Split and concat
        Q_ = torch.cat(torch.chunk(Q, self.num_heads, dim=2), dim=0)  # (h*N, T_q, C/h)
        K_ = torch.cat(torch.chunk(K, self.num_heads, dim=2), dim=0)  # (h*N, T_q, C/h)
        V_ = torch.cat(torch.chunk(V, self.num_heads, dim=2), dim=0)  # (h*N, T_q, C/h)

        # Multiplication
        outputs = torch.bmm(Q_, K_.permute(0, 2, 1))  # (h*N, T_q, T_k)

        # Scale
        outputs = outputs / (K_.size()[-1] ** 0.5)

        # Key Masking
        key_masks = torch.sign(torch.abs(torch.sum(keys, dim=-1)))  # (N, T_k)
        key_masks = key_masks.repeat(self.num_heads, 1)  # (h*N, T_k)
        key_masks = torch.unsqueeze(key_masks, 1).repeat(1, queries.size()[1], 1)  # (h*N, T_q, T_k)
        #key_masks.requires_grad = False
        key_masks = key_masks.detach()

        padding = Variable(torch.ones(*outputs.size()).to(queries) * (-2 ** 32 + 1))
        condition = key_masks.eq(0.).float()
        outputs = padding * condition + outputs * (1. - condition)

        # Causality = Future blinding
        if self.causality:
            diag_vals = torch.ones(*outputs[0, :, :].size()).to(queries)  # (T_q, T_k)
            tril = torch.tril(diag_vals, diagonal=0)  # (T_q, T_k)
            masks = Variable(torch.unsqueeze(tril, 0).repeat(outputs.size()[0], 1, 1))  # (h*N, T_q, T_k)

            padding = Variable(torch.ones(*masks.size()).to(queries) * (-2 ** 32 + 1))
            condition = masks.eq(0.).float()
            outputs = padding * condition + outputs * (1. - condition)

        # Activation
        outputs = F.softmax(outputs, dim=-1)  # (h*N, T_q, T_k)

        # Query Masking
        query_masks = torch.sign(torch.abs(torch.sum(queries, dim=-1)))  # (N, T_q)
        query_masks = query_masks.repeat(self.num_heads, 1)  # (h*N, T_q)
        query_masks = torch.unsqueeze(query_masks, 2).repeat(1, 1, keys.size()[1])  # (h*N, T_q, T_k)
        #query_masks.requires_grad = False
        query_masks = query_masks.detach()
        outputs = outputs * query_masks

        # Dropouts
        outputs = self.output_dropout(outputs)  # (h*N, T_q, T_k)

        # Weighted sum
        outputs = torch.bmm(outputs, V_)  # (h*N, T_q, C/h)

        # Restore shape
        outputs = torch.cat(torch.chunk(outputs, self.num_heads, dim=0), dim=2)  # (N, T_q, C)
        
        # Normalize (experimental position)
        #outputs = self.normalization(outputs)  # (N, T_q, C)

        # Residual connection
        #outputs += queries
        outputs += queries_skip

        # Normalize (old position)
        #outputs = self.normalization(outputs)  # (N, T_q, C)

        return outputs

class feedforward(nn.Module):

    def __init__(self, in_channels, num_units=[2048, 512]):
        '''Point-wise feed forward net.
        Args:
          in_channels: a number of channels of inputs
          num_units: A list of two integers.
        '''
        super(feedforward, self).__init__()
        self.in_channels = in_channels
        self.num_units = num_units
        
        # nn.Linear is faster than nn.Conv1d
        self.conv = False
        if self.conv:
            params = {'in_channels': self.in_channels, 'out_channels': self.num_units[0],
                      'kernel_size': 1, 'stride': 1, 'bias': True}
            self.conv1 = nn.Sequential(nn.Conv1d(**params), nn.ReLU())
            params = {'in_channels': self.num_units[0], 'out_channels': self.num_units[1],
                      'kernel_size': 1, 'stride': 1, 'bias': True}
            self.conv2 = nn.Conv1d(**params)
        else:
            self.conv1 = nn.Sequential(nn.Linear(self.in_channels, self.num_units[0]), nn.ReLU())
            self.conv2 = nn.Linear(self.num_units[0], self.num_units[1])
            
        # tensorflow compatible initializer
        #def init_weights(m):
        #    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
        #        torch.nn.init.xavier_uniform_(m.weight)
        #        torch.nn.init.zeros_(m.bias)
        #self.conv1.apply(init_weights)
        #self.conv2.apply(init_weights)
        
        self.normalization = layer_normalization(self.in_channels)

    def forward(self, inputs):
        # Layer normalization (new position)
        inputs_skip = inputs
        
        if self.conv:
            inputs = inputs.permute(0, 2, 1)
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        if self.conv:
            outputs = outputs.permute(0, 2, 1)
            
        # Layer normalization (experimental position)
        outputs = self.normalization(outputs)

        # Residual connection
        outputs += inputs_skip

        # Layer normalization (old position)
        #if self.conv:
        #    outputs = self.normalization(outputs.permute(0, 2, 1))
        #else:
        #    outputs = self.normalization(outputs)

        return outputs

class ScriptWriter_cpre(nn.Module):
    
    def __init__(
        self,
        eta=0.7,
        max_sentence_len = 50,
        max_num_utterance = 11,
        embedding_file = EMBEDDING_FILE,
    ):
        super().__init__()
        self.scalar_loss = False
        self.max_num_utterance = max_num_utterance
        self.negative_samples = 1
        self.max_sentence_len = max_sentence_len
        self.hidden_units = 200 #word embedding size
        #self.total_words = 43514
        #self.total_words = 11883
        self.dropout_rate = 0.1
        self.num_heads = 1
        self.num_blocks = 3 
        self.eta = eta
        self.gamma = nn.Parameter(torch.tensor(0.5), requires_grad=True)
        word_emb = pickle.load(open(embedding_file, 'rb'), encoding="bytes")
        word_emb = torch.FloatTensor(word_emb)
        self.embedding = nn.Embedding.from_pretrained(word_emb, freeze=True)
        
        for i in range(self.num_blocks):
            self.__setattr__('self_multihead_attention_%d' % i, multihead_attention(
                     num_units=self.hidden_units,
                     num_heads=self.num_heads,
                     dropout_rate=self.dropout_rate,
                     causality=False))
            self.__setattr__('self_feedforward_%d' % i, feedforward(
                     self.hidden_units,
                     [self.hidden_units, self.hidden_units]))
            
        for i in range(self.num_blocks+1):
            self.__setattr__('ru_multihead_attention_%d' % i, multihead_attention(
                     num_units=self.hidden_units,
                     num_heads=self.num_heads,
                     dropout_rate=self.dropout_rate,
                     causality=False))
            self.__setattr__('ru_feedforward_%d' % i, feedforward(
                     self.hidden_units,
                     [self.hidden_units, self.hidden_units]))
            self.__setattr__('ur_multihead_attention_%d' % i, multihead_attention(
                     num_units=self.hidden_units,
                     num_heads=self.num_heads,
                     dropout_rate=self.dropout_rate,
                     causality=False))
            self.__setattr__('ur_feedforward_%d' % i, feedforward(
                     self.hidden_units,
                     [self.hidden_units, self.hidden_units]))
            self.__setattr__('nu_multihead_attention_%d' % i, multihead_attention(
                     num_units=self.hidden_units,
                     num_heads=self.num_heads,
                     dropout_rate=self.dropout_rate,
                     causality=False))
            self.__setattr__('nu_feedforward_%d' % i, feedforward(
                     self.hidden_units,
                     [self.hidden_units, self.hidden_units]))
            self.__setattr__('un_multihead_attention_%d' % i, multihead_attention(
                     num_units=self.hidden_units,
                     num_heads=self.num_heads,
                     dropout_rate=self.dropout_rate,
                     causality=False))
            self.__setattr__('un_feedforward_%d' % i, feedforward(
                     self.hidden_units,
                     [self.hidden_units, self.hidden_units]))
            self.__setattr__('nr_multihead_attention_%d' % i, multihead_attention(
                     num_units=self.hidden_units,
                     num_heads=self.num_heads,
                     dropout_rate=self.dropout_rate,
                     causality=False))
            self.__setattr__('nr_feedforward_%d' % i, feedforward(
                     self.hidden_units,
                     [self.hidden_units, self.hidden_units]))
            self.__setattr__('rn_multihead_attention_%d' % i, multihead_attention(
                     num_units=self.hidden_units,
                     num_heads=self.num_heads,
                     dropout_rate=self.dropout_rate,
                     causality=False))
            self.__setattr__('rn_feedforward_%d' % i, feedforward(
                     self.hidden_units,
                     [self.hidden_units, self.hidden_units]))
                                       
                                       
        self.n_dense = nn.Linear(self.hidden_units, self.hidden_units)
        torch.nn.init.xavier_uniform_(self.n_dense.weight)
        torch.nn.init.zeros_(self.n_dense.bias)
        self.lastu_dense = nn.Linear(self.max_sentence_len, 1) 
        torch.nn.init.xavier_uniform_(self.lastu_dense.weight)
        torch.nn.init.zeros_(self.lastu_dense.bias)
        self.lastur_dense = nn.Linear(self.max_sentence_len, 1)
        torch.nn.init.xavier_uniform_(self.lastur_dense.weight)
        torch.nn.init.zeros_(self.lastur_dense.bias)
        
        depth = self.max_num_utterance # 11
        height = self.max_sentence_len # 50
        width = self.max_sentence_len # 50
        padding = ((depth%3 + 1)//2, (height%3 + 1)//2, (width%3 + 1)//2,)
        conv3d_1_layer = nn.Conv3d((self.num_blocks+1)*2, 32, 3, padding='same') 
        nn.init.uniform_(conv3d_1_layer.weight, -0.01, 0.01) 
        nn.init.zeros_(conv3d_1_layer.bias)
        self.conv3d_1 = torch.nn.Sequential(conv3d_1_layer, torch.nn.ELU())
        self.maxpool3d_1 = torch.nn.MaxPool3d(3, padding=padding)
        
        depth = (self.max_num_utterance+2)//3 # 11
        height = (self.max_sentence_len+2)//3 # 50
        width = (self.max_sentence_len+2)//3 # 50
        padding = ((depth%3 + 1)//2, (height%3 + 1)//2, (width%3 + 1)//2,)
        conv3d_2_layer = nn.Conv3d(32, 32, 3, padding='same') 
        nn.init.uniform_(conv3d_2_layer.weight, -0.01, 0.01)
        nn.init.zeros_(conv3d_2_layer.bias)
        self.conv3d_2 = torch.nn.Sequential(conv3d_2_layer, torch.nn.ELU())
        self.maxpool3d_2 = torch.nn.MaxPool3d(3, padding=padding)
        mur_flatten_size = ((depth+2)//3)*((height+2)//3)*((width+2)//3)*32
        #print('mur_flatten_size =', mur_flatten_size)
        
        height = self.max_sentence_len # 50
        width = self.max_sentence_len # 50
        padding = ((height%3 + 1)//2, (width%3 + 1)//2)
        conv2d_1_layer = nn.Conv2d((self.num_blocks+1)*2, 32, 3, padding='same')
        nn.init.uniform_(conv2d_1_layer.weight, -0.01, 0.01)
        nn.init.zeros_(conv2d_1_layer.bias)
        self.conv2d_1 = torch.nn.Sequential(conv2d_1_layer, torch.nn.ELU())
        self.maxpool2d_1 = torch.nn.MaxPool2d(3, padding=padding)
        
        height = (self.max_sentence_len+2)//3 # 50
        width = (self.max_sentence_len+2)//3 # 50
        padding = ((height%3 + 1)//2, (width%3 + 1)//2)
        conv2d_2_layer = nn.Conv2d(32, 32, 3, padding='same') 
        nn.init.uniform_(conv2d_2_layer.weight, -0.01, 0.01)
        nn.init.zeros_(conv2d_2_layer.bias)
        self.conv2d_2 = torch.nn.Sequential(conv2d_2_layer, torch.nn.ELU())
        self.maxpool2d_2 = torch.nn.MaxPool2d(3, padding=padding)
        
        total_flatten_size = mur_flatten_size*2 + ((height+2)//3)*((width+2)//3)*32
        
        self.logits_dense = nn.Linear(total_flatten_size, 1)  
        nn.init.orthogonal_(self.logits_dense.weight)
        nn.init.zeros_(self.logits_dense.bias)
        
        self.bcewithlogitsloss = nn.BCEWithLogitsLoss()
        
    def forward(
        self,
        idx: Optional[torch.Tensor] = None,
        response: Optional[torch.Tensor] = None,
        gt_response: Optional[torch.Tensor] = None,
        narrative: Optional[torch.Tensor] = None,
        utterance: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
    ):
        all_utterances = torch.unbind(utterance, dim=1)
        
        response_embeddings = self.embedding(response)
        Hr_stack = [response_embeddings]
        for i in range(self.num_blocks):
            response_embeddings = self.__getattr__('self_multihead_attention_%d' % i)(
                response_embeddings, response_embeddings, response_embeddings)
            response_embeddings = self.__getattr__('self_feedforward_%d' % i)(response_embeddings)
            Hr_stack.append(response_embeddings)
            
        gt_response_embeddings = self.embedding(gt_response)
        Hgtr_stack = [gt_response_embeddings]
        for i in range(self.num_blocks):
            gt_response_embeddings = self.__getattr__('self_multihead_attention_%d' % i)(
                gt_response_embeddings, gt_response_embeddings, gt_response_embeddings)
            gt_response_embeddings = self.__getattr__('self_feedforward_%d' % i)(gt_response_embeddings)
            Hgtr_stack.append(gt_response_embeddings)
            
        narrative_embeddings = self.embedding(narrative)
        Hn_stack = [narrative_embeddings]
        for i in range(self.num_blocks):
            narrative_embeddings = self.__getattr__('self_multihead_attention_%d' % i)(
                narrative_embeddings, narrative_embeddings, narrative_embeddings)
            narrative_embeddings = self.__getattr__('self_feedforward_%d' % i)(narrative_embeddings)
            Hn_stack.append(narrative_embeddings)
            
        Mur, Mun = [], []
        self.decay_factor = []
        last_u_reps = []
        turn_id = 0
        
        for utterance in all_utterances:
            utterance_embeddings = self.embedding(utterance)
            Hu_stack = [utterance_embeddings]
            for i in range(self.num_blocks):
                utterance_embeddings = self.__getattr__('self_multihead_attention_%d' % i)(
                    utterance_embeddings, utterance_embeddings, utterance_embeddings)
                utterance_embeddings = self.__getattr__('self_feedforward_%d' % i)(utterance_embeddings)
                Hu_stack.append(utterance_embeddings)
                
            if turn_id == self.max_num_utterance - 1:
                last_u_reps = Hu_stack
            
            r_a_u_stack = []
            u_a_r_stack = []
            for i in range(self.num_blocks + 1):
                r_a_u = self.__getattr__('ru_multihead_attention_%d' % i)(
                    Hr_stack[i], Hu_stack[i], Hu_stack[i])
                r_a_u = self.__getattr__('ru_feedforward_%d' % i)(r_a_u)
                r_a_u_stack.append(r_a_u)
                u_a_r = self.__getattr__('ur_multihead_attention_%d' % i)(
                    Hu_stack[i], Hr_stack[i], Hr_stack[i])
                u_a_r = self.__getattr__('ur_feedforward_%d' % i)(u_a_r)
                u_a_r_stack.append(u_a_r)
            r_a_u_stack.extend(Hr_stack)
            u_a_r_stack.extend(Hu_stack)
            
            n_a_u_stack = []
            u_a_n_stack = []
            for i in range(self.num_blocks + 1):
                n_a_u = self.__getattr__('nu_multihead_attention_%d' % i)(
                    Hn_stack[i], Hu_stack[i], Hu_stack[i])
                n_a_u = self.__getattr__('nu_feedforward_%d' % i)(n_a_u)
                n_a_u_stack.append(n_a_u)
                u_a_n = self.__getattr__('un_multihead_attention_%d' % i)(
                    Hu_stack[i], Hn_stack[i], Hn_stack[i])
                u_a_n = self.__getattr__('un_feedforward_%d' % i)(u_a_n)
                u_a_n_stack.append(u_a_n)
            n_a_u_stack.extend(Hn_stack)
            u_a_n_stack.extend(Hu_stack)
            
            r_a_u = torch.stack(r_a_u_stack, dim=-1)
            u_a_r = torch.stack(u_a_r_stack, dim=-1)
            n_a_u = torch.stack(n_a_u_stack, dim=-1)
            u_a_n = torch.stack(u_a_n_stack, dim=-1)
            
            # sim shape [batch, max_sent_len, max_sent_len, 2 * (stack_num + 1)]
            # divide sqrt(200) to prevent gradient explosion
            # (-1, 50, 50, 8)
            sim_ur = torch.einsum('biks,bjks->bijs', u_a_r, r_a_u) / torch.sqrt(torch.tensor(200.0))  # for no rp and normal
            sim_un = torch.einsum('biks,bjks->bijs', u_a_n, n_a_u) / torch.sqrt(torch.tensor(200.0))  # for no rp and normal
            
            self_n = torch.nn.functional.normalize(torch.stack(Hn_stack, dim=-1), p=2, dim=None)  # for no rp
            self_u = torch.nn.functional.normalize(torch.stack(Hu_stack, dim=-1), p=2, dim=None)  # for no rp
            Hn_stack_tensor = torch.stack(Hn_stack, dim=-1)  # [batch, o_len, embedding_size, stack]
            
            self_sim = torch.einsum('biks,bjks->bijs', self_u, self_n)  # [batch, u_len, o_len, stack]
            self_sim = 1 - self.gamma * torch.sum(self_sim, dim=1)  # [batch, (1), o_len, stack]
            Hn_stack = torch.einsum('bjkl,bjl->bjkl', Hn_stack_tensor, self_sim)
            Hn_stack = torch.unbind(Hn_stack, dim=-1)
            
            Mur.append(sim_ur)
            Mun.append(sim_un)
            turn_id += 1
            
        # Hn_stack ( (-1,50,200), ... ) len = block_num
        Hn_stack_for_tracking = self.n_dense(torch.stack(Hn_stack, dim=2))  # [batch, o_len, stack, embedding_size]
        Hn_stack_for_tracking = Hn_stack_for_tracking.permute((0, 1, 3, 2))  # [batch, o_len, embedding_size, stack]
        Hlastu_stack_for_tracking = torch.stack(last_u_reps, dim=-1)  # [batch, u_len, embedding_size, stack]
        Hr_stack_for_tracking = torch.stack(Hgtr_stack, dim=-1)  # [batch, r_len, embedding_size, stack]
        Hlastu = Hlastu_stack_for_tracking.permute((0, 2, 3, 1)) # [batch, embedding_size, stack, u_len]
        Hlastu = torch.squeeze(self.lastu_dense(Hlastu), dim=-1)  # [batch, embedding_size, stack]
        p1_tensor = nn.functional.log_softmax(torch.einsum('bnds,bds->bns', Hn_stack_for_tracking, Hlastu), dim=1)  # [batch, o_len, stack]
        Hlastur = Hr_stack_for_tracking.permute((0, 2, 3, 1))
        Hlastur = torch.squeeze(self.lastur_dense(Hlastur), dim=-1)  # [batch, embedding_size, stack]
        # log_softmax is better
        #p2_tensor = nn.functional.softmax(torch.einsum('bnds,bds->bns', Hn_stack_for_tracking, Hlastur), dim=1)  # [batch, o_len, stack]
        p2_tensor = nn.functional.log_softmax(torch.einsum('bnds,bds->bns', Hn_stack_for_tracking, Hlastur), dim=1)  # [batch, o_len, stack]
        p1 = torch.unbind(p1_tensor, dim=-1)
        p2 = torch.unbind(p2_tensor, dim=-1)
        
        n_a_r_stack = []
        r_a_n_stack = []
        for i in range(self.num_blocks + 1):
            n_a_r = self.__getattr__('nr_multihead_attention_%d' % i)(
                Hn_stack[i], Hr_stack[i], Hr_stack[i])
            n_a_r = self.__getattr__('nr_feedforward_%d' % i)(n_a_r)
            n_a_r_stack.append(n_a_r)
            r_a_n = self.__getattr__('rn_multihead_attention_%d' % i)(
                Hr_stack[i], Hn_stack[i], Hn_stack[i])
            r_a_n = self.__getattr__('rn_feedforward_%d' % i)(r_a_n)
            r_a_n_stack.append(r_a_n)
        n_a_r_stack.extend(Hn_stack)
        r_a_n_stack.extend(Hr_stack)

        n_a_r = torch.stack(n_a_r_stack, dim=-1)
        r_a_n = torch.stack(r_a_n_stack, dim=-1)
        
        Mrn = torch.einsum('biks,bjks->bijs', n_a_r, r_a_n) / torch.sqrt(torch.tensor(200.0))
        self.rosim = Mrn
        Mur = torch.stack(Mur, dim=1)
        Mun = torch.stack(Mun, dim=1)
        
        conv3d = self.conv3d_1(Mur.permute(0,4,1,2,3)) # (-1, 11, 50, 50, 8) -> (-1, 8, 11, 50, 50)
        pool3d = self.maxpool3d_1(conv3d)              # (-1, 32, 4, 17, 17)
        conv3d2 = self.conv3d_2(pool3d)
        pool3d2 = self.maxpool3d_2(conv3d2)            # (-1, 32, 2, 6, 6)
        mur = torch.flatten(pool3d2.permute(0,2,3,4,1), start_dim=1)
        
        conv3d = self.conv3d_1(Mun.permute(0,4,1,2,3))
        pool3d = self.maxpool3d_1(conv3d)
        conv3d2 = self.conv3d_2(pool3d)
        pool3d2 = self.maxpool3d_2(conv3d2)
        mun = torch.flatten(pool3d2.permute(0,2,3,4,1), start_dim=1)
        
        conv2d = self.conv2d_1(Mrn.permute((0,3,1,2)))
        pool2d = self.maxpool2d_1(conv2d)
        conv2d2 = self.conv2d_2(pool2d)
        pool2d2 = self.maxpool2d_2(conv2d2)
        mrn = torch.flatten(pool2d2.permute(0,2,3,1), start_dim=1) 
        
        eps = 1e-7
        KL_loss = 0.0
        for i in range(self.num_blocks + 1):
            #KL_loss += torch.mean(nn.functional.kl_div((p2[i]+eps).log(), p1[i], reduction='batchmean'))
            #KL_loss += torch.mean(nn.functional.kl_div(p2[i], p1[i]+eps, reduction='batchmean'))
            #KL_loss += torch.mean(nn.functional.kl_div(p2[i], torch.clamp(p1[i], min=eps), reduction='batchmean')
            KL_loss += torch.mean(nn.functional.kl_div(p2[i], p1[i], reduction='batchmean', log_target=True))
        KL_loss /= (self.num_blocks + 1)
        
        all_vector = torch.cat([mur, mun, mrn], dim=-1)
        logits = torch.reshape(self.logits_dense(all_vector), shape=(-1,))
        y_pred = torch.sigmoid(logits)
        
        y_true = labels
        RS_loss = torch.mean(torch.clip(self.bcewithlogitsloss(logits, y_true), -10, 10))
        loss = self.eta * RS_loss + (1 - self.eta) * KL_loss
        
        if torch.isnan(KL_loss):
            raise
        
        # This is only for warning removal.
        if not self.scalar_loss:
            loss = torch.unsqueeze(loss, dim=0)
        
        return {
            'loss': loss,
            'y_pred': y_pred,
        }


# In[ ]:


model = ScriptWriter_cpre()


# In[ ]:


from transformers import Trainer, TrainingArguments, TrainerCallback
from typing import Dict, Union, Any
from transformers.utils import  is_apex_available, is_sagemaker_dp_enabled, is_sagemaker_mp_enabled

class ScriptwriterTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bcewithlogitsloss = nn.BCEWithLogitsLoss()
        self.kl_loss = nn.KLDivLoss(reduction = 'batchmean')

    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        
        return (loss, outputs) if return_outputs else loss
        
import Evaluate

def compute_metrics(evalpred):
    
    preds, labels = evalpred
    result = Evaluate.evaluate_all(preds, labels)
            
    return {
        "accuracy" : result[0], 
        "r2@1"     : result[1],
        "r10@1"    : result[2],
        "r10@2"    : result[3],
        "r10@5"    : result[4],
        "mrr"      : result[5],
        "AvgScore" : (result[1]+result[2]+result[3]+result[4]+result[5])/5.0, 
    }


# In[ ]:


from datasets import load_dataset
datasets = load_dataset('story_data', dataset_name)


# In[ ]:


model_checkpoint = "checkpoint"
training_args = TrainingArguments(
    f"{model_checkpoint}-{dataset_name}",
    max_grad_norm = 5.0, # from the original source
    #optim="adamw_torch",
    #learning_rate=0.001,
    #adam_beta1=0.9,
    #adam_beta2=0.98,
    #adam_epsilon=1e-8,
    num_train_epochs=1,
    per_device_train_batch_size = 128,
    per_device_eval_batch_size = 128*4,
    load_best_model_at_end = True,
    metric_for_best_model = "AvgScore",
    evaluation_strategy = "steps",
    save_strategy = "steps", # no, steps   # save_stratge should be same as eval stratege for best 
    eval_steps = 50 if dataset_name == "original" else 100,  # 25000/(batch*GPU)=50 for original 100 for 246578 helper
    save_steps = 50 if dataset_name == "original" else 100,  # save steps should be multiple of eval_steps
    save_total_limit = 2, # limit 1 and best_model automatically set to 2
    logging_steps = 80, # not working????
    logging_first_step = True, # not working???
    report_to="none",
    push_to_hub=False,
    seed=random_seed,
)

#print(training_args.world_size)
#print(training_args._n_gpu)

parallel_no = training_args._n_gpu if training_args._n_gpu > 0 else 1
if parallel_no == 1:
    model.scalar_loss = True
    
training_args.eval_steps = (int(25000/(training_args.per_device_train_batch_size*parallel_no)+25))//50*50
training_args.save_steps = training_args.eval_steps

if dataset_name == 'final':
    training_args.eval_steps *= 16
    training_args.save_steps = training_args.eval_steps


# In[ ]:


from transformers import DefaultDataCollator
from datasets import load_dataset

datasets = load_dataset('story_data', dataset_name)
data_collator = DefaultDataCollator()


# In[ ]:


class patience_scheduler(torch.optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer, last_epoch=-1, verbose=False):
        
        self.lr = optimizer.param_groups[0]['lr']
        self._last_lr = [self.lr]
        
        def lr_lambda(step):
            lr = 0.5**(step+1)
            return lr
        
        super().__init__(optimizer, lr_lambda, last_epoch=-1, verbose=False)
        
    def step(self, from_callback=False):
        if (from_callback):
            super().step()
            print('lr changed to:', self.optimizer.param_groups[0]['lr'])
        else:
            pass

class LrCallback(TrainerCallback):
    "A callback after evaluation"
    def __init__(self):
        self.best_score = -1
        self.patience = 0

    def on_evaluate(self, args, state, control, **kwargs):
        # eval_ is appended to the metric name with Trainer
        name = 'eval_' + args.metric_for_best_model
        score = kwargs['metrics'][name]
        if score > self.best_score:
            self.best_score = score
            self.patience = 0
        else:
            self.patience += 1
            if self.patience >= 3:
                lr_scheduler = kwargs['lr_scheduler']
                lr_scheduler.step(from_callback=True)
                self.patience = 0

optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.98), eps=1e-8)
scheduler = patience_scheduler(optimizer)


# In[ ]:


trainer = ScriptwriterTrainer(
    model=model,
    args=training_args,
    train_dataset=datasets["train"],
    eval_dataset=datasets["validation"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    optimizers=(optimizer, scheduler),
    callbacks=[LrCallback],
)


# **Train**

# In[ ]:


#torch.autograd.set_detect_anomaly(True)


# In[ ]:


trainer.train()


# In[ ]:


# save best model
save_path = f"{model_checkpoint}-{dataset_name}/checkpoint-best"
get_ipython().system('mv $trainer.state.best_model_checkpoint $save_path')


# **Testset Evaluation**

# In[ ]:


trainer._load_from_checkpoint(f"{model_checkpoint}-{dataset_name}/checkpoint-best")


# In[ ]:


metrics = trainer.evaluate(eval_dataset=datasets["test"])
print(metrics)


# In[ ]:




