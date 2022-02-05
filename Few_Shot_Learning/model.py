import torch
from torch.functional import norm
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import math
from itertools import combinations
from torch.nn.init import xavier_normal_ 

from torch.nn.modules.activation import MultiheadAttention

from torch.autograd import Variable
import torchvision.models as models
from utils import extract_class_indices, cos_sim
from einops import rearrange

from soft_dtw import SoftDTW


def initialize_weight(x):
    nn.init.xavier_uniform_(x.weight)
    if x.bias is not None:
        nn.init.constant_(x.bias, 0)

class CNN_FSHead(nn.Module):
    """
    Base class which handles a few-shot method. Contains a resnet backbone which computes features.
    """
    def __init__(self, args):
        super(CNN_FSHead, self).__init__()
        self.train()
        self.args = args

        last_layer_idx = -1
        
        if self.args.backbone == "resnet18":
            backbone = models.resnet18(pretrained=True)  
        elif self.args.backbone == "resnet34":
            backbone = models.resnet34(pretrained=True)
        elif self.args.backbone == "resnet50":
            backbone = models.resnet50(pretrained=True)

        if self.args.pretrained_backbone is not None:
            checkpoint = torch.load(self.args.pretrained_backbone)
            backbone.load_state_dict(checkpoint)

        self.backbone = nn.Sequential(*list(backbone.children())[:last_layer_idx])

    def get_feats(self, support_images, target_images):
        """
        Takes in images from the support set and query video and returns CNN features.
        """
        support_features = self.backbone(support_images).squeeze()
        target_features = self.backbone(target_images).squeeze()

        dim = int(support_features.shape[1])

        support_features = support_features.reshape(-1, self.args.seq_len, dim)
        target_features = target_features.reshape(-1, self.args.seq_len, dim)

        return support_features, target_features

    def forward(self, support_images, support_labels, target_images):
        """
        Should return a dict containing logits which are required for computing accuracy. Dict can also contain
        other info needed to compute the loss. E.g. inter class distances.
        """
        raise NotImplementedError

    def distribute_model(self):
        """
        Use to split the backbone evenly over all GPUs. Modify if you have other components
        """
        if self.args.num_gpus > 1:
            self.backbone.cuda(0)
            self.backbone = torch.nn.DataParallel(self.backbone, device_ids=[i for i in range(0, self.args.num_gpus)])
    
    def loss(self, task_dict, model_dict):
        """
        Takes in a the task dict containing labels etc.
        Takes in the model output dict, which contains "logits", as well as any other info needed to compute the loss.
        Default is cross entropy loss.
        """
        return F.cross_entropy(model_dict["logits"], task_dict["target_labels"].long())
        


class PositionalEncoding(nn.Module):
    """
    Positional encoding from the Transformer paper.
    """
    def __init__(self, d_model, dropout, max_len=5000, pe_scale_factor=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pe_scale_factor = pe_scale_factor
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term) * self.pe_scale_factor
        pe[:, 1::2] = torch.cos(position * div_term) * self.pe_scale_factor
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
                          
    def forward(self, x):
       x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
       return self.dropout(x)





class TemporalCrossTransformer(nn.Module):
    """
    A temporal cross transformer for a single tuple cardinality. E.g. pairs or triples.
    """
    def __init__(self, args, temporal_set_size=3):
        super(TemporalCrossTransformer, self).__init__()
       
        self.args = args
        self.temporal_set_size = temporal_set_size

        max_len = int(self.args.seq_len * 1.5)
        self.pe = PositionalEncoding(self.args.trans_linear_in_dim, self.args.trans_dropout, max_len=max_len)

        self.k_linear = nn.Linear(self.args.trans_linear_in_dim * temporal_set_size, self.args.trans_linear_out_dim)#.cuda()
        self.v_linear = nn.Linear(self.args.trans_linear_in_dim * temporal_set_size, self.args.trans_linear_out_dim)#.cuda()

        self.norm_k = nn.LayerNorm(self.args.trans_linear_out_dim)
        self.norm_v = nn.LayerNorm(self.args.trans_linear_out_dim)
        
        self.class_softmax = torch.nn.Softmax(dim=1)
        
        # generate all tuples
        frame_idxs = [i for i in range(self.args.seq_len)]
        frame_combinations = combinations(frame_idxs, temporal_set_size)
        self.tuples = nn.ParameterList([nn.Parameter(torch.tensor(comb), requires_grad=False) for comb in frame_combinations])
        self.tuples_len = len(self.tuples) 
    
    
    def forward(self, support_set, support_labels, queries):
        n_queries = queries.shape[0]
        n_support = support_set.shape[0]
        
        # static pe
        support_set = self.pe(support_set)
        queries = self.pe(queries)

        # construct new queries and support set made of tuples of images after pe
        s = [torch.index_select(support_set, -2, p).reshape(n_support, -1) for p in self.tuples]
        q = [torch.index_select(queries, -2, p).reshape(n_queries, -1) for p in self.tuples]
        support_set = torch.stack(s, dim=-2)
        queries = torch.stack(q, dim=-2)

        # apply linear maps
        support_set_ks = self.k_linear(support_set)
        queries_ks = self.k_linear(queries)
        support_set_vs = self.v_linear(support_set)
        queries_vs = self.v_linear(queries)
        
        # apply norms where necessary
        mh_support_set_ks = self.norm_k(support_set_ks)
        mh_queries_ks = self.norm_k(queries_ks)
        mh_support_set_vs = support_set_vs
        mh_queries_vs = queries_vs
        
        unique_labels = torch.unique(support_labels)

        # init tensor to hold distances between every support tuple and every target tuple
        all_distances_tensor = torch.zeros(n_queries, self.args.way, device=queries.device)

        for label_idx, c in enumerate(unique_labels):
        
            # select keys and values for just this class
            class_k = torch.index_select(mh_support_set_ks, 0, extract_class_indices(support_labels, c))
            class_v = torch.index_select(mh_support_set_vs, 0, extract_class_indices(support_labels, c))
            k_bs = class_k.shape[0]

            class_scores = torch.matmul(mh_queries_ks.unsqueeze(1), class_k.transpose(-2,-1)) / math.sqrt(self.args.trans_linear_out_dim)

            # reshape etc. to apply a softmax for each query tuple
            class_scores = class_scores.permute(0,2,1,3)
            class_scores = class_scores.reshape(n_queries, self.tuples_len, -1)
            class_scores = [self.class_softmax(class_scores[i]) for i in range(n_queries)]
            class_scores = torch.cat(class_scores)
            class_scores = class_scores.reshape(n_queries, self.tuples_len, -1, self.tuples_len)
            class_scores = class_scores.permute(0,2,1,3)
            
            # get query specific class prototype         
            query_prototype = torch.matmul(class_scores, class_v)
            query_prototype = torch.sum(query_prototype, dim=1)
            
            # calculate distances from queries to query-specific class prototypes
            diff = mh_queries_vs - query_prototype
            norm_sq = torch.norm(diff, dim=[-2,-1])**2
            distance = torch.div(norm_sq, self.tuples_len)
            
            # multiply by -1 to get logits
            distance = distance * -1
            c_idx = c.long()
            all_distances_tensor[:,c_idx] = distance
        
        return_dict = {'logits': all_distances_tensor}
        
        return return_dict


class CNN_TRX(CNN_FSHead):
    """
    Backbone connected to Temporal Cross Transformers of multiple cardinalities.
    """
    def __init__(self, args):
        super(CNN_TRX, self).__init__(args)

        #fill default args
        self.args.trans_linear_out_dim = 1152
        self.args.temp_set = [2,3]
        self.args.trans_dropout = 0.1

        self.transformers = nn.ModuleList([TemporalCrossTransformer(args, s) for s in args.temp_set]) 

    def forward(self, support_images, support_labels, target_images):
        support_features, target_features = self.get_feats(support_images, target_images)
        all_logits = [t(support_features, support_labels, target_features)['logits'] for t in self.transformers]
        all_logits = torch.stack(all_logits, dim=-1)
        sample_logits = all_logits 
        sample_logits = torch.mean(sample_logits, dim=[-1])

        return_dict = {'logits': sample_logits}
        return return_dict

    def distribute_model(self):
        """
        Distributes the CNNs over multiple GPUs. Leaves TRX on GPU 0.
        :return: Nothing
        """
        if self.args.num_gpus > 1:
            self.backbone.cuda(0)
            self.backbone = torch.nn.DataParallel(self.backbone, device_ids=[i for i in range(0, self.args.num_gpus)])

            self.transformers.cuda(0)





def OTAM_cum_dist(dists, lbda=0.1):
    """
    Calculates the OTAM distances for sequences in one direction (e.g. query to support).
    :input: Tensor with frame similarity scores of shape [n_queries, n_support, query_seq_len, support_seq_len] 
    TODO: clearn up if possible - currently messy to work with pt1.8. Possibly due to stack operation?
    """
    dists = F.pad(dists, (1,1), 'constant', 0)

    cum_dists = torch.zeros(dists.shape, device=dists.device)

    # top row
    for m in range(1, dists.shape[3]):
        # cum_dists[:,:,0,m] = dists[:,:,0,m] - lbda * torch.log( torch.exp(- cum_dists[:,:,0,m-1]))
        # paper does continuous relaxation of the cum_dists entry, but it trains faster without, so using the simpler version for now:
        cum_dists[:,:,0,m] = dists[:,:,0,m] + cum_dists[:,:,0,m-1] 


    # remaining rows
    for l in range(1,dists.shape[2]):
        #first non-zero column
        cum_dists[:,:,l,1] = dists[:,:,l,1] - lbda * torch.log( torch.exp(- cum_dists[:,:,l-1,0] / lbda) + torch.exp(- cum_dists[:,:,l-1,1] / lbda) + torch.exp(- cum_dists[:,:,l,0] / lbda) )
        
        #middle columns
        for m in range(2,dists.shape[3]-1):
            cum_dists[:,:,l,m] = dists[:,:,l,m] - lbda * torch.log( torch.exp(- cum_dists[:,:,l-1,m-1] / lbda) + torch.exp(- cum_dists[:,:,l,m-1] / lbda ) )
            
        #last column
        #cum_dists[:,:,l,-1] = dists[:,:,l,-1] - lbda * torch.log( torch.exp(- cum_dists[:,:,l-1,-2] / lbda) + torch.exp(- cum_dists[:,:,l,-2] / lbda) )
        cum_dists[:,:,l,-1] = dists[:,:,l,-1] - lbda * torch.log( torch.exp(- cum_dists[:,:,l-1,-2] / lbda) + torch.exp(- cum_dists[:,:,l-1,-1] / lbda) + torch.exp(- cum_dists[:,:,l,-2] / lbda) )
    
    return cum_dists[:,:,-1,-1]

class CNN_OTAM(CNN_FSHead):
    """
    OTAM with a CNN backbone.
    """
    def __init__(self, args):
        super(CNN_OTAM, self).__init__(args)

    def forward(self, support_images, support_labels, target_images):

        support_features, target_features = self.get_feats(support_images, target_images)
        unique_labels = torch.unique(support_labels)

        n_queries = target_features.shape[0]
        n_support = support_features.shape[0]

        support_features = rearrange(support_features, 'b s d -> (b s) d')
        target_features = rearrange(target_features, 'b s d -> (b s) d')

        frame_sim = cos_sim(target_features, support_features)
        frame_dists = 1 - frame_sim
        
        dists = rearrange(frame_dists, '(tb ts) (sb ss) -> tb sb ts ss', tb = n_queries, sb = n_support)

        # calculate query -> support and support -> query
        cum_dists = OTAM_cum_dist(dists) + OTAM_cum_dist(rearrange(dists, 'tb sb ts ss -> tb sb ss ts'))

        class_dists = [torch.mean(torch.index_select(cum_dists, 1, extract_class_indices(support_labels, c)), dim=1) for c in unique_labels]
        class_dists = torch.stack(class_dists)
        class_dists = rearrange(class_dists, 'c q -> q c')
        return_dict = {'logits': - class_dists}
        return return_dict

    def loss(self, task_dict, model_dict):
        return F.cross_entropy(model_dict["logits"], task_dict["target_labels"].long())



class CNN_TSN(CNN_FSHead):
    """
    TSN with a CNN backbone.
    Either cosine similarity or negative norm squared distance. 
    Use mean distance from query to class videos.
    """
    def __init__(self, args):
        super(CNN_TSN, self).__init__(args)
        self.norm_sq_dist = False


    def forward(self, support_images, support_labels, target_images):
        support_features, target_features = self.get_feats(support_images, target_images)
        unique_labels = torch.unique(support_labels)

        support_features = torch.mean(support_features, dim=1)
        target_features = torch.mean(target_features, dim=1)

        if self.norm_sq_dist:
            class_prototypes = [torch.mean(torch.index_select(support_features, 0, extract_class_indices(support_labels, c)), dim=0) for c in unique_labels]
            class_prototypes = torch.stack(class_prototypes)
            
            diffs = [target_features - class_prototypes[i] for i in unique_labels]
            diffs = torch.stack(diffs)

            norm_sq = torch.norm(diffs, dim=[-1])**2
            distance = - rearrange(norm_sq, 'c q -> q c')
            return_dict = {'logits': distance}

        else:
            class_sim = cos_sim(target_features, support_features)
            class_sim = [torch.mean(torch.index_select(class_sim, 1, extract_class_indices(support_labels, c)), dim=1) for c in unique_labels]
            class_sim = torch.stack(class_sim)
            class_sim = rearrange(class_sim, 'c q -> q c')
            return_dict = {'logits': class_sim}

        return return_dict




class CNN_PAL(CNN_FSHead):
    """
    PAL with a CNN backbone. Cosine similarity as distance measure.
    """
    def __init__(self, args):
        super(CNN_PAL, self).__init__(args)
        self.mha = MultiheadAttention(embed_dim=self.args.trans_linear_in_dim, num_heads=1, dropout=0)
        self.cos_sim = torch.nn.CosineSimilarity()
        self.loss_lambda = 1


    def forward(self, support_images, support_labels, target_images):
        support_features, target_features = self.get_feats(support_images, target_images)
        unique_labels = torch.unique(support_labels)

        support_features = torch.mean(support_features, dim=1)
        target_features = torch.mean(target_features, dim=1)

        support_features = rearrange(support_features, 'n d -> n 1 d')
        target_features = rearrange(target_features, 'n d -> n 1 d')

        support_features = support_features + self.mha(support_features, support_features, support_features)[0]
        target_features = target_features + self.mha(target_features, support_features, support_features)[0]

        support_features = rearrange(support_features, 'b 1 d -> b d')
        target_features = rearrange(target_features, 'b 1 d -> b d')

        prototypes = [torch.mean(torch.index_select(support_features, 0, extract_class_indices(support_labels, c)), dim=0) for c in unique_labels]
        prototypes = torch.stack(prototypes)

        q_s_sim = cos_sim(target_features, prototypes)

        return_dict = {'logits': q_s_sim}

        return return_dict


    def loss(self, task_dict, model_dict):
        """
        Computes cross entropy loss on the logits, and the additional loss between the queries and their correct classes.
        """
        q_s_sim = model_dict["logits"]
        l_meta = F.cross_entropy(q_s_sim, task_dict["target_labels"].long())

        pcc_q_s_sim = q_s_sim
        pcc_q_s_sim = torch.sigmoid(q_s_sim)

        unique_labels = torch.unique(task_dict["support_labels"])
        total_q_c_sim = torch.sum(pcc_q_s_sim, dim=0) + 0.1

        q_c_sim = [torch.sum(torch.index_select(pcc_q_s_sim, 0, extract_class_indices(task_dict["target_labels"], c)), dim=0) for c in unique_labels]
        q_c_sim = torch.stack(q_c_sim)
        q_c_sim = torch.diagonal(q_c_sim)
        q_c_sim = torch.div(q_c_sim, total_q_c_sim)

        l_pcc = - torch.mean(torch.log(q_c_sim))

        return l_meta + self.loss_lambda * l_pcc





class OAP(nn.Module):
    def __init__(self, input_dim):
        super(OAP, self).__init__()
        self.input_dim = input_dim
        self.att_size = att_size = 30  #input_dim 30
        self.scale = att_size ** -0.5

        self.linear_q = nn.Linear(input_dim, att_size, bias=False)
        self.linear_k = nn.Linear(input_dim, att_size, bias=False)        
        initialize_weight(self.linear_q)
        initialize_weight(self.linear_k)

        self.lam = 50.
        self.device = torch.device('cuda' if (torch.cuda.is_available() ) else 'cpu')

    def forward(self, seq_q, seq_k):
        # len_q = len_q.detach()
        # len_k = len_k.detach()
        T_q = seq_q.size(1)
        T_k = seq_k.size(1)
        len_q = torch.tensor([float(t+1)/float(T_q) for t in range(T_q)])
        len_k = torch.tensor([float(t+1)/float(T_k) for t in range(T_k)])
        P = torch.cdist(len_q.view(T_q,1), len_k.view(T_k,1), 2).view(1,1,T_q,T_k).to(self.device)

        n_queries = seq_q.shape[0]
        n_support = seq_k.shape[0]
        #print(n_queries,n_support)

        query_features = rearrange(seq_q, 'b s d -> (b s) d')
        support_features = rearrange(seq_k, 'b s d -> (b s) d')

        q = self.linear_q(query_features).view(1, -1, self.att_size)        
        k = self.linear_k(support_features).view(1, -1, self.att_size)  

        A = torch.cdist(q, k, 2).squeeze(0)
        D = torch.cdist(seq_q.view(1, -1, self.input_dim), seq_k.view(1, -1, self.input_dim), 2).squeeze(0)
        A = rearrange(A, '(tb ts) (sb ss) -> tb sb ts ss', tb = n_queries, sb = n_support)
        D = rearrange(D, '(tb ts) (sb ss) -> tb sb ts ss', tb = n_queries, sb = n_support)
        A = A + self.lam*P #+ A2 #torch.reciprocal(P)
        A = torch.softmax(-A.mul_(self.scale), dim=3)

        At = A.view(n_queries*n_support,T_q*T_k,1)
        #At = rearrange(A, 'tb sb ts ss -> (tb sb) (ts ss)', tb = n_queries, sb = n_support).unsqueeze(-1)
        norm = At.norm(dim=1, p=1, keepdim=True)
        At = At.div(norm.expand_as(At))  #.view(batch_size,T_q*T_k,1)

        #dis = torch.matmul(D.view(n_queries*n_support,1,T_q*T_k),At) #/T_k
        D = rearrange(D, 'tb sb ts ss -> (tb sb) (ts ss)', tb = n_queries, sb = n_support).unsqueeze(1)
        dis = torch.matmul(D,At)
        dis = dis.view(n_queries, n_support)
        return dis

class OAPconv2(nn.Module):
    def __init__(self, input_dim):
        super(OAPconv2, self).__init__()
        #self.lam = lam
        self.input_dim = input_dim
        self.att_size = att_size = 64  #input_dim
        self.att_size2 = att_size2 = 100
        #self.att_size3 = att_size3 = 30
        self.scale = att_size ** -0.5
        self.scale2 = att_size2 ** -0.5

        self.conv1 = nn.Conv2d(2, 64, 5, padding=2)
        self.conv2 = nn.Conv2d(64, 64, 5, padding=2)
        self.conv3 = nn.Conv2d(64, 1, 3, padding=1)
        self.device = torch.device('cuda' if (torch.cuda.is_available() ) else 'cpu')

    def forward(self, seq_q, seq_k):
        T_q = seq_q.size(1)
        T_k = seq_k.size(1)
        len_q = torch.tensor([float(t+1)/float(T_q) for t in range(T_q)])
        len_k = torch.tensor([float(t+1)/float(T_k) for t in range(T_k)])
        P = torch.cdist(len_q.view(T_q,1), len_k.view(T_k,1), 2).view(1,1,T_q,T_k).to(self.device)       

        n_queries = seq_q.shape[0]
        n_support = seq_k.shape[0]
        
        D = torch.cdist(seq_q.view(1, -1, self.input_dim), seq_k.view(1, -1, self.input_dim), 2).squeeze(0)
        #A = rearrange(A, '(tb ts) (sb ss) -> tb sb ts ss', tb = n_queries, sb = n_support)
        D = rearrange(D, '(tb ts) (sb ss) -> (tb sb) ts ss', tb = n_queries, sb = n_support)
        # A = A + self.lam*P #+ A2 #torch.reciprocal(P)
        # A = torch.softmax(-A.mul_(self.scale), dim=3)

        P = P.expand(n_queries*n_support, 1, T_q, T_k)
        A = torch.cat([D.unsqueeze(1), P], 1)
        #A = torch.cat([D, P], 0).unsqueeze(0)
        A = F.relu(self.conv1(A))
        A = F.relu(self.conv2(A))
        A = self.conv3(A).squeeze(1) + D
        A = torch.softmax(-A, dim=2)

        At = A.view(n_queries*n_support,T_q*T_k,1)
        #At = rearrange(A, 'tb sb ts ss -> (tb sb) (ts ss)', tb = n_queries, sb = n_support).unsqueeze(-1)
        norm = At.norm(dim=1, p=1, keepdim=True)
        At = At.div(norm.expand_as(At))  #.view(batch_size,T_q*T_k,1)

        #dis = torch.matmul(D.view(n_queries*n_support,1,T_q*T_k),At) #/T_k
        D = rearrange(D, '(tb sb) ts ss -> (tb sb) (ts ss)', tb = n_queries, sb = n_support).unsqueeze(1)
        dis = torch.matmul(D,At)
        
        dis = dis.view(n_queries, n_support)
        return dis


class OAPconv2_deep(nn.Module):
    def __init__(self, input_dim): #, lam=1.
        super(OAPconv2_deep, self).__init__()
        #self.lam = lam
        self.input_dim = input_dim
        self.att_size = att_size = 64  #input_dim
        self.att_size2 = att_size2 = 100
        #self.att_size3 = att_size3 = 30
        self.scale = att_size ** -0.5
        self.scale2 = att_size2 ** -0.5

        self.conv1 = nn.Conv2d(2, 64, 5, padding=2)
        self.conv2 = nn.Conv2d(64, 64, 5, padding=2)
        self.conv3 = nn.Conv2d(64, 64, 5, padding=2)
        self.conv4 = nn.Conv2d(64, 1, 3, padding=1)
        self.device = torch.device('cuda' if (torch.cuda.is_available() ) else 'cpu')

    def forward(self, seq_q, seq_k):
        T_q = seq_q.size(1)
        T_k = seq_k.size(1)
        len_q = torch.tensor([float(t+1)/float(T_q) for t in range(T_q)])
        len_k = torch.tensor([float(t+1)/float(T_k) for t in range(T_k)])
        P = torch.cdist(len_q.view(T_q,1), len_k.view(T_k,1), 2).view(1,1,T_q,T_k).to(self.device)       

        n_queries = seq_q.shape[0]
        n_support = seq_k.shape[0]
        
        D = torch.cdist(seq_q.view(1, -1, self.input_dim), seq_k.view(1, -1, self.input_dim), 2).squeeze(0)
        #A = rearrange(A, '(tb ts) (sb ss) -> tb sb ts ss', tb = n_queries, sb = n_support)
        D = rearrange(D, '(tb ts) (sb ss) -> (tb sb) ts ss', tb = n_queries, sb = n_support)
        # A = A + self.lam*P #+ A2 #torch.reciprocal(P)
        # A = torch.softmax(-A.mul_(self.scale), dim=3)

        P = P.expand(n_queries*n_support, 1, T_q, T_k)
        A = torch.cat([D.unsqueeze(1), P], 1)
        #A = torch.cat([D, P], 0).unsqueeze(0)
        A = F.relu(self.conv1(A))
        A = F.relu(self.conv2(A))
        A = F.relu(self.conv3(A))
        A = self.conv4(A).squeeze(1) + D
        A = torch.softmax(-A, dim=2)

        At = A.view(n_queries*n_support,T_q*T_k,1)
        #At = rearrange(A, 'tb sb ts ss -> (tb sb) (ts ss)', tb = n_queries, sb = n_support).unsqueeze(-1)
        norm = At.norm(dim=1, p=1, keepdim=True)
        At = At.div(norm.expand_as(At))  #.view(batch_size,T_q*T_k,1)

        #dis = torch.matmul(D.view(n_queries*n_support,1,T_q*T_k),At) #/T_k
        D = rearrange(D, '(tb sb) ts ss -> (tb sb) (ts ss)', tb = n_queries, sb = n_support).unsqueeze(1)
        dis = torch.matmul(D,At)
        
        dis = dis.view(n_queries, n_support)
        return dis



class CNN_OAP(CNN_FSHead):
    """
    OAP with a CNN backbone.
    """
    def __init__(self, args):
        super(CNN_OAP, self).__init__(args)
        #self.alignment = OAP(2048)
        self.alignment = OAPconv2(2048)

    def forward(self, support_images, support_labels, target_images):

        support_features, target_features = self.get_feats(support_images, target_images)
        #print(support_features.shape)
        unique_labels = torch.unique(support_labels)

        cum_dists = self.alignment(target_features,support_features) + rearrange(self.alignment(support_features,target_features), 'tb sb -> sb tb')
        #print(cum_dists.shape)

        class_dists = [torch.mean(torch.index_select(cum_dists, 1, extract_class_indices(support_labels, c)), dim=1) for c in unique_labels]
        class_dists = torch.stack(class_dists)
        class_dists = rearrange(class_dists, 'c q -> q c')
        return_dict = {'logits': - class_dists}
        return return_dict

    def loss(self, task_dict, model_dict):
        return F.cross_entropy(model_dict["logits"], task_dict["target_labels"].long())


class CNN_SA_OAP(CNN_FSHead):
    """
    OAP with a CNN backbone.
    """
    def __init__(self, args):
        super(CNN_SA_OAP, self).__init__(args)
        

        self.args = args
        self.args.trans_linear_out_dim = self.args.trans_linear_in_dim
        #self.alignment = OAP(self.args.trans_linear_out_dim)
        self.alignment = OAPconv2(self.args.trans_linear_out_dim)
        self.args.trans_dropout = 0
        max_len = int(self.args.seq_len * 1.5)
        self.pe = PositionalEncoding(self.args.trans_linear_in_dim, self.args.trans_dropout, max_len=max_len)

        self.q_linear = nn.Linear(self.args.trans_linear_in_dim, self.args.trans_linear_out_dim)
        self.k_linear = nn.Linear(self.args.trans_linear_in_dim, self.args.trans_linear_out_dim)
        self.v_linear = nn.Linear(self.args.trans_linear_in_dim, self.args.trans_linear_out_dim)
        self.norm_v = nn.LayerNorm(self.args.trans_linear_out_dim)
        
        self.softmax = torch.nn.Softmax(dim=-1)


    def forward(self, support_images, support_labels, target_images):

        support_features, target_features = self.get_feats(support_images, target_images)
        # print(support_features.shape)
        # print(target_features.shape)

        n_queries = target_features.shape[0]
        n_support = support_features.shape[0]
        
        # static pe
        support_set = self.pe(support_features) + support_features
        queries = self.pe(target_features) + target_features

        # apply linear maps
        support_set_qs = self.q_linear(support_set)
        queries_qs = self.q_linear(queries)
        support_set_ks = self.k_linear(support_set)
        queries_ks = self.k_linear(queries)
        support_set_vs = self.v_linear(support_set)
        queries_vs = self.v_linear(queries)
        # print(queries_qs.shape)

        att_query_q = torch.matmul(queries_qs, queries_ks.transpose(-2,-1)) / math.sqrt(self.args.trans_linear_out_dim)
        # print(att_query_q.shape).unsqueeze(1).unsqueeze(1)
        att_support_q = torch.matmul(support_set_qs, support_set_ks.transpose(-2,-1)) / math.sqrt(self.args.trans_linear_out_dim)
        att_query_q = self.softmax(att_query_q)
        att_support_q = self.softmax(att_support_q)
        # print(att_query_q.shape, support_set_vs.shape)
        # print(att_support_q.shape, queries_vs.shape).unsqueeze(0).unsqueeze(0)
        context_query_q = torch.matmul(att_query_q,queries_vs)
        context_support_q = torch.matmul(att_support_q,support_set_vs)

        target_features = self.norm_v(queries + context_query_q)
        support_features = self.norm_v(support_set + context_support_q)
        
        unique_labels = torch.unique(support_labels)
        
        cum_dists = self.alignment(target_features,support_features) + rearrange(self.alignment(support_features,target_features), 'tb sb -> sb tb')
        #print(cum_dists.shape)

        class_dists = [torch.mean(torch.index_select(cum_dists, 1, extract_class_indices(support_labels, c)), dim=1) for c in unique_labels]
        class_dists = torch.stack(class_dists)
        class_dists = rearrange(class_dists, 'c q -> q c')
        return_dict = {'logits': - class_dists}
        return return_dict

    def loss(self, task_dict, model_dict):
        return F.cross_entropy(model_dict["logits"], task_dict["target_labels"].long())



class CNN_SoftDTW(CNN_FSHead):
    """
    SoftDTW with a CNN backbone.
    
    """
    def __init__(self, args):
        super(CNN_SoftDTW, self).__init__(args)
        self.alignment = SoftDTW(gamma=0.1)
        #self.alignment = SoftDTW(gamma=10)

    def forward(self, support_images, support_labels, target_images):

        support_features, target_features = self.get_feats(support_images, target_images)
        unique_labels = torch.unique(support_labels)

        n_queries = target_features.shape[0]
        n_support = support_features.shape[0]

        target_features_temp = target_features[0:1,:,:].expand(n_support, target_features.shape[1], target_features.shape[2])
        cum_dists = self.alignment(target_features_temp,support_features).view(1,n_support)

        for i in range(1,n_queries):
            target_features_temp = target_features[i:i+1,:,:].expand(n_support, target_features.shape[1], target_features.shape[2])
            dis = self.alignment(target_features_temp,support_features).view(1,n_support)
            cum_dists = torch.cat((cum_dists,dis),0)

        class_dists = [torch.mean(torch.index_select(cum_dists, 1, extract_class_indices(support_labels, c)), dim=1) for c in unique_labels]
        class_dists = torch.stack(class_dists)
        class_dists = rearrange(class_dists, 'c q -> q c')
        return_dict = {'logits': - class_dists}
        return return_dict

    def loss(self, task_dict, model_dict):
        return F.cross_entropy(model_dict["logits"], task_dict["target_labels"].long())


if __name__ == "__main__":
    class ArgsObject(object):
        def __init__(self):
            self.trans_linear_in_dim = 512
            self.trans_linear_out_dim = 128

            self.way = 4
            self.shot = 3
            self.query_per_class = 2
            self.trans_dropout = 0.1
            self.seq_len = 5 
            self.img_size = 84
            self.backbone = "resnet18"
            self.num_gpus = 1
            self.temp_set = [2,3]
            self.pretrained_backbone = None
    args = ArgsObject()
    torch.manual_seed(0)
    
    device = 'cpu'
    # device = 'cuda:0'
    # model = CNN_TRX(args).to(device)
    model = CNN_OTAM(args).to(device)
    # model = CNN_TSN(args).to(device)
    # model = CNN_PAL(args).to(device)
    
    support_imgs = torch.rand(args.way * args.shot * args.seq_len,3, args.img_size, args.img_size).to(device)
    target_imgs = torch.rand(args.way * args.query_per_class * args.seq_len ,3, args.img_size, args.img_size).to(device)
    support_labels = torch.tensor([n for n in range(args.way)] * args.shot).to(device)
    target_labels = torch.tensor([n for n in range(args.way)] * args.query_per_class).to(device)

    print("Support images input shape: {}".format(support_imgs.shape))
    print("Target images input shape: {}".format(target_imgs.shape))
    print("Support labels input shape: {}".format(support_imgs.shape))

    task_dict = {}
    task_dict["support_set"] = support_imgs
    task_dict["support_labels"] = support_labels
    task_dict["target_set"] = target_imgs
    task_dict["target_labels"] = target_labels

    model_dict = model(support_imgs, support_labels, target_imgs)
    print("Model returns the distances from each query to each class prototype.  Use these as logits.  Shape: {}".format(model_dict['logits'].shape))

    loss = model.loss(task_dict, model_dict)







