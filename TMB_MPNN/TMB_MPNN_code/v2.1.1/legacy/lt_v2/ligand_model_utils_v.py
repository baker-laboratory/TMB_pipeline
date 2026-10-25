from __future__ import print_function
import numpy as np
import torch
import torch.utils
import torch.utils.checkpoint
import torch.nn as nn

def featurize(batch, device, mode="backbone", num_nn=4):
    batch = [b for b in batch if type(b[0])!=list]
    B = len(batch)
    if B > 0:
        L_stack = torch.stack([b[1] for b in batch])
        L_max = torch.max(L_stack)
        if mode == "backbone":
            X = torch.zeros([B, L_max, 4, 3], dtype=torch.float32)
            X_m = torch.zeros([B, L_max, 4], dtype=torch.int32)
        elif mode == "all":
            X = torch.zeros([B, L_max, 37, 3], dtype=torch.float32)
            X_m = torch.zeros([B, L_max, 37], dtype=torch.int32)

        Y = torch.zeros([B, L_max, num_nn, 3], dtype=torch.float32)
        Y_m = torch.zeros([B, L_max, num_nn], dtype=torch.int32)
        Y_t = torch.zeros([B, L_max, num_nn], dtype=torch.int32)

        mask = torch.zeros([B, L_max], dtype=torch.float32)
        S = 20*torch.ones([B, L_max], dtype=torch.int64)
        R_idx = -100*torch.ones([B, L_max], dtype=torch.int32)
        chain_labels = -1*torch.ones([B, L_max], dtype=torch.int64)
        for i, b in enumerate(batch):
            out_dict = b[0]
            X[i,:L_stack[i]] = out_dict["X"][None,]
            X_m[i,:L_stack[i]] = out_dict["X_m"][None,]
            mask[i,:L_stack[i]] = torch.ones_like(out_dict["S"][None,], dtype=torch.float32)
            S[i,:L_stack[i]] = out_dict["S"][None,]
            R_idx[i,:L_stack[i]] = out_dict["R_idx"][None,]
            chain_labels[i,:L_stack[i]] = out_dict["chain_labels"][None,]

            num_atoms = out_dict["Y"].shape[-2]
            Y[i,:L_stack[i],:num_atoms] = out_dict["Y"][None,]
            Y_t[i,:L_stack[i], :num_atoms] = out_dict["Y_t"][None,]
            mask_t_0 = (out_dict["Y_t"][None,] != 0)
            Y_m[i,:L_stack[i], :num_atoms] = mask_t_0*torch.ones_like(out_dict["Y_t"][None,], dtype=torch.int32)

        out_dict = {}

        out_dict["X"] = X.to(device) #[B, L, num_prot_atoms, 3]
        out_dict["X_m"] = X_m.to(device) #[B, L, num_prot_atoms]
        out_dict["mask"] = mask.to(device) #[B, L, num_prot_atoms]

        out_dict["S"] = S.long().to(device) #[B, L]
        out_dict["R_idx"] = R_idx.to(device) #[B, L]
        out_dict["chain_labels"] = chain_labels.to(device) #[B, L]

        out_dict["Y"] = Y.to(device) #[B, L, num_nn, 3]
        out_dict["Y_t"] = Y_t.to(device) #[B, L, num_nn]
        out_dict["Y_m"] = Y_m.to(device) #[B, L, num_nn]

        return out_dict

    else:
        return "pass"

def loss_nll(S, log_probs, mask):
    """ Negative log probabilities """
    criterion = torch.nn.NLLLoss(reduction='none')
    loss = criterion(
        log_probs.contiguous().view(-1, log_probs.size(-1)), S.contiguous().view(-1)
    ).view(S.size())
    S_argmaxed = torch.argmax(log_probs,-1) #[B, L]
    true_false = (S == S_argmaxed).float()
    loss_av = torch.sum(loss * mask) / torch.sum(mask)
    return loss, loss_av, true_false


def loss_smoothed(S, log_probs, mask, weight=0.1, tokens=2000.0):
    """ Negative log probabilities """
    S_onehot = torch.nn.functional.one_hot(S, 21).float()

    # Label smoothing
    S_onehot = S_onehot + weight / float(S_onehot.size(-1))
    S_onehot = S_onehot / S_onehot.sum(-1, keepdim=True)

    loss = -(S_onehot * log_probs).sum(-1)
    loss_av = torch.sum(loss * mask) / tokens #fixed 
    return loss, loss_av


# The following gather functions
def gather_edges(edges, neighbor_idx):
    # Features [B,N,N,C] at Neighbor indices [B,N,K] => Neighbor features [B,N,K,C]
    neighbors = neighbor_idx.unsqueeze(-1).expand(-1, -1, -1, edges.size(-1))
    edge_features = torch.gather(edges, 2, neighbors)
    return edge_features

def gather_nodes(nodes, neighbor_idx):
    # Features [B,N,C] at Neighbor indices [B,N,K] => [B,N,K,C]
    # Flatten and expand indices per batch [B,N,K] => [B,NK] => [B,NK,C]
    neighbors_flat = neighbor_idx.reshape((neighbor_idx.shape[0], -1))
    neighbors_flat = neighbors_flat.unsqueeze(-1).expand(-1, -1, nodes.size(2))
    # Gather and re-pack
    neighbor_features = torch.gather(nodes, 1, neighbors_flat)
    neighbor_features = neighbor_features.view(list(neighbor_idx.shape)[:3] + [-1])
    return neighbor_features

def gather_nodes_t(nodes, neighbor_idx):
    # Features [B,N,C] at Neighbor index [B,K] => Neighbor features[B,K,C]
    idx_flat = neighbor_idx.unsqueeze(-1).expand(-1, -1, nodes.size(2))
    neighbor_features = torch.gather(nodes, 1, idx_flat)
    return neighbor_features

def cat_neighbors_nodes(h_nodes, h_neighbors, E_idx):
    h_nodes = gather_nodes(h_nodes, E_idx)
    h_nn = torch.cat([h_neighbors, h_nodes], -1)
    return h_nn


class EncLayer(nn.Module):
    def __init__(self, num_hidden, num_in, dropout=0.1, num_heads=None, scale=30):
        super(EncLayer, self).__init__()
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.scale = scale
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(num_hidden)
        self.norm2 = nn.LayerNorm(num_hidden)
        self.norm3 = nn.LayerNorm(num_hidden)

        self.W1 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W2 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W3 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W11 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W12 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W13 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.act = torch.nn.GELU()
        self.dense = PositionWiseFeedForward(num_hidden, num_hidden * 4)

    def forward(self, h_V, h_E, E_idx, mask_V=None, mask_attend=None):
        """ Parallel computation of full transformer layer """

        h_EV = cat_neighbors_nodes(h_V, h_E, E_idx)
        h_V_expand = h_V.unsqueeze(-2).expand(-1,-1,h_EV.size(-2),-1)
        h_EV = torch.cat([h_V_expand, h_EV], -1)
        h_message = self.W3(self.act(self.W2(self.act(self.W1(h_EV)))))
        if mask_attend is not None:
            h_message = mask_attend.unsqueeze(-1) * h_message
        dh = torch.sum(h_message, -2) / self.scale
        h_V = self.norm1(h_V + self.dropout1(dh))

        dh = self.dense(h_V)
        h_V = self.norm2(h_V + self.dropout2(dh))
        if mask_V is not None:
            mask_V = mask_V.unsqueeze(-1)
            h_V = mask_V * h_V

        h_EV = cat_neighbors_nodes(h_V, h_E, E_idx)
        h_V_expand = h_V.unsqueeze(-2).expand(-1,-1,h_EV.size(-2),-1)
        h_EV = torch.cat([h_V_expand, h_EV], -1)
        h_message = self.W13(self.act(self.W12(self.act(self.W11(h_EV)))))
        h_E = self.norm3(h_E + self.dropout3(h_message))
        return h_V, h_E



class DecLayer(nn.Module):
    def __init__(self, num_hidden, num_in, dropout=0.1, num_heads=None, scale=30):
        super(DecLayer, self).__init__()
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.scale = scale
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(num_hidden)
        self.norm2 = nn.LayerNorm(num_hidden)

        self.W1 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W2 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W3 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.act = torch.nn.GELU()
        self.dense = PositionWiseFeedForward(num_hidden, num_hidden * 4)

    def forward(self, h_V, h_E, mask_V=None, mask_attend=None):
        """ Parallel computation of full transformer layer """

        # Concatenate h_V_i to h_E_ij
        h_V_expand = h_V.unsqueeze(-2).expand(-1,-1,h_E.size(-2),-1)
        h_EV = torch.cat([h_V_expand, h_E], -1)

        h_message = self.W3(self.act(self.W2(self.act(self.W1(h_EV)))))
        if mask_attend is not None:
            h_message = mask_attend.unsqueeze(-1) * h_message
        dh = torch.sum(h_message, -2) / self.scale

        h_V = self.norm1(h_V + self.dropout1(dh))

        # Position-wise feedforward
        dh = self.dense(h_V)
        h_V = self.norm2(h_V + self.dropout2(dh))

        if mask_V is not None:
            mask_V = mask_V.unsqueeze(-1)
            h_V = mask_V * h_V
        return h_V



class DecLayerJ(nn.Module):
    def __init__(self, num_hidden, num_in, dropout=0.1, num_heads=None, scale=30):
        super(DecLayerJ, self).__init__()
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.scale = scale
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(num_hidden)
        self.norm2 = nn.LayerNorm(num_hidden)

        self.W1 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W2 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W3 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.act = torch.nn.GELU()
        self.dense = PositionWiseFeedForward(num_hidden, num_hidden * 4)

    def forward(self, h_V, h_E, mask_V=None, mask_attend=None):
        """ Parallel computation of full transformer layer """

        # Concatenate h_V_i to h_E_ij
        h_V_expand = h_V.unsqueeze(-2).expand(-1,-1,-1,h_E.size(-2),-1) #the only difference
        h_EV = torch.cat([h_V_expand, h_E], -1)

        h_message = self.W3(self.act(self.W2(self.act(self.W1(h_EV)))))
        if mask_attend is not None:
            h_message = mask_attend.unsqueeze(-1) * h_message
        dh = torch.sum(h_message, -2) / self.scale

        h_V = self.norm1(h_V + self.dropout1(dh))

        # Position-wise feedforward
        dh = self.dense(h_V)
        h_V = self.norm2(h_V + self.dropout2(dh))

        if mask_V is not None:
            mask_V = mask_V.unsqueeze(-1)
            h_V = mask_V * h_V
        return h_V

class PositionWiseFeedForward(nn.Module):
    def __init__(self, num_hidden, num_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.W_in = nn.Linear(num_hidden, num_ff, bias=True)
        self.W_out = nn.Linear(num_ff, num_hidden, bias=True)
        self.act = torch.nn.GELU()
    def forward(self, h_V):
        h = self.act(self.W_in(h_V))
        h = self.W_out(h)
        return h

class PositionalEncodings(nn.Module):
    def __init__(self, num_embeddings, max_relative_feature=32):
        super(PositionalEncodings, self).__init__()
        self.num_embeddings = num_embeddings
        self.max_relative_feature = max_relative_feature
        self.linear = nn.Linear(2*max_relative_feature+1+1, num_embeddings)

    def forward(self, offset, mask):
        d = torch.clip(offset + self.max_relative_feature, 0, 2*self.max_relative_feature)*mask + (1-mask)*(2*self.max_relative_feature+1)
        d_onehot = torch.nn.functional.one_hot(d, 2*self.max_relative_feature+1+1)
        E = self.linear(d_onehot.float())
        return E


class ProteinFeatures(nn.Module):
    def __init__(self, edge_features, node_features, num_positional_embeddings=16,
        num_rbf=16, top_k=30, augment_eps=0., num_chain_embeddings=16, atom37_order=False, device=None, atom_context_num=16):
        """ Extract protein features """
        super(ProteinFeatures, self).__init__()

        self.N_idx = 0
        self.CA_idx = 1
        self.C_idx = 2

        if atom37_order:
            self.O_idx = 4
        else:
            self.O_idx = 3

        self.edge_features = edge_features
        self.node_features = node_features
        self.top_k = top_k
        self.augment_eps = augment_eps 
        self.num_rbf = num_rbf
        self.num_positional_embeddings = num_positional_embeddings

        self.embeddings = PositionalEncodings(num_positional_embeddings)
        node_in, edge_in = 6, num_positional_embeddings + num_rbf*25
        self.edge_embedding = nn.Linear(edge_in, edge_features, bias=False)
        self.norm_edges = nn.LayerNorm(edge_features)


        self.node_project_down = nn.Linear(5*num_rbf+64+4, node_features, bias=True)
        self.norm_nodes = nn.LayerNorm(node_features)


        self.type_linear = nn.Linear(147, 64)


        self.y_nodes = nn.Linear(147, node_features, bias=False)
        self.y_edges = nn.Linear(num_rbf, node_features, bias=False)

        self.norm_y_edges = nn.LayerNorm(node_features)
        self.norm_y_nodes = nn.LayerNorm(node_features)

        self.atom_context_num = atom_context_num
        self.side_chain_atom_types = torch.tensor([6, 6, 6, 8, 8, 16, 6, 6, 6, 7, 7, 8, 8, 16, 6, 6, 6, 6, 7, 7, 7, 8, 8, 6, 7, 7, 8, 6, 6, 6, 7, 8], device=device)




        self.periodic_table_features = torch.tensor([[  0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,  13,
          14,  15,  16,  17,  18,  19,  20,  21,  22,  23,  24,  25,  26,  27,
          28,  29,  30,  31,  32,  33,  34,  35,  36,  37,  38,  39,  40,  41,
          42,  43,  44,  45,  46,  47,  48,  49,  50,  51,  52,  53,  54,  55,
          56,  57,  58,  59,  60,  61,  62,  63,  64,  65,  66,  67,  68,  69,
          70,  71,  72,  73,  74,  75,  76,  77,  78,  79,  80,  81,  82,  83,
          84,  85,  86,  87,  88,  89,  90,  91,  92,  93,  94,  95,  96,  97,
          98,  99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111,
         112, 113, 114, 115, 116, 117, 118],
        [  0,   1,  18,   1,   2,  13,  14,  15,  16,  17,  18,   1,   2,  13,
          14,  15,  16,  17,  18,   1,   2,   3,   4,   5,   6,   7,   8,   9,
          10,  11,  12,  13,  14,  15,  16,  17,  18,   1,   2,   3,   4,   5,
           6,   7,   8,   9,  10,  11,  12,  13,  14,  15,  16,  17,  18,   1,
           2,   3,   3,   3,   3,   3,   3,   3,   3,   3,   3,   3,   3,   3,
           3,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,  13,  14,  15,
          16,  17,  18,   1,   2,   3,   3,   3,   3,   3,   3,   3,   3,   3,
           3,   3,   3,   3,   3,   3,   4,   5,   6,   7,   8,   9,  10,  11,
          12,  13,  14,  15,  16,  17,  18],
        [  0,   1,   1,   2,   2,   2,   2,   2,   2,   2,   2,   3,   3,   3,
           3,   3,   3,   3,   3,   4,   4,   4,   4,   4,   4,   4,   4,   4,
           4,   4,   4,   4,   4,   4,   4,   4,   4,   5,   5,   5,   5,   5,
           5,   5,   5,   5,   5,   5,   5,   5,   5,   5,   5,   5,   5,   6,
           6,   6,   6,   6,   6,   6,   6,   6,   6,   6,   6,   6,   6,   6,
           6,   6,   6,   6,   6,   6,   6,   6,   6,   6,   6,   6,   6,   6,
           6,   6,   6,   7,   7,   7,   7,   7,   7,   7,   7,   7,   7,   7,
           7,   7,   7,   7,   7,   7,   7,   7,   7,   7,   7,   7,   7,   7,
           7,   7,   7,   7,   7,   7,   7]], dtype=torch.long, device=device)




    def _dist(self, X, mask, eps=1E-6):
        mask_2D = torch.unsqueeze(mask,1) * torch.unsqueeze(mask,2)
        dX = torch.unsqueeze(X,1) - torch.unsqueeze(X,2)
        D = mask_2D * torch.sqrt(torch.sum(dX**2, 3) + eps)
        D_max, _ = torch.max(D, -1, keepdim=True)
        D_adjust = D + (1. - mask_2D) * D_max
        sampled_top_k = self.top_k
        D_neighbors, E_idx = torch.topk(D_adjust, np.minimum(self.top_k, X.shape[1]), dim=-1, largest=False)
        return D_neighbors, E_idx

    def _rbf(self, D):
        device = D.device
        D_min, D_max, D_count = 2., 22., self.num_rbf
        D_mu = torch.linspace(D_min, D_max, D_count, device=device)
        D_mu = D_mu.view([1,1,1,-1])
        D_sigma = (D_max - D_min) / D_count
        D_expand = torch.unsqueeze(D, -1)
        RBF = torch.exp(-((D_expand - D_mu) / D_sigma)**2)
        return RBF

    def _get_rbf(self, A, B, E_idx):
        D_A_B = torch.sqrt(torch.sum((A[:,:,None,:] - B[:,None,:,:])**2,-1) + 1e-6) #[B, L, L]
        D_A_B_neighbors = gather_edges(D_A_B[:,:,:,None], E_idx)[:,:,:,0] #[B,L,K]
        RBF_A_B = self._rbf(D_A_B_neighbors)
        return RBF_A_B


    def _make_angle_features(self, A, B, C, Y):
        v1 = A - B
        v2 = C - B
        e1 = torch.nn.functional.normalize(v1, dim=-1)
        e1_v2_dot = torch.einsum("bli, bli -> bl", e1, v2)[...,None]
        u2 = v2 - e1*e1_v2_dot
        e2 = torch.nn.functional.normalize(u2, dim=-1)
        e3 = torch.cross(e1, e2, dim=-1)
        R_residue = torch.cat((e1[:,:,:,None],e2[:,:,:,None],e3[:,:,:,None]), dim=-1)

        local_vectors = torch.einsum("blqp, blyq -> blyp", R_residue, Y-B[:,:,None,:])

        rxy = torch.sqrt(local_vectors[...,0]**2 + local_vectors[...,1]**2 + 1e-8)
        f1 = local_vectors[...,0]/rxy
        f2 = local_vectors[...,1]/rxy
        rxyz = torch.norm(local_vectors, dim=-1)+1e-8
        f3 = rxy/rxyz
        f4 = local_vectors[...,2]/rxyz

        f = torch.cat([f1[...,None],f2[...,None],f3[...,None],f4[...,None]],-1)
        return f



    def forward(self, Y, Y_m, Y_t, X, mask, R_idx, chain_labels, X_m, mask_residues, use_sc=True):
        """
        Y - [B, L, M, 3]
        Y_m, Y_t - [B, L, M]
        """
        if self.training and self.augment_eps > 0:
            X = X + self.augment_eps * torch.randn_like(X)
            Y = Y + self.augment_eps * torch.randn_like(Y)
        
        Ca = X[:,:,self.CA_idx,:]
        N = X[:,:,self.N_idx,:]
        C = X[:,:,self.C_idx,:]
        O = X[:,:,self.O_idx,:]

        b = Ca - N
        c = C - Ca
        a = torch.cross(b, c, dim=-1)
        Cb = -0.58273431*a + 0.56802827*b - 0.54067466*c + Ca #shift from CA
 
        D_neighbors, E_idx = self._dist(Ca, mask)

        RBF_all = []
        RBF_all.append(self._rbf(D_neighbors)) #Ca-Ca
        RBF_all.append(self._get_rbf(N, N, E_idx)) #N-N
        RBF_all.append(self._get_rbf(C, C, E_idx)) #C-C
        RBF_all.append(self._get_rbf(O, O, E_idx)) #O-O
        RBF_all.append(self._get_rbf(Cb, Cb, E_idx)) #Cb-Cb
        RBF_all.append(self._get_rbf(Ca, N, E_idx)) #Ca-N
        RBF_all.append(self._get_rbf(Ca, C, E_idx)) #Ca-C
        RBF_all.append(self._get_rbf(Ca, O, E_idx)) #Ca-O
        RBF_all.append(self._get_rbf(Ca, Cb, E_idx)) #Ca-Cb
        RBF_all.append(self._get_rbf(N, C, E_idx)) #N-C
        RBF_all.append(self._get_rbf(N, O, E_idx)) #N-O
        RBF_all.append(self._get_rbf(N, Cb, E_idx)) #N-Cb
        RBF_all.append(self._get_rbf(Cb, C, E_idx)) #Cb-C
        RBF_all.append(self._get_rbf(Cb, O, E_idx)) #Cb-O
        RBF_all.append(self._get_rbf(O, C, E_idx)) #O-C
        RBF_all.append(self._get_rbf(N, Ca, E_idx)) #N-Ca
        RBF_all.append(self._get_rbf(C, Ca, E_idx)) #C-Ca
        RBF_all.append(self._get_rbf(O, Ca, E_idx)) #O-Ca
        RBF_all.append(self._get_rbf(Cb, Ca, E_idx)) #Cb-Ca
        RBF_all.append(self._get_rbf(C, N, E_idx)) #C-N
        RBF_all.append(self._get_rbf(O, N, E_idx)) #O-N
        RBF_all.append(self._get_rbf(Cb, N, E_idx)) #Cb-N
        RBF_all.append(self._get_rbf(C, Cb, E_idx)) #C-Cb
        RBF_all.append(self._get_rbf(O, Cb, E_idx)) #O-Cb
        RBF_all.append(self._get_rbf(C, O, E_idx)) #C-O
        RBF_all = torch.cat(tuple(RBF_all), dim=-1)

        offset = R_idx[:,:,None]-R_idx[:,None,:]
        offset = gather_edges(offset[:,:,:,None], E_idx)[:,:,:,0] #[B, L, K]

        d_chains = ((chain_labels[:, :, None] - chain_labels[:,None,:])==0).long() #find self vs non-self interaction
        E_chains = gather_edges(d_chains[:,:,:,None], E_idx)[:,:,:,0]
        E_positional = self.embeddings(offset.long(), E_chains)
        E = torch.cat((E_positional, RBF_all), -1)
        E = self.edge_embedding(E)
        E = self.norm_edges(E)


        #protein graph node features
        #-----------------
        if use_sc:
            E_idx_sub = E_idx[:,:,:16] #[B, L, 15] 
            X_m = X_m*(1-mask_residues[:,:,None])
            R_m = gather_nodes(X_m[:,:,5:], E_idx_sub)

            X_sidechain = X[:,:,5:,:].view(X.shape[0], X.shape[1], -1)
            R = gather_nodes(X_sidechain, E_idx_sub).view(X.shape[0], X.shape[1], E_idx_sub.shape[2], -1, 3)
            R_t = self.side_chain_atom_types[None,None,None,:].repeat(X.shape[0], X.shape[1], E_idx_sub.shape[2], 1)

            #Side chain atom context 
            R = R.view(X.shape[0], X.shape[1], -1, 3) #coordinates
            R_m = R_m.view(X.shape[0], X.shape[1], -1) #mask
            R_t = R_t.view(X.shape[0], X.shape[1], -1) #atom types

            Y = torch.concat((R, Y), 2) #[B, L, atoms, 3]
            Y_m = torch.concat((R_m, Y_m), 2) #[B, L, atoms]
            Y_t = torch.concat((R_t, Y_t), 2) #[B, L, atoms]

            Cb_Y_distances = torch.sum((Cb[:,:,None,:] - Y)**2,-1)
            mask_Y = mask[:,:,None]*Y_m
            Cb_Y_distances_adjusted = Cb_Y_distances*mask_Y+(1. - mask_Y)*10000.0
            D_Y, E_idx_Y = torch.topk(Cb_Y_distances_adjusted, self.atom_context_num, dim=-1, largest=False)

            Y = torch.gather(Y, 2, E_idx_Y[:,:,:,None].repeat(1,1,1,3))
            Y_t = torch.gather(Y_t, 2, E_idx_Y)
            Y_m = torch.gather(Y_m, 2, E_idx_Y)

        Y_t = Y_t.long()
        Y_t_g = self.periodic_table_features[1][Y_t] #group; 19 categories including 0
        Y_t_p = self.periodic_table_features[2][Y_t] #period; 8 categories including 0

        Y_t_g_1hot_ = torch.nn.functional.one_hot(Y_t_g, 19) #[B, L, M, 19]
        Y_t_p_1hot_ = torch.nn.functional.one_hot(Y_t_p, 8) #[B, L, M, 8]
        Y_t_1hot_ = torch.nn.functional.one_hot(Y_t, 120) #[B, L, M, 120]

        Y_t_1hot_ = torch.cat([Y_t_1hot_, Y_t_g_1hot_, Y_t_p_1hot_], -1) #[B, L, M, 147]
        Y_t_1hot = self.type_linear(Y_t_1hot_.float())

        D_N_Y = self._rbf(torch.sqrt(torch.sum((N[:,:,None,:] - Y)**2,-1) + 1e-6)) #[B, L, M, num_bins]
        D_Ca_Y = self._rbf(torch.sqrt(torch.sum((Ca[:,:,None,:] - Y)**2,-1) + 1e-6))
        D_C_Y = self._rbf(torch.sqrt(torch.sum((C[:,:,None,:] - Y)**2,-1) + 1e-6))
        D_O_Y = self._rbf(torch.sqrt(torch.sum((O[:,:,None,:] - Y)**2,-1) + 1e-6))
        D_Cb_Y = self._rbf(torch.sqrt(torch.sum((Cb[:,:,None,:] - Y)**2,-1) + 1e-6))


        f_angles = self._make_angle_features(N, Ca, C, Y) #[B, L, M, 4]


        D_all = torch.cat((D_N_Y, D_Ca_Y, D_C_Y, D_O_Y, D_Cb_Y, Y_t_1hot, f_angles), dim=-1) #[B,L,M,5*num_bins+5+4]
        V = self.node_project_down(D_all) #[B, L, M, node_features] 
        V = self.norm_nodes(V)
        #-----------------

        #atom graph edge features
        Y_edges = self._rbf(torch.sqrt(torch.sum((Y[:,:,:,None,:] - Y[:,:,None,:,:])**2,-1) + 1e-6)) #[B, L, M, M, num_bins]


        Y_edges =  self.y_edges(Y_edges)
        Y_nodes = self.y_nodes(Y_t_1hot_.float())

        Y_edges = self.norm_y_edges(Y_edges)
        Y_nodes = self.norm_y_nodes(Y_nodes)


        return V, E, E_idx, Y_nodes, Y_edges, Y_m



class ProteinMPNN(nn.Module):
    def __init__(self, num_letters=21, node_features=128, edge_features=128,
        hidden_dim=128, num_encoder_layers=3, num_decoder_layers=3,
        vocab=21, k_neighbors=32, augment_eps=0.1, dropout=0.1, atom37_order=False, device=None, atom_context_num=32):
        super(ProteinMPNN, self).__init__()

        # Hyperparameters
        self.node_features = node_features
        self.edge_features = edge_features
        self.hidden_dim = hidden_dim

        self.features = ProteinFeatures(node_features, edge_features, top_k=k_neighbors, augment_eps=augment_eps, atom37_order=atom37_order, device=device, atom_context_num=atom_context_num)

        self.W_e = nn.Linear(edge_features, hidden_dim, bias=True)
        self.W_s = nn.Embedding(vocab, hidden_dim)
        self.W_v = nn.Linear(node_features, hidden_dim, bias=True)
        self.W_c = nn.Linear(hidden_dim, hidden_dim, bias=True)


        self.W_nodes_y = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.W_edges_y = nn.Linear(hidden_dim, hidden_dim, bias=True)


        self.V_C = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.V_C_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)





        # Encoder layers
        self.encoder_layers = nn.ModuleList([
            EncLayer(hidden_dim, hidden_dim*2, dropout=dropout)
            for _ in range(num_encoder_layers)
        ])

        # Decoder layers
        self.decoder_layers = nn.ModuleList([
            DecLayer(hidden_dim, hidden_dim*3, dropout=dropout)
            for _ in range(num_decoder_layers)
        ])


        self.context_encoder_layers = nn.ModuleList([
            DecLayer(hidden_dim, hidden_dim*2, dropout=dropout)
            for _ in range(2)
        ])

        self.y_context_encoder_layers = nn.ModuleList([
            DecLayerJ(hidden_dim, hidden_dim, dropout=dropout)
            for _ in range(2)
        ])



        self.W_out = nn.Linear(hidden_dim, num_letters, bias=True)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, feature_dict):
        X = feature_dict["X"] 
        S = feature_dict["S"]
        R_idx = feature_dict["R_idx"]
        mask = feature_dict["mask"]
        chain_labels = feature_dict["chain_labels"]
        Y = feature_dict["Y"]
        Y_m = feature_dict["Y_m"]
        Y_t = feature_dict["Y_t"]
        X_m = feature_dict["X_m"]
        mask_residues = feature_dict["mask_residues"]
       
        device=X.device
        # Prepare node and edge embeddings
        V, E, E_idx, Y_nodes, Y_edges, Y_m = self.features(Y, Y_m, Y_t, X, mask, R_idx, chain_labels, X_m, mask_residues, use_sc=feature_dict["use_sc"])
        h_V = torch.zeros((E.shape[0], E.shape[1], E.shape[-1]), device=E.device)
        h_E = self.W_e(E)
        h_E_context = self.W_v(V)
        

        # Encoder is unmasked self-attention
        mask_attend = gather_nodes(mask.unsqueeze(-1),  E_idx).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend
        for layer in self.encoder_layers:
            h_V, h_E = torch.utils.checkpoint.checkpoint(layer, h_V, h_E, E_idx, mask, mask_attend)


        h_V_C = self.W_c(h_V)
        Y_m_edges = Y_m[:,:,:,None]*Y_m[:,:,None,:]
        Y_nodes = self.W_nodes_y(Y_nodes)
        Y_edges = self.W_edges_y(Y_edges)
        for i in range(len(self.context_encoder_layers)):
            Y_nodes = torch.utils.checkpoint.checkpoint(self.y_context_encoder_layers[i],Y_nodes, Y_edges, Y_m, Y_m_edges)
            h_E_context_cat = torch.cat([h_E_context, Y_nodes], -1)
            h_V_C = torch.utils.checkpoint.checkpoint(self.context_encoder_layers[i],h_V_C, h_E_context_cat, mask, Y_m)

        h_V_C = self.V_C(h_V_C)
        h_V = h_V + self.V_C_norm(self.dropout(h_V_C))

        # Concatenate sequence embeddings for autoregressive decoder
        h_S = self.W_s(S)
        h_ES = cat_neighbors_nodes(h_S, h_E, E_idx)

        # Build encoder embeddings
        h_EX_encoder = cat_neighbors_nodes(torch.zeros_like(h_S), h_E, E_idx)
        h_EXV_encoder = cat_neighbors_nodes(h_V, h_EX_encoder, E_idx)


        chain_M = mask #update chain_M to include missing regions
        decoding_order = torch.argsort((chain_M+0.0001)*(torch.abs(torch.randn(chain_M.shape, device=device)))) #[numbers will be smaller for places where chain_M = 0.0 and higher for places where chain_M = 1.0]
        mask_size = E_idx.shape[1]
        permutation_matrix_reverse = torch.nn.functional.one_hot(decoding_order, num_classes=mask_size).float()
        order_mask_backward = torch.einsum('ij, biq, bjp->bqp',(1-torch.triu(torch.ones(mask_size,mask_size, device=device))), permutation_matrix_reverse, permutation_matrix_reverse)
        mask_attend = torch.gather(order_mask_backward, 2, E_idx).unsqueeze(-1)
        mask_1D = mask.view([mask.size(0), mask.size(1), 1, 1])
        mask_bw = mask_1D * mask_attend
        mask_fw = mask_1D * (1. - mask_attend)

        h_EXV_encoder_fw = mask_fw * h_EXV_encoder
        for layer in self.decoder_layers:
            h_ESV = cat_neighbors_nodes(h_V, h_ES, E_idx)
            h_ESV = mask_bw * h_ESV + h_EXV_encoder_fw
            h_V = torch.utils.checkpoint.checkpoint(layer, h_V, h_ESV, mask)

        logits = self.W_out(h_V)
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        return log_probs



class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, model_size, factor, warmup, optimizer, step):
        self.optimizer = optimizer
        self._step = step
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    @property
    def param_groups(self):
        """Return param_groups."""
        return self.optimizer.param_groups

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5)))

    def zero_grad(self):
        self.optimizer.zero_grad()

def get_std_opt(parameters, d_model, step):
    return NoamOpt(
        d_model, 2, 4000, torch.optim.Adam(parameters, lr=0, betas=(0.9, 0.98), eps=1e-9), step
    )
