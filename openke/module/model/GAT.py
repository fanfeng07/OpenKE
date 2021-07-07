import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphAttentionNetwork(nn.Module):
    def __init__(self, kg, in_dim, hid_dim, out_dim, num_heads):
        super(GraphAttentionNetwork, self).__init__()
        self.kg = kg
        self.layer1 = MultiHeadLayer(kg, in_dim, hid_dim, num_heads, concat=True)
        self.layer2 = MultiHeadLayer(kg, hid_dim*num_heads, out_dim, 1, concat=False)
	
    def forward(self, h):
        output = self.layer1(h)
        print('output1:\n', output.shape)
        output = self.layer2(output)
        # print('output2:\n', output.shape)
        output = F.elu(output)

        print('output2:\n', output.shape, output)
        # return F.log_softmax(F.elu(output1), dim=-1)
        return output



class MultiHeadLayer(nn.Module):
    def __init__(self, kg, in_dim, out_dim, num_heads, concat=True):
        super(MultiHeadLayer, self).__init__()

        self.heads = nn.ModuleList()
        for i in range(num_heads):
            self.heads.append(GraphAttentionLayer(in_dim, out_dim, kg))
        self.concat = concat

    def forward(self, h):
        # apply nonlinearity to first layer's output then concatenate them
        head_outs = [F.elu(attn_head(h)) for attn_head in self.heads]
        if self.concat:
            print('heads:\n', head_outs)
            # concat on the output feature dimension (dim=1)
            return torch.cat(head_outs, dim=1) # ? apply nonlinear F.elu
        else:
            # merge using average
            print('stack:\n',torch.stack(head_outs).shape)
            print('mean:\n', torch.mean(torch.stack(head_outs), dim=0))
            return torch.mean(torch.stack(head_outs),dim=0)



# Paper: Graph Attention Networks
# Source: https://docs.dgl.ai/en/0.4.x/tutorials/models/1_gnn/9_gat.html
class GraphAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, kg, dropout=0.6):
        super(GraphAttentionLayer, self).__init__()

        self.in_dim = in_dim # initial embedding
        self.out_dim = out_dim # output embedding
        self.kg = kg # networkx knowledge graph
        self.dropout = nn.Dropout(p=dropout)

        # define learnable params of a GAT layer
        # parameter W
        # equation (3) W*h: linear transformation
        self.fc = nn.Linear(in_dim, out_dim, bias=False) #learn xA(+bias)
        # parameter a
        # equation (1) a[Wh1 || Wh2]: attention coefficient
        self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False) #input: concatenation, output coefficient
        self.reset_parameters() 



    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')
        # or
        # nn.init.xavier_uniform_(self.W.data, gain=1.414)
        # gain = nn.init.calculate_gain('leaky_relu', 0.2)
        nn.init.xavier_normal_(self.fc.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_fc.weight, gain=gain)

 
    def self_attention(self, Wh):
        # initialize a matrix for a(Wh1 || Wh2)
        N = Wh.shape[0]
        # init (N*N, F_out*2) matrix
        W_ij = -9e15*torch.ones_like(torch.cat([Wh.repeat_interleave(N, dim=0),Wh.repeat(N, 1)], dim=1))
        print('init_W_ij:\n', W_ij.shape, W_ij)
        W_ij = W_ij.view(N, N, 2*self.out_dim)

        # update to concatenated vector
        for n in self.kg.nodes: # from node n
            for edge in self.kg.edges(n): # to all neighbors via (n, j)
                i, j = edge[0], edge[1]
                new_vec = torch.cat([Wh[i], Wh[j]], dim=-1)
                # print('new_vec: \n', new_vec.shape, new_vec)
                # print('W_ij[i]',W_ij[i])

                W_ij[i][j] = new_vec
                # W_ij[j][i] = torch.cat([Wh[j], Wh[i]], dim=-1)
        print('view_W_ij:\n', W_ij.shape, W_ij)
        return W_ij



    # inductive setting according to official implementation
    def forward(self, h):
        # Step 1: linear transformation W*h
        # h.shape: (N, in_dim), Wh.shape: (N, out_dim)
        h = self.dropout(h) #apply dropout to all input features 
        Wh = self.fc(h) # linear transformation
        print('Wh:\n',Wh.shape)

        # Step 2: edge attention 
        W_cat = self.self_attention(Wh) # attention
        print('W_cat:\n', W_cat.shape)
        W_cat = self.attn_fc(W_cat) # linear transformation to learn attention weights
        e = F.leaky_relu(W_cat, negative_slope=0.2) # for each j: (N, 2*hidden_dim) -> (N, out_dim), shared attentional mechnism, element-wise leakyRelu
        alpha = F.softmax(e, dim=1) # apply softmax among all neighbors j, which is dim=1
        print('alpha:\n', alpha.shape)
        alpha = self.dropout(alpha) #apply dropout to normalized attention coefficients
        h_out = torch.sum(alpha * Wh, dim=1) # shape: (N, )
        print('h_out:\n', h_out.shape)
        return h_out
        # return W_cat
        