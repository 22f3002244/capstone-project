import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GCNConv, GATConv, SAGEConv,
    BatchNorm, LayerNorm,
    MessagePassing,
)
from torch_geometric.utils import add_self_loops, degree
from torch import Tensor
from typing import List, Optional


class EGraphSAGEConv(MessagePassing):

    def __init__(self, in_channels: int, out_channels: int, edge_feat_dim: int):
        super().__init__(aggr="mean")
        self.lin = nn.Linear(in_channels + edge_feat_dim, out_channels, bias=True)
        nn.init.xavier_uniform_(self.lin.weight)

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        agg = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        out = torch.cat([x, agg], dim=-1)
        return F.relu(self.lin(out))

    def message(self, edge_attr: Tensor) -> Tensor:
        return edge_attr


class EGraphSAGE(nn.Module):

    def __init__(
        self,
        edge_feat_dim: int,
        hidden_dims: List[int] = [128, 64],
        output_dim: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        in_dim = edge_feat_dim

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        dims = [in_dim] + hidden_dims
        for i in range(len(dims) - 1):
            self.convs.append(EGraphSAGEConv(dims[i], dims[i + 1], edge_feat_dim=in_dim))
            self.norms.append(LayerNorm(dims[i + 1]))

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dims[-1] * 2, hidden_dims[-1]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[-1], output_dim),
        )
        self.dropout      = dropout
        self.hidden_dims  = hidden_dims
        self.edge_feat_dim = edge_feat_dim

    def get_node_embeddings(self, x: Tensor, edge_index: Tensor,
                             edge_attr: Tensor) -> Tensor:
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index, edge_attr)
            x = norm(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def forward(self, x: Tensor, edge_index: Tensor,
                edge_attr: Optional[Tensor] = None) -> Tensor:
        if edge_attr is None:
            edge_attr = torch.zeros(edge_index.shape[1], self.edge_feat_dim,
                                    device=x.device)
        z = self.get_node_embeddings(x, edge_index, edge_attr)
        src, dst   = edge_index[0], edge_index[1]
        edge_emb   = torch.cat([z[src], z[dst]], dim=-1)
        return self.classifier(edge_emb)


class GCN(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int] = [128, 64],
                 output_dim: int = 2, dropout: float = 0.3):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns   = nn.ModuleList()
        dims = [input_dim] + hidden_dims
        for i in range(len(dims) - 1):
            self.convs.append(GCNConv(dims[i], dims[i + 1]))
            self.bns.append(BatchNorm(dims[i + 1]))
        self.fc      = nn.Linear(hidden_dims[-1], output_dim)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_attr=None):
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.fc(x)


class GAT(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int] = [128, 64],
                 output_dim: int = 2, heads: int = 4, dropout: float = 0.3):
        super().__init__()
        self.convs   = nn.ModuleList()
        self.dropout = dropout

        self.convs.append(GATConv(input_dim, hidden_dims[0],
                                  heads=heads, dropout=dropout))
        for i in range(len(hidden_dims) - 1):
            self.convs.append(GATConv(hidden_dims[i] * heads, hidden_dims[i + 1],
                                      heads=heads, dropout=dropout))
        self.convs.append(GATConv(hidden_dims[-1] * heads, output_dim,
                                  heads=1, concat=False, dropout=dropout))

    def forward(self, x, edge_index, edge_attr=None):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.convs[-1](x, edge_index)


class GraphSAGE(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int] = [128, 64],
                 output_dim: int = 2, dropout: float = 0.3):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns   = nn.ModuleList()
        dims = [input_dim] + hidden_dims
        for i in range(len(dims) - 1):
            self.convs.append(SAGEConv(dims[i], dims[i + 1]))
            self.bns.append(BatchNorm(dims[i + 1]))
        self.fc      = nn.Linear(hidden_dims[-1], output_dim)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_attr=None):
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.fc(x)


class Hybrid(nn.Module):

    def __init__(self, input_dim: int, hidden_dims: List[int] = [128, 64],
                 output_dim: int = 2, heads: int = 4, dropout: float = 0.3):
        super().__init__()
        h = hidden_dims[-1]
        self.dropout = dropout

        self.gcn1 = GCNConv(input_dim,       hidden_dims[0])
        self.gcn2 = GCNConv(hidden_dims[0],  h)

        self.gat1 = GATConv(input_dim,           hidden_dims[0], heads=heads, dropout=dropout)
        self.gat2 = GATConv(hidden_dims[0]*heads, h,              heads=1, concat=False, dropout=dropout)

        self.sage1 = SAGEConv(input_dim,       hidden_dims[0])
        self.sage2 = SAGEConv(hidden_dims[0],  h)

        self.bn_gcn  = BatchNorm(h)
        self.bn_gat  = BatchNorm(h)
        self.bn_sage = BatchNorm(h)

        self.fc = nn.Sequential(
            nn.Linear(h * 3, h),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h, output_dim),
        )

    def forward(self, x, edge_index, edge_attr=None):
        d = self.dropout

        g = F.relu(self.gcn1(x, edge_index))
        g = F.dropout(g, d, self.training)
        g = self.bn_gcn(F.relu(self.gcn2(g, edge_index)))

        a = F.elu(self.gat1(x, edge_index))
        a = F.dropout(a, d, self.training)
        a = self.bn_gat(F.elu(self.gat2(a, edge_index)))

        s = F.relu(self.sage1(x, edge_index))
        s = F.dropout(s, d, self.training)
        s = self.bn_sage(F.relu(self.sage2(s, edge_index)))

        out = torch.cat([g, a, s], dim=-1)
        out = F.dropout(out, d, self.training)
        return self.fc(out)


def get_model(
    name: str,
    input_dim: int,
    hidden_dims: List[int] = [128, 64],
    output_dim: int = 2,
    dropout: float = 0.3,
    heads: int = 4,
    edge_feat_dim: Optional[int] = None,
) -> nn.Module:
    n = name.lower().replace("-", "").replace("_", "")

    if n == "egraphsage":
        dim = edge_feat_dim if edge_feat_dim is not None else input_dim
        return EGraphSAGE(dim, hidden_dims, output_dim, dropout)
    elif n == "gcn":
        return GCN(input_dim, hidden_dims, output_dim, dropout)
    elif n == "gat":
        return GAT(input_dim, hidden_dims, output_dim, heads, dropout)
    elif n in ("graphsage", "sage"):
        return GraphSAGE(input_dim, hidden_dims, output_dim, dropout)
    elif n == "hybrid":
        return Hybrid(input_dim, hidden_dims, output_dim, heads, dropout)
    else:
        raise ValueError(f"Unknown model '{name}'. Valid: EGraphSAGE | GCN | GAT | GraphSAGE | Hybrid")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
