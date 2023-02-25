import torch
from torch_geometric.nn import GCNConv, Set2Set, NNConv, GATConv, GINConv
from torch_geometric.nn.models import AttentiveFP
# from torch_geometric.nn import AttentiveFP
from torch.nn import functional as F, Sequential
from torch.nn import Sequential, Linear, ReLU, GRU
try:
    from graph_layer import DMPNNLayer1, DMPNNLayer2
except:
    from web_core.graph_layer import DMPNNLayer1, DMPNNLayer2


class GCN_net(torch.nn.Module):
    def __init__(self, mol_in_dim=25, out_dim=1, dim=64):
        super(GCN_net, self).__init__()
        self.lin0 = torch.nn.Linear(mol_in_dim, dim)
        self.conv = GCNConv(dim, dim)
        self.set2set = Set2Set(dim, processing_steps=3)
        self.lin1 = torch.nn.Linear(2 * dim, dim)
        self.lin2 = torch.nn.Linear(dim, out_dim)

    def forward(self, data):
        out = F.relu(self.lin0(data.x))
        out = F.relu(self.conv(out, data.edge_index))
        out = self.set2set(out, data.batch)
        out = F.relu(self.lin1(out))
        out = self.lin2(out)
        return out.view(-1)


class MPNN_net(torch.nn.Module):
    def __init__(self, mol_in_node_dim=25, mol_in_edge_dim=4, out_dim=1, dim=64, mess_nn_dim=128):
        super(MPNN_net, self).__init__()
        self.lin0 = torch.nn.Linear(mol_in_node_dim, dim)

        nn = Sequential(Linear(mol_in_edge_dim, mess_nn_dim), ReLU(), Linear(mess_nn_dim, dim * dim))
        self.conv = NNConv(dim, dim, nn, aggr='mean')
        self.gru = GRU(dim, dim)

        self.set2set = Set2Set(dim, processing_steps=3)
        self.lin1 = torch.nn.Linear(2 * dim, dim)
        self.lin2 = torch.nn.Linear(dim, out_dim)

    def forward(self, data):
        out = F.relu(self.lin0(data.x))
        h = out.unsqueeze(0)

        for i in range(3):
            m = F.relu(self.conv(out, data.edge_index, data.edge_attr))
            out, h = self.gru(m.unsqueeze(0), h)
            out = out.squeeze(0)

        out = self.set2set(out, data.batch)
        out = F.relu(self.lin1(out))
        out = self.lin2(out)
        return out.view(-1)


class GAT_net(torch.nn.Module):
    def __init__(self, mol_in_dim=25, gat_heads=8, out_dim=1, dim=64):
        super(GAT_net, self).__init__()
        self.lin0 = torch.nn.Linear(mol_in_dim, dim)
        self.conv = GATConv(in_channels=dim, out_channels=dim, heads=gat_heads)
        self.head_nn = torch.nn.Linear(gat_heads * dim, dim)
        self.set2set = Set2Set(dim, processing_steps=3)
        self.lin1 = torch.nn.Linear(2 * dim, dim)
        self.lin2 = torch.nn.Linear(dim, out_dim)

    def forward(self, data):
        out = F.relu(self.lin0(data.x))
        out = F.relu(self.conv(out, data.edge_index))
        out = F.relu(self.head_nn(out))
        out = self.set2set(out, data.batch)
        out = F.relu(self.lin1(out))
        out = self.lin2(out)
        return out.view(-1)


class GIN_net(torch.nn.Module):
    def __init__(self, mol_in_dim=25, mol_edge_in_dim=4, out_dim=1, dim=64):
        super(GIN_net, self).__init__()
        self.lin0 = torch.nn.Linear(mol_in_dim, dim)
        lin = torch.nn.Linear(dim, dim)
        self.conv = GINConv(nn=lin)
        self.set2set = Set2Set(dim, processing_steps=3)
        self.lin1 = torch.nn.Linear(2 * dim, dim)
        self.lin2 = torch.nn.Linear(dim, out_dim)

    def forward(self, data):
        out = F.relu(self.lin0(data.x))
        out = F.relu(self.conv(out, data.edge_index))
        out = self.set2set(out, data.batch)
        out = F.relu(self.lin1(out))
        out = self.lin2(out)
        return out.view(-1)


class AttentiveFP_net(torch.nn.Module):
    def __init__(self, mol_in_dim=25, mol_in_edge_dim=4, out_dim=1, dim=64, num_layers=5):
        super(AttentiveFP_net, self).__init__()
        self.attentivefp = AttentiveFP(in_channels=mol_in_dim, hidden_channels=dim, out_channels=out_dim,
                                       edge_dim=mol_in_edge_dim, num_layers=num_layers, num_timesteps=3, dropout=0.5)

    def forward(self, data):
        out = self.attentivefp(data.x, data.edge_index, data.edge_attr, data.batch)
        return out.view(-1)


class DMPNN(torch.nn.Module):
    def __init__(self, mol_in_dim=25, out_dim=1, dim=64, f_ab_size=100):
        super(DMPNN, self).__init__()
        self.lin0 = torch.nn.Linear(mol_in_dim, dim)
        nn = torch.nn.Linear(f_ab_size, dim)
        self.conv = DMPNNLayer1(dim, dim, dim, nn, dropout=0.3)
        self.set2set = Set2Set(dim, processing_steps=3)
        self.lin1 = torch.nn.Linear(2 * dim, dim)
        self.lin2 = torch.nn.Linear(dim, out_dim)

    def forward(self, data):
        out = F.relu(self.lin0(data.x))
        out = F.relu(self.conv(out, data.edge_index, data.edge_attr))
        out = self.set2set(out, data.batch)
        out = F.relu(self.lin1(out))
        out = self.lin2(out)
        return out.view(-1)


class DMPNN_Change(torch.nn.Module):
    def __init__(self, mol_in_dim=25, out_dim=1, dim=256, mol_in_edge_dim=14, use_gru=False,
                 massage_depth=1,
                 dropout_rate=0.3):
        super(DMPNN_Change, self).__init__()
        self.dropout_rate = dropout_rate
        self.use_gru = use_gru
        self.massage_depth = massage_depth
        self.lin0 = torch.nn.Linear(mol_in_dim, dim)
        self.conv = DMPNNLayer2(dim, dim, dim, mol_in_edge_dim, dropout=0.3)
        if self.use_gru:
            self.gru = GRU(dim, dim)
        self.set2set = Set2Set(dim, processing_steps=3)
        self.lin1 = torch.nn.Linear(2 * dim, dim)
        self.lin2 = torch.nn.Linear(dim, out_dim)

    def forward(self, data):
        out = F.relu(self.lin0(data.x))
        if self.use_gru:
            h = out.unsqueeze(0)
        for i in range(self.massage_depth):

            if self.use_gru:
                m = F.relu(self.conv(out, data.edge_index, data.edge_attr))
                out, h = self.gru(m.unsqueeze(0), h)
                out = out.squeeze(0)
            else:
                out = F.relu(self.conv(out, data.edge_index, data.edge_attr))

        out = self.set2set(out, data.batch)
        out = F.dropout(F.relu(self.lin1(out)), p=0.3)
        out = self.lin2(out)
        return out.view(-1)

