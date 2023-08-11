import math

import pandas as pd
from typing import Optional, Callable, List

import sys
import os
import os.path as osp

from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from torch_scatter import scatter
from torch_geometric.data import (InMemoryDataset, Data)

try:
    from utils import one_of_k_encoding, bond_features
except:
    from web_core.utils import one_of_k_encoding, bond_features

atom_types = {'H': 0, 'B': 1, 'C': 2, 'N': 3, 'O': 4, 'F': 5, 'Na': 6, 'Si': 7, 'P': 8, 'S': 9, 'Cl': 10, 'K': 11,
              'Ca': 12, 'Zn': 13, 'Se': 14, 'Br': 15, 'Ru': 16, 'I': 17, 'Pt': 18}


def mol2fp(mol, fp_dim):
    # mol = Chem.MolFromSmiles(smi)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=int(fp_dim), useChirality=True)
    onbits = list(fp.GetOnBits())
    arr = np.zeros(fp.GetNumBits())
    arr[onbits] = 1
    return arr


def ic50_to_pic50(ic50_value):
    pic50_value = (-1) * (math.log10(math.pow(10, -6) * ic50_value))  # 传入的为um值
    return pic50_value


class TargetDataset(InMemoryDataset):

    def __init__(self, root: str,
                 target_data_space_name: str,
                 target_name: str,
                 target_columns_name: str,
                 to_pic50=False,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None):
        self.target_columns_name = target_columns_name
        self.target_name = target_name
        self.to_pic50 = to_pic50
        self.target_data_space = os.path.join(root, 'raw', target_data_space_name)
        self.target_data_split = os.path.join(self.target_data_space, target_name)
        try:
            self.atom_types = torch.load(os.path.join(self.target_data_space, f'{self.target_name}_atom_types.pkle'))
        except:
            self.get_atom_types()
            self.atom_types = torch.load(os.path.join(self.target_data_space, f'{self.target_name}_atom_types.pkle'))
        self.dataset_split_index = torch.load(
            os.path.join(self.target_data_split, f'{self.target_name}_split_index.pkl'))
        super(TargetDataset, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.split_save_path = self.processed_paths[0].replace('.pt', '_split.pt')
        self.train, self.val, self.test = self.split()
        self.self_num_node_features = self.data.x.shape[1]
        self.self_num_edge_features = self.data.edge_attr.shape[1]
        dataset_dic = {
            'atom_types': self.atom_types,
            'self_num_node_features': self.self_num_node_features,
            'self_num_edge_features': self.self_num_edge_features,
            'y_max': self.data.y.max(),
            'y_min': self.data.y.min(),
            'y_mean': self.data.y.mean(),
            'y_std': self.data.y.std(),
        }

        uom_flag = 'IC50' if not self.to_pic50 else 'PIC50'
        dataset_dic_path = os.path.join(self.target_data_space, f'{self.target_name}_{uom_flag}_dataset_parm.pkle')
        if not os.path.exists(dataset_dic_path):
            print(f'Saving dataset infomation to {dataset_dic_path}')
            torch.save(dataset_dic, dataset_dic_path)

    def mean(self, target: int) -> float:
        y = torch.cat([self.get(i).y for i in range(len(self))], dim=0)
        return float(y[:, target].mean())

    def std(self, target: int) -> float:
        y = torch.cat([self.get(i).y for i in range(len(self))], dim=0)
        return float(y[:, target].std())

    @property
    def raw_file_names(self) -> List[str]:
        import rdkit  # noqa
        return [self.target_name]

    @property
    def processed_file_names(self) -> str:
        if self.to_pic50:
            return f'{self.target_name}_processed_PIC50.pt'
        else:
            return f'{self.target_name}_processed_IC50.pt'

    def process(self):
        import pandas as pd
        from rdkit import Chem
        from rdkit.Chem.rdchem import HybridizationType as HT
        from rdkit.Chem.rdchem import BondType as BT
        from rdkit import RDLogger
        RDLogger.DisableLog('rdApp.*')

        all_atoms_degrees = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        all_atoms_valence = [0, 1, 2, 3, 4, 5, 6]
        all_atoms_charges_to_int = {-1: 0, -2: 1, 1: 2, 2: 3, 0: 4, 3: 5, 4: 6, 5: 7}
        all_atoms_charges = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        all_atoms_num_hs = [0, 1, 2, 3, 4, 5]

        def get_charge_int(charge):
            if charge not in all_atoms_charges_to_int:
                return 8
            else:
                return all_atoms_charges_to_int[charge]

        # read raw dataset
        dataset = pd.DataFrame()
        for dataset_flag in ['train', 'val', 'test']:
            dataset = dataset.append(
                pd.read_csv(os.path.join(self.target_data_split, f'{dataset_flag}.csv'), encoding='GBK'))
        smiles_list = dataset['smiles'].tolist()
        target = [float(x) for x in dataset[self.target_columns_name].tolist()]
        target = torch.tensor(target, dtype=torch.float).view(-1)

        data_list = []
        for i, smi in enumerate(tqdm(smiles_list)):
            if pd.isna(smi):
                continue
            mol = Chem.MolFromSmiles(smi)
            fp = mol2fp(mol, 2048)
            fp = torch.tensor(fp, dtype=torch.float).view(1, -1)
            if mol is None:
                continue
            N = mol.GetNumAtoms()  # 获取分子中的原子数目

            type_idx = []
            atomic_number = []
            aromatic = []
            num_hs = []
            hybridization = []
            atom_charges = []
            atom_valence = []
            atom_degrees = []
            for atom in mol.GetAtoms():
                type_idx.append(self.atom_types[atom.GetSymbol()])  # atom.GetSymbol() --> 获得元素符号
                atomic_number.append(atom.GetAtomicNum())  # atom.GetAtomicNum --> 获得元素序号
                aromatic.append(1 if atom.GetIsAromatic() else 0)
                hybridization.append(atom.GetHybridization())
                num_hs.append(atom.GetTotalNumHs())
                atom_charges.append(get_charge_int(atom.GetFormalCharge()))
                atom_valence.append(atom.GetExplicitValence())
                atom_degrees.append(atom.GetTotalDegree())

            row, col, edge_type = [], [], []
            for bond in mol.GetBonds():
                start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                row += [start, end]
                col += [end, start]
                edge_type.append(bond_features(bond))
                edge_type.append(bond_features(bond))
            edge_index = torch.tensor([row, col], dtype=torch.long)
            edge_attr = torch.FloatTensor(edge_type)

            perm = (edge_index[0] * N + edge_index[1]).argsort()
            edge_index = edge_index[:, perm]
            edge_attr = edge_attr[perm]

            x1 = F.one_hot(torch.tensor(type_idx), num_classes=len(self.atom_types))
            x_num_hs = [one_of_k_encoding(nh, all_atoms_num_hs) for nh in num_hs]
            x_hybridization = [one_of_k_encoding(h, [HT.SP, HT.SP2, HT.SP3, HT.SP3D, HT.SP3D2]) for h in hybridization]
            x_atom_charges = [one_of_k_encoding(x, all_atoms_charges) for x in atom_charges]
            x_atom_valence = [one_of_k_encoding(x, all_atoms_valence) for x in atom_charges]
            x_atom_degrees = [one_of_k_encoding(x, all_atoms_degrees) for x in atom_degrees]
            x2 = torch.tensor([atomic_number,
                               aromatic],
                              dtype=torch.float).t().contiguous()
            x3 = torch.cat([torch.FloatTensor(x_hybridization),
                            torch.FloatTensor(x_num_hs),
                            torch.FloatTensor(x_atom_charges),
                            torch.FloatTensor(x_atom_valence),
                            torch.FloatTensor(x_atom_degrees)], dim=-1)
            x = torch.cat([x1.to(torch.float), x2, x3], dim=-1)
            if self.to_pic50:
                y = ic50_to_pic50(target[i].unsqueeze(0))
            else:
                y = target[i].unsqueeze(0)

            data = Data(x=x, edge_index=edge_index,
                        edge_attr=edge_attr, y=y, idx=i, smiles=smi, fp=fp)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)

        torch.save(self.collate(data_list), self.processed_paths[0])

    def split(self):
        if not os.path.exists(self.split_save_path):
            print('Split dataset with index file: {}'.format(self.dataset_split_index))
            data_split_list = []
            for head, end in self.dataset_split_index:
                data_split_list.append(self[head:end])

            # train, val, test = data_split_list
            torch.save(data_split_list, self.split_save_path)
            train, val, test = torch.load(self.split_save_path)
        else:
            print('Read splited dataset from {}'.format(self.split_save_path))
            train, val, test = torch.load(self.split_save_path)
        return train, val, test

    def get_atom_types(self):
        dataset_split_index = []
        dataset_df = pd.DataFrame()
        for data_flag in ['train', 'val', 'test']:
            head_index = len(dataset_df)
            current_df = pd.read_csv(os.path.join(self.target_data_split, f'{data_flag}.csv'))
            dataset_df = dataset_df.append(current_df)
            dataset_split_index.append((head_index, head_index + len(current_df)))

        smiles_list = dataset_df['smiles'].tolist()
        atom_types = {}
        for smi in smiles_list:
            if pd.isna(smi):
                continue
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                for atom in mol.GetAtoms():
                    if atom not in atom_types:
                        atom_types[atom.GetSymbol()] = atom.GetAtomicNum()
        types_list = list(atom_types.items())
        types_list.sort(key=lambda x: x[1])
        # print(types_list)
        atom_types = {types_list[i][0]: i for i in range(len(types_list))}
        # print(atom_types)

        torch.save(dataset_split_index, os.path.join(self.target_data_split, f'{self.target_name}_split_index.pkl'))
        torch.save(atom_types, os.path.join(self.target_data_space, f'{self.target_name}_atom_types.pkle'))


if __name__ == '__main__':
    dataset_raw_path = './self_dataset_ic50/raw/'
    target_name = 'EGFR_str_act'
    # target_data_space = os.path.join(dataset_raw_path, target_name)
    # target_data_split = os.path.join(target_data_space, target_name)
    #
    # dataset_split_index = []
    # dataset_df = pd.DataFrame()
    # for data_flag in ['train', 'val', 'test']:
    #     head_index = len(dataset_df)
    #     current_df = pd.read_csv(os.path.join(target_data_split, f'{data_flag}.csv'))
    #     dataset_df = dataset_df.append(current_df)
    #     dataset_split_index.append((head_index, head_index + len(current_df)))
    #
    # smiles_list = dataset_df['smiles'].tolist()
    # types = {}
    # for smi in smiles_list:
    #     if pd.isna(smi):
    #         continue
    #     mol = Chem.MolFromSmiles(smi)
    #     if mol is not None:
    #         for atom in mol.GetAtoms():
    #             if atom not in types:
    #                 types[atom.GetSymbol()] = atom.GetAtomicNum()
    # types_list = list(types.items())
    # types_list.sort(key=lambda x: x[1])
    # print(types_list)
    # types = {types_list[i][0]: i for i in range(len(types_list))}
    # print(types)
    #
    # torch.save(dataset_split_index, os.path.join(target_data_split, f'{target_name}_split_index.pkl'))
    # torch.save(types, os.path.join(target_data_space, f'{target_name}_atom_types.pkl'))

    dataset = TargetDataset(root='./self_dataset_ic50', target_name=target_name, to_pic50=False,
                            target_columns_name='new_activity_values(uM)')
