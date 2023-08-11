from collections import OrderedDict
from logging import Logger
import math
import os
import random
from typing import List
import numpy as np
import torch
import torch.nn.functional as F
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem.rdchem import HybridizationType as HT
from torch_geometric.data import in_memory_dataset
from tqdm import tqdm
import pandas as pd
import rdkit.Chem.QED as QED
from rdkit.Chem import Descriptors
from chemprop.train import predict
from chemprop.data import MoleculeDataset, MoleculeDataLoader, MoleculeDatapoint
from chemprop.data.utils import get_data, filter_invalid_smiles
from chemprop.utils import load_args, load_checkpoint, load_scalers
try:
    import script.sascorer as sascorer
except:
    from web_core.script import sascorer as sascorer

RDLogger.DisableLog('rdApp.*')


def cal_ci(y, f):
    ind = np.argsort(y)
    y = y[ind]
    f = f[ind]
    i = len(y) - 1
    j = i - 1
    z = 0.0
    S = 0.0
    while i > 0:
        while j >= 0:
            if y[i] > y[j]:
                z = z + 1
                u = f[i] - f[j]
                if u > 0:
                    S = S + 1
                elif u == 0:
                    S = S + 0.5
            j = j - 1
        i = i - 1
        j = i - 1
    ci = S / z
    return ci


def cal_mae(y_true, y_pred):
    mae = (y_true - y_pred).abs().sum().item()
    return mae / y_true.size(0)


def cal_mse(y_true, y_pred):
    mse = torch.pow((y_true - y_pred), 2).sum().item()
    return mse / y_true.size(0)


def cal_r2_score(y_true, y_pred):
    numerator = torch.pow((y_true - y_pred), 2).sum(axis=0)
    denominator = torch.pow((y_true - y_true.mean(-1)), 2).sum(axis=0)

    nonzero_denominator = denominator != 0
    nonzero_numerator = numerator != 0
    valid_score = nonzero_denominator & nonzero_numerator
    output_scores = torch.ones(y_true.size(), dtype=torch.float)
    output_scores[valid_score] = 1 - (numerator[valid_score] /
                                      denominator[valid_score])
    output_scores[nonzero_numerator & ~nonzero_denominator] = 0.
    return output_scores.mean().item()


def regression_metrics(y_true, y_pred):
    mae = cal_mae(y_true, y_pred)
    # ci = cal_ci(y_true, y_pred)
    mse = cal_mse(y_true, y_pred)
    rmse = mse ** 0.5
    r2 = cal_r2_score(y_true, y_pred)
    d = {'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2}
    return d


def merge_results(results_list):
    # cis = []
    mse_list = []
    rmse_list = []
    mae_list = []
    r2_list = []
    for result in results_list:
        # cis.append(result['ci'])
        mse_list.append(result['mse'])
        rmse_list.append(result['rmse'])
        mae_list.append(result['mae'])
        r2_list.append(result['r2'])
    d_mean = {'mse': np.average(mse_list[-10:]), 'rmse': np.average(rmse_list[-10:]),
              'mae': np.average(mae_list[-10:]), 'r2': np.average(r2_list[-10:])}
    d = {'mse': mse_list, 'rmse': rmse_list,
         'mae': mae_list, 'r2': r2_list}
    return d_mean, d


def one_of_k_encoding(value, choices):
    """
    Creates a one-hot encoding with an extra category for uncommon values.

    :param value: The value for which the encoding should be one.
    :param choices: A list of possible values.
    :return: A one-hot encoding of the :code:`value` in a list of length :code:`len(choices) + 1`.
             If :code:`value` is not in :code:`choices`, then the final element in the encoding is 1.
    """
    encoding = [0] * (len(choices) + 1)
    index = choices.index(value) if value in choices else -1
    encoding[index] = 1

    return encoding


def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def bond_features(bond: Chem.rdchem.Bond):
    """
    Builds a feature vector for a bond.

    :param bond: An RDKit bond.
    :return: A list containing the bond features.
    """

    bt = bond.GetBondType()
    fbond = [
        0,  # bond is not None
        bt == Chem.rdchem.BondType.SINGLE,
        bt == Chem.rdchem.BondType.DOUBLE,
        bt == Chem.rdchem.BondType.TRIPLE,
        bt == Chem.rdchem.BondType.AROMATIC,
        (bond.GetIsConjugated() if bt is not None else 0),
        (bond.IsInRing() if bt is not None else 0)
    ]
    fbond += one_of_k_encoding(int(bond.GetStereo()), list(range(6)))
    return fbond


def mol2fp(mol, fp_dim):
    # mol = Chem.MolFromSmiles(smi)
    fp = AllChem.GetMorganFingerprintAsBitVect(
        mol, 2, nBits=int(fp_dim), useChirality=True)
    onbits = list(fp.GetOnBits())
    arr = np.zeros(fp.GetNumBits())
    arr[onbits] = 1
    return arr


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


def cooked_data_to_csv(dataset, save_fname):
    smiles = [data.smiles for data in tqdm(dataset)]
    y = [data.y.item() for data in tqdm(dataset)]
    df = pd.DataFrame({
        'smiles': smiles,
        'y': y
    })
    df.to_csv(save_fname, index=False)


def mol2graph(mol, types):
    # mol = Chem.MolFromSmiles(smi)

    if mol is None:
        return
    fp = mol2fp(mol, 2048)
    fp = torch.tensor(fp, dtype=torch.float).view(1, -1)
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
        type_idx.append(types[atom.GetSymbol()])  # atom.GetSymbol() --> 获得元素符号
        # atom.GetAtomicNum --> 获得元素序号
        atomic_number.append(atom.GetAtomicNum())
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

    x1 = F.one_hot(torch.tensor(type_idx), num_classes=len(types))
    x_num_hs = [one_of_k_encoding(nh, all_atoms_num_hs) for nh in num_hs]
    x_hybridization = [one_of_k_encoding(
        h, [HT.SP, HT.SP2, HT.SP3, HT.SP3D, HT.SP3D2]) for h in hybridization]
    x_atom_charges = [one_of_k_encoding(
        x, all_atoms_charges) for x in atom_charges]
    x_atom_valence = [one_of_k_encoding(
        x, all_atoms_valence) for x in atom_charges]
    x_atom_degrees = [one_of_k_encoding(
        x, all_atoms_degrees) for x in atom_degrees]
    x2 = torch.tensor([atomic_number,
                       aromatic],
                      dtype=torch.float).t().contiguous()
    x3 = torch.cat([torch.FloatTensor(x_hybridization),
                    torch.FloatTensor(x_num_hs),
                    torch.FloatTensor(x_atom_charges),
                    torch.FloatTensor(x_atom_valence),
                    torch.FloatTensor(x_atom_degrees)], dim=-1)
    x = torch.cat([x1.to(torch.float), x2, x3], dim=-1)
    return x, edge_index, edge_attr, fp


def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True  # 训练集变化不大时使训练加速


def get_fps_from_dataset(pyg_dataset, get_y=True):
    fps, y = [], []
    for data in tqdm(pyg_dataset):
        fps.append(data.fp)
        if get_y:
            y.append(data.y)

    fps = torch.cat(fps)
    if get_y:
        y = torch.cat(y)
        return fps.numpy(), y.numpy()
    else:
        return fps.numpy(), []


def canonicalize_smiles(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is not None:
        return Chem.MolToSmiles(mol)
    else:
        return ''


def pic50_to_ic50(pic50):
    return math.pow(10, -pic50)


class qed_func():

    def __call__(self, mol_list):
        scores = []
        for mol in mol_list:
            # mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                scores.append(0)
            else:
                try:
                    scores.append(QED.qed(mol))
                except Exception as e:
                    scores.append(0)
        return np.float32(scores)


class sa_func():

    def __call__(self, mol_list):
        scores = []
        for mol in mol_list:
            # mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                scores.append(100)
            else:
                try:
                    scores.append(sascorer.calculateScore(mol))
                except Exception as e:
                    scores.append(100)
        return np.float32(scores)


class logp_func():

    def __call__(self, mol_list):
        scores = []
        for mol in mol_list:
            # mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                scores.append(100)
            else:
                try:
                    scores.append(Descriptors.MolLogP(mol))
                except Exception as e:
                    scores.append(100)
        return np.float32(scores)


class logs_func():

    def __call__(self, mol_list):
        scores = []
        for mol in mol_list:
            # mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                scores.append(100)
            else:
                try:
                    scores.append(Descriptors.MolLogS(mol))
                except Exception as e:
                    scores.append(100)
        return np.float32(scores)


class tpsa_func():

    def __call__(self, mol_list):
        scores = []
        for mol in mol_list:
            # mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                scores.append(100)
            else:
                try:
                    scores.append(Descriptors.TPSA(mol))
                except Exception as e:
                    scores.append(100)
        return np.float32(scores)

class molwt_func():

    def __call__(self, mol_list):
        scores = []
        for mol in mol_list:
            # mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                scores.append(100)
            else:
                try:
                    scores.append(Descriptors.MolWt(mol))
                except Exception as e:
                    scores.append(100)
        return np.float32(scores)


def get_prop_function(prop_name):
    if prop_name == 'QED':
        return qed_func()
    elif prop_name == 'SAScore':
        return sa_func()
    elif prop_name == 'LogP':
        return logp_func()
    elif prop_name == 'TPSA':
        return tpsa_func()
    elif prop_name == 'MolWt':
        return molwt_func()
    else:
        raise ValueError


def get_data_from_smiles(smiles: List[List[str]],
                         skip_invalid_smiles: bool = True,
                         logger: Logger = None,
                         features_generator: List[str] = None) -> MoleculeDataset:
    """
    Converts a list of SMILES to a :class:`~chemprop.data.MoleculeDataset`.

    :param smiles: A list of lists of SMILES with length depending on the number of molecules.
    :param skip_invalid_smiles: Whether to skip and filter out invalid smiles using :func:`filter_invalid_smiles`
    :param logger: A logger for recording output.
    :param features_generator: List of features generators.
    :return: A :class:`~chemprop.data.MoleculeDataset` with all of the provided SMILES.
    """
    debug = logger.debug if logger is not None else print

    data = MoleculeDataset([
        MoleculeDatapoint(
            smiles=[smile], 
            row=OrderedDict({'smiles': smile}),
            features_generator=features_generator
        ) for smile in smiles
    ])

    # Filter out invalid SMILES
    if skip_invalid_smiles:
        original_data_len = len(data)
        data = filter_invalid_smiles(data)

        if len(data) < original_data_len:
            debug(f'Warning: {original_data_len - len(data)} SMILES are invalid.')

    return data
class chemprop_model():

    def __init__(self, checkpoint_dir):
        self.checkpoints = []
        for root, _, files in os.walk(checkpoint_dir):
            for fname in files:

                if fname.endswith('.pt'):
                    fname = os.path.join(root, fname)
                    self.scaler, self.features_scaler, self.atom_descriptor_scaler, self.bond_feature_scaler= load_scalers(fname)
                    self.train_args = load_args(fname)
                    model = load_checkpoint(fname, device=torch.device('cpu'))
                    self.checkpoints.append(model)
                # elif fname == 'args.json':
                #     fname = os.path.join(root, fname)
                #     self.train_args = load_args(fname)

    def __call__(self, smiles, batch_size=500):
        test_data = get_data_from_smiles(smiles=smiles, skip_invalid_smiles=False)
        valid_indices = [i for i in range(len(test_data)) if test_data[i].mol is not None]
        full_data = test_data
        test_data = MoleculeDataset([test_data[i] for i in valid_indices])

        if self.train_args.features_scaling:
            test_data.normalize_features(self.features_scaler)
        test_data_loader = MoleculeDataLoader(
            dataset=test_data,
            batch_size=self.train_args.batch_size,
            num_workers=self.train_args.num_workers
        )
        sum_preds = np.zeros((len(test_data), 1))
        for model in self.checkpoints:
            model_preds = predict(
                model=model,
                data_loader=test_data_loader,
                # batch_size=batch_size,
                scaler=self.scaler
            )
            sum_preds += np.array(model_preds)

        # Ensemble predictions
        avg_preds = sum_preds / len(self.checkpoints)
        avg_preds = avg_preds.squeeze(-1).tolist()

        # Put zero for invalid smiles
        full_preds = [0.0] * len(full_data)
        for i, si in enumerate(valid_indices):
            full_preds[si] = avg_preds[i]

        return np.array(full_preds, dtype=np.float32)


if __name__ == '__main__':
    target_name = '11_beta_HSD1_str_act_PIC50_DMPNN'
    model_path = os.path.join('chemprop_dmpnn_model', target_name)

    input_smiles = [
        'COc1cc2c(NC(=N)Nc3ccccc3C)ncnc2cc1OCC1CCN(C)CC1',
        'CCCCCCCCCC',
        'CCCCCCCCCC',
    ]

    chemprop_predictor = chemprop_model(model_path)

    print(chemprop_predictor(input_smiles))
