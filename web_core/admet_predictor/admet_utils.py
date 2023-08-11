import json
import logging
import os
import numpy as np
import pandas as pd
import torch
import pickle
from torch import device, functional as F
from rdkit import Chem
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import sys
sys.path.append('./trimnet/trimnet_drug/source')
from trimnet.trimnet_drug.source.dataset import atom_attr, bond_attr
from trimnet.trimnet_drug.source.model import TrimNet as Trimnet_Model
# from DeepPurpose import CompoundPred as TDC_model
# from DeepPurpose.utils import encode_drug
from chemprop.train import predict
from chemprop.data import MoleculeDataset, MoleculeDataLoader
from chemprop.utils import load_args, load_checkpoint, load_scalers
try:
    import sys
    sys.path.append(os.path.abspath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), '../')))
    from utils import get_data_from_smiles
except:
    from web_core.utils import get_data_from_smiles

admet_root = os.path.dirname(os.path.abspath(__file__))
prop_names = [
    'Tox21',
    'Toxcast',
    'ClinTox',
    # 'Caco2',
    'CYP1A2_ContextPred',
    'CYP2D6_ContextPred',
    'CYP2D6_substrate_ContextPred',
    'CYP2C9_ContextPred',
    'CYP2C19_ContextPred',
    'CYP3A4_ContextPred',
    'CYP3A4_substrate_CNN',
    'HIA_AttrMasking',
    # 'PPBR_NeuralFP',
    'BBB_martins_ContextPred',
    'half_life_obach_AttrMasking',
    'pgp_broccatelli_AttrMasking',
    'LogD',
    'LogS',
]

# mol_target_prob_name = [
#     'Trimnet_small_mol_target_added_neg_deepsearch_prob'
# ]

# mol_target_class_name = [
#     'ChemProp_small_mol_target_classfication'
# ]

mol_target_prob_name = [
    'Trimnet_small_mol_target_640'
]

mol_target_class_name = [
    'ChemProp_small_mol_target_classfication_640'
]


def _load_config(prop_name):
    '''
    prop_name:
    Tox21, CastTox, ClinTox, Caco2, CPY2D6_ContextPred, CYP2D6_substrate_ContextPred, CYP2C9_ContextPred, CYP3A4_substrate_CNN, HIA_AttrMasking, PPBR_NeuralFP'''
    admet_config_path = os.path.join(admet_root, 'admet_config.json')
    with open(admet_config_path, "r") as f:
        admet_config = json.load(f)
    return admet_config[prop_name]


def trimnet_mol2graph(mol):
    if mol is None:
        return None
    node_attr = atom_attr(mol)
    edge_index, edge_attr = bond_attr(mol)
    # pos = torch.FloatTensor(geom)
    data = Data(
        x=torch.FloatTensor(node_attr),
        # pos=pos,
        edge_index=torch.LongTensor(edge_index).t(),
        edge_attr=torch.FloatTensor(edge_attr),
        y=None  # None as a placeholder
    )
    return data


def load_trimnet_model(config):
    model_info_path = os.path.join(admet_root, config['model_info'])
    model_info_dict = torch.load(
        model_info_path, map_location=torch.device('cpu'))
    config.update(model_info_dict['option'])
    model = Trimnet_Model(
        model_info_dict['option']['in_dim'],
        model_info_dict['option']['edge_in_dim'], hidden_dim=model_info_dict['option']['hid'],
        depth=model_info_dict['option']['depth'],
        heads=model_info_dict['option']['heads'],
        dropout=model_info_dict['option']['dropout'],
        outdim=model_info_dict['option']['out_dim']
    )
    model_state = model_info_dict['model_state_dict']
    model.load_state_dict(model_state)
    return model, config


def load_tdc_model(config, device):
    model_config_path = os.path.join(os.path.dirname(
        os.path.abspath(__file__)), config['model_config_path'])
    with open(model_config_path, 'rb') as f:
        saved_config = pickle.load(f)
    saved_config.update({'device': device})
    config.update(saved_config)
    config.update({'device': device})
    model = TDC_model.model_initialize(**saved_config)
    model.load_pretrained(os.path.join(admet_root, config['model_state_path']))
    return model, config


class ADMET_wraper:
    def __init__(self, model, config) -> None:
        '''
        task_category: regression or classification
        '''
        self.config = config
        self.model = model
        # self.model.eval()
        self.task_category = self.config['task_category']

    def predict(self, smiles_list=None, device=torch.device('cpu')):
        dummy_smiles = ['CCCC']
        assert smiles_list
        smiles_list = dummy_smiles + smiles_list
        if self.config['model'] == 'trimnet':
            mol_list = [Chem.MolFromSmiles(x) for x in smiles_list]
            data_list = [trimnet_mol2graph(x) for x in mol_list]

            test_loader = DataLoader(data_list, batch_size=128, shuffle=False)
            pred_list = []
            self.model = self.model.to(device)
            self.model.eval()
            with torch.no_grad():
                for data in test_loader:
                    data = data.to(device)
                    pred = self.model(data).detach().cpu()
                    pred_list.append(pred)
            predictions = torch.cat(pred_list, dim=0)
            if self.task_category == 'classification':
                output_dict = {}
                task_names = self.config['tasks']
                task_num = len(task_names)
                for i, task_name in enumerate(task_names):
                    y_pred = predictions[1:, i * 2:(i + 1) * 2]
                    y_pred = torch.softmax(
                        y_pred.detach().cpu(), dim=-1)[:, 1].view(-1).numpy()
                    output_dict[task_name] = y_pred.tolist()
                output_df = pd.DataFrame.from_dict(output_dict)  # 返回阳性概率
                return output_df
        elif self.config['model'] == 'tdc_CompoundPred':
            test = pd.DataFrame(zip(smiles_list, [0.0 for _ in smiles_list]))
            test.rename(columns={0: 'SMILES',
                                 1: 'Label'},
                        inplace=True)
            test = encode_drug(
                df_data=test, drug_encoding=self.config['drug_encoding'])
            self.model.device = device
            # self.model.model = self.model.model.to(device)
            y_pred = self.model.predict(test)
            if self.task_category == 'classification':
                output_dict = {}
                if 'tasks' not in self.config:
                    output_dict[self.config['task_name'] +
                                '_classification_prob'] = y_pred[1:]
                    output_df = pd.DataFrame.from_dict(output_dict)
            elif self.task_category == 'regression':
                output_dict = {}
                if 'tasks' not in self.config:
                    output_dict[self.config['task_name'] +
                                '_regression_y'] = y_pred[1:]
                    output_df = pd.DataFrame.from_dict(output_dict)
            return output_df


class ADMET_predictor:
    def __init__(self) -> None:
        self.configs = [_load_config(n) for n in prop_names]
        self.predictor = []
        for config in self.configs:
            if config['model'] == 'trimnet':
                trimnet, config = load_trimnet_model(config)
                trimnet_wraper = ADMET_wraper(
                    trimnet, config=config)
                self.predictor.append(trimnet_wraper)
            elif config['model'] == 'tdc_CompoundPred':
                tdcmodel, config = load_tdc_model(config, device=device)
                tdcmodel_wraper = ADMET_wraper(tdcmodel, config=config)
                self.predictor.append(tdcmodel_wraper)

    def predict(self, smiles_list, device=torch.device('cpu')):
        results = [wraper.predict(smiles_list, device=device)
                   for wraper in self.predictor]
        results = pd.concat(results, axis=1)
        return results


class MolTargetProb_predictor:
    def __init__(self) -> None:
        self.configs = [_load_config(n) for n in mol_target_prob_name]
        self.predictor = []
        for config in self.configs:
            if config['model'] == 'trimnet':
                trimnet, config = load_trimnet_model(config)
                trimnet_wraper = ADMET_wraper(
                    trimnet, config=config)
                self.predictor.append(trimnet_wraper)
            elif config['model'] == 'tdc_CompoundPred':
                tdcmodel, config = load_tdc_model(config, device=device)
                tdcmodel_wraper = ADMET_wraper(tdcmodel, config=config)
                self.predictor.append(tdcmodel_wraper)

    def predict(self, smiles_list, device=torch.device('cpu')):
        results = [wraper.predict(smiles_list, device=device)
                   for wraper in self.predictor]
        results = pd.concat(results, axis=1)
        return results


class MolTarget_classfication:
    def __init__(self):
        self.config = _load_config(mol_target_class_name[0])
        target2label = pd.read_csv(os.path.join(
            admet_root, self.config['target2label']), header=None)
        print('Loaded {} target to label info'.format(len(target2label)))
        self.target2label = {k: v for k, v in zip(
            target2label[0], target2label[1])}
        self.label2target = {v: k for k, v in self.target2label.items()}
        assert len(self.target2label) == len(target2label)
        # self.checkpoints = []
        for root, _, files in os.walk(os.path.join(admet_root, self.config['model_info'])):
            for fname in files:
                if fname.endswith('.pt'):
                    fname = os.path.join(root, fname)
                    self.scaler, self.features_scaler, self.atom_descriptor_scaler, self.bond_feature_scaler = load_scalers(
                        fname)
                    self.train_args = load_args(fname)
                    self.model = load_checkpoint(
                        fname, device=torch.device('cpu'))

    def predict(self, smiles_list):
        test_data = get_data_from_smiles(
            smiles=smiles_list, skip_invalid_smiles=False)
        valid_indices = [i for i in range(
            len(test_data)) if test_data[i].mol is not None]
        full_data = test_data
        test_data = MoleculeDataset([test_data[i] for i in valid_indices])

        if self.train_args.features_scaling:
            test_data.normalize_features(self.features_scaler)
        test_data_loader = MoleculeDataLoader(
            dataset=test_data,
            batch_size=self.train_args.batch_size,
            num_workers=self.train_args.num_workers
        )

        model_preds = predict(
            model=self.model,
            data_loader=test_data_loader,
            # batch_size=batch_size,
            scaler=self.scaler
        )
        pred = np.array(model_preds).squeeze()
        # prob_df = pd.DataFrame(pred.reshape(1, -1), dtype=np.float64)
        prob_df = pd.DataFrame(pred.reshape(len(smiles_list), -1), dtype=np.float64)
        prob_df.columns = [self.label2target[i]
            for i in range(len(self.label2target))]

        return prob_df


class MolTargetProb_calculator:
    def __init__(self) -> None:
        # with open(os.path.join(admet_root, 'script', 'data','std_name2uniprot_id.json'), 'r', encoding='utf-8') as f:
        #     self.std_name2_uniport_id = json.load(f)
        # with open(os.path.join(admet_root, 'script', 'data','uniport_id2chembl_target_name.json'), 'r', encoding='utf-8') as f:
        #     self.uniport_id2chembl_target_name = json.load(f)
        print("加载靶点预测模型！！！！！！！！！！！！！！！！！！！！！！！！！！！！")
        print("加载靶点预测模型！！！！！！！！！！！！！！！！！！！！！！！！！！！！")
        print("加载靶点预测模型！！！！！！！！！！！！！！！！！！！！！！！！！！！！")
        print("加载靶点预测模型！！！！！！！！！！！！！！！！！！！！！！！！！！！！")
        print("加载靶点预测模型！！！！！！！！！！！！！！！！！！！！！！！！！！！！")
        print("加载靶点预测模型！！！！！！！！！！！！！！！！！！！！！！！！！！！！")
        self.mol_target_prob_predictor = MolTargetProb_predictor()
        self.mol_target_class_predictor = MolTarget_classfication()
    def predict(self, smiles_list):

        mol_target_prob_results = self.mol_target_prob_predictor.predict(
                smiles_list=smiles_list, device=torch.device('cpu'))
        mol_target_class_results = self.mol_target_class_predictor.predict(smiles_list=smiles_list)
        mol_target_class_results = mol_target_class_results[mol_target_prob_results.columns]
        assert mol_target_prob_results.columns.tolist() == mol_target_class_results.columns.tolist()
        end_prob_df = pd.DataFrame(mol_target_prob_results.values * mol_target_class_results.values)
        # columns_to_show = []
        # for std_name in mol_target_prob_results.columns.tolist():
        #     uniport_id = self.std_name2_uniport_id[std_name]
        #     if uniport_id not in self.uniport_id2chembl_target_name:
        #         columns_to_show.append(std_name)
        #         print(std_name)
        #     else:
        #         columns_to_show.append(self.uniport_id2chembl_target_name[self.std_name2_uniport_id[std_name]])

        # end_prob_df.columns = columns_to_show
        end_prob_df.columns = mol_target_prob_results.columns
        return end_prob_df



if __name__ == '__main__':

    input_smiles = [
        # 'CC1=C(N(N=C1C(=O)NN1CCCCC1)C1=CC=C(Cl)C=C1Cl)C1=CC=C(Cl)C=C1',
        # 'CCC[C@H]1CCCCN1',
        'OC1=CC(Cl)=CC=C1OC1=CC=C(Cl)C=C1Cl',
        'CCCCCCCCCC',
        'CCCCCCCCCC',
    ]
    input_smiles = [Chem.MolToSmiles(Chem.MolFromSmiles(x)) for x in input_smiles]
    input_mols = [Chem.MolFromSmiles(x) for x in input_smiles]

    # config = load_config_for_admet('PPBR_NeuralFP')
    # trimnet, config = load_trimnet_model(config)
    # print(trimnet)

    # trimnet_wraper = ADMET_wraper(
    #     trimnet, config=config)
    # results = trimnet_wraper.predict(smiles_list=input_smiles, device=torch.device('cuda:0'))
    # results = trimnet_wraper.predict(mol_list=input_mols)
    # for de in ['cuda:0', 'cpu']:
    #     tdcmodel, config = load_tdc_model(config, device=torch.device(de))
    #     tdcmodel_wraper = ADMET_wraper(tdcmodel, config=config)
    #     # results = tdcmodel_wraper.predict(
    #     #     smiles_list=input_smiles, device=torch.device('cuda:0'))
    #     results = tdcmodel_wraper.predict(
    #         smiles_list=input_smiles, device=config['device'])
    #     print(results.head())
    # admet_predictor = ADMET_predictor()
    # for de in ['cuda:0', 'cpu']:

    #     results = admet_predictor.predict(
    #         smiles_list=input_smiles, device=torch.device(de))
    #     # [print(x.head()) for x in results]
    #     print(results.head())
    #     print(results.shape)

    # mol_target_prob_predictor = MolTargetProb_predictor()
    # for de in ['cuda:0', 'cpu']:

    #     results = mol_target_prob_predictor.predict(
    #         smiles_list=input_smiles, device=torch.device(de))
    #     # [print(x.head()) for x in results]
    #     print(results.head())
    #     print(results.shape)
    # mol_target_prob_results = mol_target_prob_predictor.predict(
    #         smiles_list=input_smiles, device=torch.device('cpu'))
    # mol_target_class_predictor = MolTarget_classfication()
    # mol_target_class_results = mol_target_class_predictor.predict(smiles_list=input_smiles)
    # mol_target_class_results = mol_target_class_results[mol_target_prob_results.columns]
    # assert mol_target_prob_results.columns.tolist() == mol_target_class_results.columns.tolist()

    # print(mol_target_prob_results.shape)
    # print(mol_target_class_results.shape)

    mol_target_prob_calculator = MolTargetProb_calculator()

    results = mol_target_prob_calculator.predict(input_smiles)
    std_name_list = results.columns.tolist()
    # prob_list = []
    # for std_name in std_name_list:
    #     prob_list.append(results[std_name][0])
    prob_list = [results[key].tolist()[0] for key in std_name_list]
    target_prob_pair = list(zip(std_name_list, prob_list))
    print('Prob > 0.95: {}'.format(
        len([x for x in target_prob_pair if x[1] > 0.95])))
    print('Prob > 0.9: {}'.format(
        len([x for x in target_prob_pair if x[1] > 0.9])))
    print('Prob > 0.5: {}'.format(
        len([x for x in target_prob_pair if x[1] > 0.5])))

    target_prob_pair.sort(key=lambda x: x[1], reverse=True)
    # target_prob_pair = [x for x in target_prob_pair if x[1] > 0.001]
    target_prob_pair = target_prob_pair[:15]
    print('+++++++++++++++++++++++++++++++++++++++++++++++++')
    for pair in target_prob_pair:
        print('{} : {}'.format(pair[0], pair[1]))