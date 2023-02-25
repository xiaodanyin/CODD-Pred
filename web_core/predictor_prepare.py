import os
from socket import IPV6_UNICAST_HOPS
import subprocess
from rdkit import Chem
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import pandas as pd
import numpy as np
import re

try:
    from model import GCN_net, MPNN_net, GAT_net, GIN_net, AttentiveFP_net, DMPNN, DMPNN_Change
    from utils import mol2graph, seed_torch, get_prop_function, pic50_to_ic50, chemprop_model
except:
    from web_core.model import GCN_net, MPNN_net, GAT_net, GIN_net, AttentiveFP_net, DMPNN, DMPNN_Change
    from web_core.utils import mol2graph, seed_torch, get_prop_function, pic50_to_ic50, chemprop_model

import os, sys

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

support_graph_model_list = [
    'GAT_net',
    'GCN_net',
    'GIN_net',
    'MPNN_net',
    'DMPNN',
    'DMPNN_Change',
    'AttentiveFP_net',
    'ChemProp_DMPNN',
]

support_classic_model_list = [
    'svm',
    'xgboost',
    'rf'
]
support_model_list = support_graph_model_list + support_classic_model_list

support_target_list = [

    # PIC50:

    'VEGFR1_str_act',
    'VEGFR2_str_act',
    'VEGFR3_str_act',
    'EGFR_str_act',
    'Anaplastic_lymphoma_kinase_str_act',
    'C_src_tyrosine_kinase_str_act',
    'Aurora_kinase_B_str_act',
    'BRAF_str_act',
    'Bromodomain_containing_4_str_act',
    'Aurora_kinase_A_str_act',
    'PDGFR_alpha_str_act',
    'PDGFR_beta_str_act',
    'HDAC1_str_act',
    'HDAC2_str_act',
    'HDAC6_str_act',
    'IGF1R_str_act',
    'Brutons_tyrosine_kinase_str_act',
    'CHK1_str_act',
    'PIK3CA_str_act',
    'PIK3CB_str_act',
    'PIK3CD_str_act',
    'PIK3CG_str_act',
    'FGFR1_str_act',
    'FGFR2_str_act',
    'FGFR3_str_act',
    'FLT3_str_act',
    'RET_proto_oncogene_str_act',
    '11_beta_HSD1_str_act',
    'Cannabinoid_receptor_1_str_act',
    'Cathepsin_K_str_act',
    'Farnesyl_protein_transferase_str_act',
    'Janus_kinase_1_str_act',
    'Janus_kinase_2_str_act',
    'Janus_kinase_3_str_act',
    'Renin_str_act',
    'Cathepsin_S_str_act',
    'Dipeptidyl_peptidase_4_str_act',
    'Janus_kinase_3_str_act',
    'MMP_13_str_act',
    'MMP_3_str_act',
    'MMP_9_str_act',
    'Nuclear_receptor_ROR_gamma_T_str_act',
    'Phosphodiesterase_4_str_act',
    'Spleen_tyrosine_kinase_str_act',
    'TNF_alpha_str_act',
    'AKT1_str_act',
    'AKT2_str_act',
    'AKT3_str_act',
    'Angiotensin_converting_enzyme_str_act',
    'BCL2_str_act',
    'CDK1_str_act',
    'CDK2_str_act',
    'MAP2K1_str_act',
    'Mineralocorticoid_receptor_str_act',
    'MMP_14_str_act',
    'Nicotinamide_phosphoribosyltransferase_str_act',
    'PIK3_str_act',
    'PKA_str_act',
    'PKC_theta_str_act',
    'C_C_chemokine_receptor_type_2_str_act',
    'C_C_chemokine_receptor_type_3_str_act',
    'MMP_2_str_act',
    'ZAP70_str_act',
    'Cathepsin_B_str_act',

    # IC50: plus: > 1um; minus: <= 1um

    'VEGFR2_str_act_plus',
    'VEGFR2_str_act_minus',
    'EGFR_str_act_plus',
    'EGFR_str_act_minus',
    'PIK3CA_str_act_plus',
    'PIK3CA_str_act_minus',
    'Spleen_tyrosine_kinase_str_act_plus',
    'Spleen_tyrosine_kinase_str_act_minus',
    'Janus_kinase_3_str_act_plus',
    'Janus_kinase_3_str_act_minus',
    'PDGFR_beta_str_act_plus',
    'PDGFR_beta_str_act_minus',
    'C_src_tyrosine_kinase_str_act_plus',
    'C_src_tyrosine_kinase_str_act_minus',
    'Aurora_kinase_A_str_act_plus',
    'Aurora_kinase_A_str_act_minus',
    'C_C_chemokine_receptor_type_3_str_act_plus',
    'C_C_chemokine_receptor_type_3_str_act_minus',
    'PIK3CD_str_act_plus',
    'PIK3CD_str_act_minus',
    'Cannabinoid_receptor_1_str_act_plus',
    'Cannabinoid_receptor_1_str_act_minus',
    'IGF1R_str_act_plus',
    'IGF1R_str_act_minus',
]


_MODEL_SHOW2BACKEND = {
    'MPNN': 'MPNN_net',
    'self_model': 'DMPNN_Change',
    'GCN': 'GCN_net',
    'GAT': 'GAT_net',
    'GIN': 'GIN_net',
    'AttentiveFP': 'AttentiveFP_net',
    'ChemProp DMPNN': 'ChemProp_DMPNN',
    'SVM': 'svm',
    'XGBoost': 'xgboost',
    'RF': 'rf',

}

_TARGET_SHOW2BACKEND = {

    # PIC50:

    'VEGFR-1': 'VEGFR1_str_act',
    'VEGFR-2': 'VEGFR2_str_act',
    'VEGFR-3': 'VEGFR3_str_act',
    'EGFR': 'EGFR_str_act',
    'Anaplastic_lymphoma_kinase': 'Anaplastic_lymphoma_kinase_str_act',
    'C_src_tyrosine_kinase': 'C_src_tyrosine_kinase_str_act',
    'Aurora_kinase_B': 'Aurora_kinase_B_str_act',
    'BRAF': 'BRAF_str_act',
    'Bromodomain_containing_4': 'Bromodomain_containing_4_str_act',
    'Brutons_tyrosine_kinase': 'Brutons_tyrosine_kinase_str_act',
    'Aurora_kinase_A':   'Aurora_kinase_A_str_act',
    'PDGFR_alpha': 'PDGFR_alpha_str_act',
    'PDGFR_beta': 'PDGFR_beta_str_act',
    'HDAC1': 'HDAC1_str_act',
    'HDAC2': 'HDAC2_str_act',
    'HDAC6': 'HDAC6_str_act',
    'IGF1R': 'IGF1R_str_act',
    'CHK1': 'CHK1_str_act',
    'PIK3CA': 'PIK3CA_str_act',
    'PIK3CB': 'PIK3CB_str_act',
    'PIK3CD': 'PIK3CD_str_act',
    'PIK3CG': 'PIK3CG_str_act',
    'FGFR1': 'FGFR1_str_act',
    'FGFR2': 'FGFR2_str_act',
    'FGFR3': 'FGFR3_str_act',
    'FLT3': 'FLT3_str_act',
    'RET_proto_oncogene': 'RET_proto_oncogene_str_act',
    '11_beta_HSD1': '11_beta_HSD1_str_act',
    'Cannabinoid_receptor_1': 'Cannabinoid_receptor_1_str_act',
    'Cathepsin_K': 'Cathepsin_K_str_act',
    'Farnesyl_protein_transferase': 'Farnesyl_protein_transferase_str_act',
    'Janus_kinase_1': 'Janus_kinase_1_str_act',
    'Janus_kinase_2': 'Janus_kinase_2_str_act',
    'Janus_kinase_3': 'Janus_kinase_3_str_act',
    'Renin': 'Renin_str_act',
    'Cathepsin_S': 'Cathepsin_S_str_act',
    'Dipeptidyl_peptidase_4': 'Dipeptidyl_peptidase_4_str_act',
    'Janus_kinase_3': 'Janus_kinase_3_str_act',
    'MMP_13': 'MMP_13_str_act',
    'MMP_3': 'MMP_3_str_act',
    'MMP_9': 'MMP_9_str_act',
    'Nuclear_receptor_ROR_gamma_T': 'Nuclear_receptor_ROR_gamma_T_str_act',
    'Phosphodiesterase_4': 'Phosphodiesterase_4_str_act',
    'Spleen_tyrosine_kinase': 'Spleen_tyrosine_kinase_str_act',
    'TNF_alpha': 'TNF_alpha_str_act',
    'AKT-1': 'AKT1_str_act',
    'AKT-2': 'AKT2_str_act',
    'AKT-3': 'AKT3_str_act',
    'Angiotensin_converting_enzyme': 'Angiotensin_converting_enzyme_str_act',
    'BCL-2': 'BCL2_str_act',
    'CDK-1': 'CDK1_str_act',
    'CDK-2': 'CDK2_str_act',
    'MAP2K1': 'MAP2K1_str_act',
    'Mineralocorticoid_receptor': 'Mineralocorticoid_receptor_str_act',
    'MMP_14': 'MMP_14_str_act',
    'Nicotinamide_phosphoribosyltransferase': 'Nicotinamide_phosphoribosyltransferase_str_act',
    'PIK3': 'PIK3_str_act',
    'PKA': 'PKA_str_act',
    'PKC θ': 'PKC_theta_str_act',
    'C_C_chemokine_receptor_type_2': 'C_C_chemokine_receptor_type_2_str_act',
    'C_C_chemokine_receptor_type_3': 'C_C_chemokine_receptor_type_3_str_act',
    'MMP_2': 'MMP_2_str_act',
    'ZAP70': 'ZAP70_str_act',
    'Cathepsin_B': 'Cathepsin_B_str_act',


    # IC50: plus: > 1um; minus: <= 1um

    'VEGFR-2_p': 'VEGFR2_str_act_plus',
    'VEGFR-2_m': 'VEGFR2_str_act_minus',
    'EGFR_p': 'EGFR_str_act_plus',
    'EGFR_m': 'EGFR_str_act_minus',
    'PIK3CA_p': 'PIK3CA_str_act_plus',
    'PIK3CA_m': 'PIK3CA_str_act_minus',
    'Spleen_tyrosine_kinase_p': 'Spleen_tyrosine_kinase_str_act_plus',
    'Spleen_tyrosine_kinase_m': 'Spleen_tyrosine_kinase_str_act_minus',
    'Janus_kinase_3_p': 'Janus_kinase_3_str_act_plus',
    'Janus_kinase_3_m': 'Janus_kinase_3_str_act_minus',
    'PDGFR_beta_p': 'PDGFR_beta_str_act_plus',
    'PDGFR_beta_m': 'PDGFR_beta_str_act_minus',
    'C_src_tyrosine_kinase_p': 'C_src_tyrosine_kinase_str_act_plus',
    'C_src_tyrosine_kinase_m': 'C_src_tyrosine_kinase_str_act_minus',
    'Aurora_kinase_A_p': 'Aurora_kinase_A_str_act_plus',
    'Aurora_kinase_A_m': 'Aurora_kinase_A_str_act_minus',
    'C_C_chemokine_receptor_type_3_p': 'C_C_chemokine_receptor_type_3_str_act_plus',
    'C_C_chemokine_receptor_type_3_m': 'C_C_chemokine_receptor_type_3_str_act_minus',
    'PIK3CD_p': 'PIK3CD_str_act_plus',
    'PIK3CD_m': 'PIK3CD_str_act_minus',
    'Cannabinoid_receptor_1_p': 'Cannabinoid_receptor_1_str_act_plus',
    'Cannabinoid_receptor_1_m': 'Cannabinoid_receptor_1_str_act_minus',
    'IGF1R_p': 'IGF1R_str_act_plus',
    'IGF1R_m': 'IGF1R_str_act_minus',

}

ref_path = os.path.join(os.path.abspath(
    os.path.dirname(__file__)), 'predict_ref', 'data.csv')
ref_folder = os.path.abspath(os.path.dirname(ref_path))
split_info_folder = os.path.join(ref_folder, 'split_info')
if not os.path.exists(split_info_folder):
    os.makedirs(split_info_folder)

ref_ic50_path = os.path.join(os.path.abspath(
    os.path.dirname(__file__)), 'predict_ref', 'data_ic50.csv')
split_info_ic50_folder = os.path.join(ref_folder, 'split_info_ic50')
if not os.path.exists(split_info_ic50_folder):
    os.makedirs(split_info_ic50_folder)


class GoPredictor:

    def __init__(self, model_net: str, target_name: str, uom: str, device=torch.device('cpu'), train_try_num=0):

        if model_net not in support_model_list:
            raise ValueError('{} is not support! Please select:\n    {}'.format(
                model_net, ', '.join(support_model_list)))
            return
        if target_name not in support_target_list:
            raise ValueError('{} is not support! Please select:\n    {}'.format(
                target_name, ', '.join(support_target_list)))
            return
        self.model_net = model_net
        self.target_name = target_name
        self.uom = uom
        self.train_try_num = train_try_num
        self.device = device

        self.task_mark = f'{self.target_name}_{self.uom}_model_{self.model_net}_{self.train_try_num}'

        self.current_path = os.path.dirname(os.path.abspath(__file__))
        if uom == 'IC50':
            data_path = 'dataset_path_layered'
            target_data_space = re.split('_plus|_minus', self.target_name)[0]
        elif uom == 'PIC50':
            data_path = 'dataset_path'
            target_data_space = self.target_name
        self.dataset_parm_path = os.path.join(
            self.current_path, data_path, 'raw', f'{target_data_space}')
        self.dataset_parm = torch.load(os.path.join(
            self.dataset_parm_path, f'{self.target_name}_{self.uom}_dataset_parm.pkle'))
        self.atom_types = self.dataset_parm['atom_types']

        if self.model_net in support_graph_model_list:
            self.is_classic = False
            if self.model_net != 'ChemProp_DMPNN':
                self.prepare_graph_model()
            elif self.model_net == 'ChemProp_DMPNN':
                self.prepare_chemprop_model()
        else:
            self.is_classic = True
            self.prepare_classic_model()

    def prepare_graph_model(self):
        uom_flag = self.uom.lower()
        model_save_path = os.path.join(self.current_path, f'model_{uom_flag}')

        train_parm_dic = torch.load(os.path.join(
            model_save_path, f'{self.task_mark}_train_parm.pkl'))
        train_parm_dic['mol_in_dim'] = train_parm_dic.get(
            'mol_in_dim', 61)
        train_parm_dic['mol_in_edge_dim'] = train_parm_dic.get(
            'mol_in_edge_dim', 14)
        self.train_parm_dic = train_parm_dic
        self_num_node_features = train_parm_dic['mol_in_dim']
        self_num_edge_features = train_parm_dic['mol_in_edge_dim']
        out_dim = train_parm_dic['out_dim']
        hidden_dim = train_parm_dic['hidden_dim']
        gat_heads = train_parm_dic['gat_heads']
        massage_depth = train_parm_dic['massage_depth']
        mess_nn_dim = train_parm_dic['mess_nn_dim']
        use_gru = train_parm_dic['use_gru']

        if self.model_net == 'GCN_net':
            model = GCN_net(mol_in_dim=self_num_node_features,
                            out_dim=out_dim, dim=hidden_dim)
        elif self.model_net == 'MPNN_net':
            model = MPNN_net(mol_in_node_dim=self_num_node_features, mol_in_edge_dim=self_num_edge_features,
                             out_dim=out_dim, dim=hidden_dim, mess_nn_dim=mess_nn_dim)
        elif self.model_net == 'GAT_net':
            model = GAT_net(mol_in_dim=self_num_node_features,
                            out_dim=out_dim, gat_heads=gat_heads, dim=hidden_dim)
        elif self.model_net == 'GIN_net':
            model = GIN_net(mol_in_dim=self_num_node_features,
                            out_dim=out_dim, dim=hidden_dim)
        elif self.model_net == 'AttentiveFP_net':
            model = AttentiveFP_net(mol_in_dim=self_num_node_features,
                                    mol_in_edge_dim=self_num_edge_features, out_dim=out_dim, dim=hidden_dim)
        elif self.model_net == 'DMPNN':
            model = DMPNN(mol_in_dim=self_num_node_features, out_dim=out_dim, dim=hidden_dim,
                          f_ab_size=hidden_dim + self_num_edge_features)
        elif self.model_net == 'DMPNN_Change':
            model = DMPNN_Change(mol_in_dim=self_num_node_features, out_dim=out_dim, dim=hidden_dim,
                                 mol_in_edge_dim=self_num_edge_features, use_gru=use_gru,
                                 massage_depth=massage_depth,
                                 dropout_rate=0.4)
        else:
            raise ValueError
        model_state = torch.load(os.path.join(
            model_save_path, f'{self.task_mark}_state.pkl'), map_location='cpu')
        model.load_state_dict(model_state)
        model.eval()
        self.model = model

    def prepare_chemprop_model(self):
        with HiddenPrints():
            model_dir = os.path.join(
                self.current_path, 'chemprop_dmpnn_model', f'{self.target_name}_{self.uom}_DMPNN')
            self.model = chemprop_model(model_dir)

    def prepare_classic_model(self):
        model_save_path = os.path.join(self.current_path, 'model_classic')
        self.model = torch.load(os.path.join(
            model_save_path,  f'{self.target_name}_{self.uom}_{self.model_net}_train.pkl'))

    def predict_single(self, smi):
        if self.model_net == 'ChemProp_DMPNN':
            pred = self.run_chemprop([smi])[0]
        else:
            mol = Chem.MolFromSmiles(smi)
            x, edge_index, edge_attr, fp = mol2graph(mol, self.atom_types)
            data = Data(x=x, edge_index=edge_index,
                        edge_attr=edge_attr, smiles=smi, fp=fp)
            data.batch = torch.LongTensor([0])
            if not self.is_classic:
                with torch.no_grad():
                    pred = self.model(data)
            else:
                pred = torch.from_numpy(self.model.predict(data.fp.numpy()))

            pred = self.restore_pred(pred)
            pred = pred.item()

        return pred

    def predict_multi(self, smiles_list):
        dummy_smiles = ['CCCC']
        smiles_list = dummy_smiles + smiles_list
        if self.model_net == 'ChemProp_DMPNN':
            pred_list = self.run_chemprop(smiles_list)
        else:
            mols = [Chem.MolFromSmiles(x) for x in smiles_list]
            graph_list = [mol2graph(mol, self.atom_types) for mol in mols]
            testset = [Data(x=x, edge_index=edge_index,
                            edge_attr=edge_attr, smiles=smiles_list[i], fp=fp) for i, (x, edge_index, edge_attr, fp) in enumerate(graph_list)]

            pred_list = []
            if not self.is_classic:
                self.model = self.model.to(self.device)
                test_loader = DataLoader(
                    testset, batch_size=128, shuffle=False)
                with torch.no_grad():
                    for data in test_loader:
                        data = data.to(self.device)
                        # data.batch = torch.LongTensor([0])

                        pred = self.model(data)

                        pred_list.append(pred.detach().cpu().numpy())
                pred_list = np.hstack(pred_list).tolist()

            else:
                fp_list = [data.fp.detach().cpu().numpy() for data in testset]
                fps = np.array(fp_list).squeeze()
                pred_list = self.model.predict(fps).tolist()
                # for fp in fp_list:
                #     pred = self.model.predict(fp)
                #     pred_list.append(pred)
            pred_list = [self.restore_pred(pred) for pred in pred_list]
            # pred = self.restore_pred(pred)
            # pred_list.append(pred.item())
        return pred_list[1:]

    # def run_chemprop(self, smiles_list):
    #     import pandas as pd
    #     cache_path = os.path.join(self.current_path, 'cache')
    #     if not os.path.exists(cache_path):
    #         os.makedirs(cache_path)
    #     test_fpath = os.path.join(cache_path, 'test.csv')
    #     pred_fpath = os.path.join(cache_path, 'test_pred.csv')
    #     model_dir = os.path.join(
    #         self.current_path, 'chemprop_dmpnn_model', f'{self.target_name}_{self.uom}_DMPNN')
    #     test_df = pd.DataFrame(
    #         {'smiles': smiles_list}
    #     )
    #     test_df.to_csv(test_fpath, index=False)
    #     commands = f"chemprop_predict --test_path {test_fpath} --checkpoint_dir {model_dir} --preds_path {pred_fpath}"

    #     devNull = open(os.devnull, 'w')
    #     proc = subprocess.Popen(commands.split(), stdout=devNull)
    #     proc.communicate()

    #     pred_df = pd.read_csv(pred_fpath)
    #     smiles_list = pred_df['smiles'].tolist()
    #     pred = torch.from_numpy(pred_df['y'].values)
    #     pred = self.restore_pred(pred)
    #     return pred.tolist()

    def run_chemprop(self, smiles_list):
        pred = self.model(smiles=smiles_list)
        pred = torch.from_numpy(pred)
        pred = self.restore_pred(pred)
        return pred.tolist()

    def restore_pred(self, pred):
        if self.uom == 'PIC50':
            return pred
        elif self.uom == 'IC50':
            y_max = self.dataset_parm['y_max']
            y_min = self.dataset_parm['y_min']
            maxy_miny_gap = y_max - y_min
            pred = pred * maxy_miny_gap + y_min
            return pred


def get_valid_info(ref_data, split_info_folder, r2_cutoff):
    data_dict = {}
    last_target = None

    # less_3_list = []
    for row in ref_data.itertuples():
        # print(row)
        idx, target_name, uom, model_name, _, _, _, _, test_mae, test_mse, test_rmse, test_r2 = row
        # if test_r2 < r2_cutoff:
        #     continue
        if target_name not in data_dict:
            # if last_target:
            #     print('{} have {} model\n'.format(last_target, len(
            #         data_dict[last_target]['model'])))
            # if len(
            #         data_dict[last_target]['model']) < 3:
            #     less_3_list.append(last_target)

            one_record = {
                'model': [],
                'uom': [],
                'mae': [],
                'mse': [],
                'rmes': [],
                'r2': [],
            }
            data_dict[target_name] = one_record
            last_target = target_name

        data_dict[target_name]['model'].append(model_name)
        data_dict[target_name]['uom'].append(uom)
        data_dict[target_name]['mae'].append(test_mae)
        data_dict[target_name]['mse'].append(test_mse)
        data_dict[target_name]['rmes'].append(test_rmse)
        data_dict[target_name]['r2'].append(test_r2)
    # print('{} have {} model'.format(target_name, len(
    #     data_dict[target_name]['model'])))
    # if len(
    #         data_dict[target_name]['model']) < 3:
    #     less_3_list.append(target_name)
    for k in data_dict:
        one_df = pd.DataFrame.from_dict(data_dict[k])
        r2_cut_df = one_df.loc[one_df['r2'] > r2_cutoff]
        if len(r2_cut_df) < 3:
            r2_list = one_df['r2'].tolist()
            r2_list.sort(reverse=True)
            extend_r2_cutoff = r2_list[2]

            one_df = one_df.loc[one_df['r2'] >= extend_r2_cutoff]
        elif len(r2_cut_df) > 5:
            r2_list = one_df['r2'].tolist()
            r2_list.sort(reverse=True)
            extend_r2_cutoff = r2_list[5]
            one_df = one_df.loc[one_df['r2'] > extend_r2_cutoff]
        else:
            one_df = r2_cut_df
        assert len(one_df) >= 3

        data_dict[k] = one_df
        data_dict[k].to_csv(os.path.join(split_info_folder,
                            f'{k}_valid_models.csv'), index=False)


def model_analyasis(ref_path=ref_path, support_target_list=support_target_list, pic_r2_cutoff=0.65, ic50_r2_cutoff=0.65):
    ref_pic50_data = pd.read_csv(ref_path, sep=',')

    ref_pic50_data = ref_pic50_data.loc[~np.isnan(ref_pic50_data['test_R2'])]
    ref_record_target_name_pic50 = set(
        sorted(ref_pic50_data['target'].tolist()))

    ref_ic50_data = pd.read_csv(ref_ic50_path)
    ref_ic50_data = ref_ic50_data.loc[~np.isnan(ref_ic50_data['test_R2'])]
    ref_record_target_name_ic50 = set(sorted(ref_ic50_data['target'].tolist()))

    support_target = set(sorted(support_target_list))
    # assert ref_record_target_name == support_target

    if not support_target.issubset(
            ref_record_target_name_pic50 | ref_record_target_name_ic50):
        print('There is no reference data for the following targets:')
        for t in support_target:
            if t not in (ref_record_target_name_pic50 | ref_record_target_name_ic50):
                print(t)
        raise ValueError

    # assert support_target.issubset(
    #     ref_record_target_name_pic50 | ref_record_target_name_ic50)

    get_valid_info(ref_data=ref_pic50_data,
                   split_info_folder=split_info_folder, r2_cutoff=pic_r2_cutoff)
    get_valid_info(ref_data=ref_ic50_data,
                   split_info_folder=split_info_ic50_folder, r2_cutoff=ic50_r2_cutoff)


def merge_predictor(target_name, best=False, uom='PIC50', device=torch.device('cpu')):
    if uom == 'PIC50':
        info_folder = split_info_folder
    else:
        info_folder = split_info_ic50_folder
    ref_df = pd.read_csv(os.path.join(
        info_folder, f'{target_name}_valid_models.csv'))
    if best:
        model_net = ref_df.loc[ref_df['mae'].argmin()]['model']
        predictor = GoPredictor(model_net=model_net, target_name=target_name,
                                uom=uom, device=device, train_try_num=0)
        return predictor, model_net
    else:
        model_net_list = ref_df['model'].to_list()
        return [GoPredictor(model_net=model_net, target_name=target_name,
                            uom=uom, device=device, train_try_num=0) for model_net in model_net_list], model_net_list


def reject_outliers(data, m=2):
    _idx = abs(data - np.mean(data)) < m * np.std(data)
    return data[_idx], _idx


def merge_predict(input_smiles, target_name, best=False, uom='PIC50', device=torch.device('cpu')):
    if not best:
        predictor_fn, mode_net_list = merge_predictor(
            target_name=target_name, best=best, uom=uom, device=device)
        results = [predictor.predict_multi(input_smiles
                                           ) for predictor in predictor_fn]
        results = np.array(results)
        filter_output = [reject_outliers(
            results[:, i], m=1) for i in range(results.shape[1])]
        filter_results, filter_idx = [], []
        for x in filter_output:
            filter_results.append(x[0].reshape(-1, 1))
            filter_idx.append(x[1])
        output = [result.mean() for result in filter_results]
        model_net_array = np.asanyarray(mode_net_list)
        model_net_flag = [','.join(model_net_array[_idx])
                          for _idx in filter_idx]
        return output, model_net_flag
    else:
        predictor, model_net_flag = merge_predictor(
            target_name=target_name, best=best, uom=uom, device=device)
        output = predictor.predict_multi(
            input_smiles)
        return output, [model_net_flag for i in range(len(output))]


def calculate_drug_prop(smiles_list):
    function_names = [
        'QED',
        'SAScore',
        'LogP',
        'TPSA',
        'MolWt'
    ]
    mol_list = [Chem.MolFromSmiles(x) for x in smiles_list]
    functions = [get_prop_function(name) for name in function_names]
    props = np.array([func(mol_list) for func in functions]).T
    props = pd.DataFrame(props)
    props.columns = function_names
    return props


def accurate_pipeline(input_smiles, target_name, best, device=torch.device('cpu')):
    '''
    先使用PIC50预测流程估计PIC50，根据预测值判断调用的IC50的分区间模型:
    pred(pic50) <= 6 : 调用uM模型，标记为plus;
    pred(pic50) > 6 : 调用nM模型，标记为minus
    '''
    pred_pic50, model_net_flag_pic50 = merge_predict(input_smiles=input_smiles,
                                                     target_name=target_name, best=best, uom='PIC50', device=device)

    mask_plus = (np.array(pred_pic50) <= 6)
    mask_minus = (np.array(pred_pic50) > 6)

    smiles_to_plus = np.asanyarray(input_smiles)[mask_plus].tolist()
    smiles_to_minus = np.asanyarray(input_smiles)[mask_minus].tolist()
    pred_ic50 = np.zeros((len(input_smiles)))
    if smiles_to_plus:
        pred_ic50_p, model_net_flag_ic50_p = merge_predict(input_smiles=smiles_to_plus,
                                                           target_name=f'{target_name}_plus', best=best, uom='IC50', device=device)
        pred_ic50[mask_plus] = pred_ic50_p
    if smiles_to_minus:
        pred_ic50_m, model_net_flag_ic50_m = merge_predict(input_smiles=smiles_to_minus,
                                                           target_name=f'{target_name}_minus', best=best, uom='IC50', device=device)
        pred_ic50[mask_minus] = pred_ic50_m

    return pred_ic50, []


if __name__ == '__main__':

    '''
    'MPNN': 'MPNN_net',
    'self_model': 'DMPNN_Change',
    'GCN': 'GCN_net',
    'GAT': 'GAT_net',
    'GIN': 'GIN_net',
    'AttentiveFP': 'AttentiveFP_net',
    'ChemProp DMPNN': 'ChemProp_DMPNN',
    'SVM': 'svm',
    'XGBoost': 'xgboost',
    'RF': 'rf',
    '''

    seed_torch(123)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    best = False
    uom = 'PIC50'
    target_name_test = [
        'MMP_14',
        'PKA',
    ]
    input_smiles = [
        'COc1cc2c(NC(=N)Nc3ccccc3C)ncnc2cc1OCC1CCN(C)CC1',
        'CCCCCCCCCC',
        'CCCCCCCCCC',
    ]
    model_name_test = [
        'MPNN',
        # 'self_model',
        # 'GAT',
        # 'GIN',
        'ChemProp DMPNN',
        'SVM',
        'XGBoost',
        'RF',
    ]
    # error_group = []
    # for t in target_name_test:
    #     for m in model_name_test:
    #         try:
    #             predictor = GoPredictor(model_net=_MODEL_SHOW2BACKEND[m], target_name=_TARGET_SHOW2BACKEND[t],
    #                                     uom=uom, device=device, train_try_num=0)
    #             # print(predictor.predict_single('CC(C)(C)n1nc(-c2ccc3cc(C(=O)Nc4cnoc4)[nH]c3c2)c2c(N)ncnc21'))
    #             print(predictor.predict_multi(
    #                 input_smiles))
    #         except Exception as e:
    #             print(e)
    #             error_group.append(f'{t} + {m}')
    # print('############## Error [Target] + [Model] ################')
    # print(error_group)

    model_analyasis(ref_path=ref_path,
                    support_target_list=support_target_list, pic_r2_cutoff=0.65, ic50_r2_cutoff=0.55)

    error_group = []
    for t in target_name_test:

        try:
            print(merge_predict(input_smiles=input_smiles,
                  target_name=_TARGET_SHOW2BACKEND[t], best=best, uom=uom))
        except Exception as e:
            print(e)
            error_group.append(f'{t}')
    print('############## Merge Error [Target] ################')
    print(error_group)
    # drug_props = calculate_drug_prop(input_smiles)
    # print(drug_props)

    # error_group = []
    # for t in target_name_test:
    #     try:
    #         print(accurate_pipeline(input_smiles=input_smiles,
    #                                 target_name=_TARGET_SHOW2BACKEND[t], best=best, device=device))
    #     except Exception as e:
    #         print(e)
    #         error_group.append(f'{t}')
    # print('############## Accurate Pipeline Error [Target] ################')
    # print(error_group)

    # drug_props = calculate_drug_prop(input_smiles)
    # print(drug_props)
