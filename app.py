
from itertools import islice
import json
from werkzeug.utils import secure_filename
import pandas as pd
import random
import time
import re
import csv
import numpy as np
from utils import get_h5_list, generate_smi, handle_file_deepxxx, handle_file_deepames, mol2svg
from wtforms.validators import DataRequired, Length, Email, EqualTo, NumberRange, regexp
from wtforms.fields import (StringField, PasswordField, DateField, BooleanField,
                            SelectField, SelectMultipleField, TextAreaField,
                            RadioField, IntegerField, DecimalField, SubmitField, FileField)
from wtforms import Form, validators
from flask import Flask, render_template, request, redirect, url_for
from rdkit import Chem
from flask_bootstrap import Bootstrap
import logging
import os
import sys
import torch

from torch._C import device
from web_core.predictor_prepare import GoPredictor, calculate_drug_prop, merge_predict, accurate_pipeline
from web_core.utils import canonicalize_smiles, seed_torch
from web_core.admet_predictor.admet_utils import ADMET_predictor, MolTargetProb_calculator
# from ml_model.inference import prediction
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ADMET'))

abs_dir = os.path.dirname(os.path.realpath(__file__))

seed_torch(123)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
queue_path = os.path.join('static', 'queue')
if not os.path.exists(queue_path):
    os.makedirs(queue_path)

app = Flask(__name__)
WTF_CSRF_ENABLED = True  # prevents CSRF attacks


@app.before_first_request
def first_request():
    del_list = os.listdir(queue_path)
    for f in del_list:
        file_path = os.path.join(queue_path, f)
        if os.path.isfile(file_path):
            os.remove(file_path)
    app.moltarget_prob_calculator = MolTargetProb_calculator()
    return app.moltarget_prob_calculator

class MolForm(Form):
    drawn_smiles = StringField(label='drawn_smiles')
    smiles = TextAreaField(label='smiles')
    file = FileField(label=None, render_kw={'class': 'form-control'})
    model_name = SelectField(u'Model',
                             choices=[
                                 'MPNN',
                                 'self_model',
                                 #  'GCN',
                                 'GAT',
                                 'GIN',
                                 #  'AttentiveFP',
                                 'ChemProp DMPNN',
                                 'SVM',
                                 'XGBoost',
                                 'RF',
                             ],
                             default=['MPNN']
                             )
    # target_name = SelectField(u'Target',
    #                           choices=[
    #                               'VEGFR-1',
    #                               'VEGFR-2',
    #                               'VEGFR-3',
    #                               'EGFR',
    #                               'Anaplastic Lymphoma Kinase',
    #                               'C Src Tyrosine Kinase',
    #                               'Aurora Kinase A',
    #                               'Aurora Kinase B',
    #                               'BRAF',
    #                               'Bromodomain Containing 4',
    #                               'Brutons Tyrosine Kinase',
    #                               'PDGFR-α',
    #                               'PDGFR-β',
    #                               'HDAC-1',
    #                               'HDAC-2',
    #                               'HDAC-6',
    #                               'IGF1R',
    #                               'CHK1',
    #                               'PIK3CA',
    #                               'PIK3CB',
    #                               'PIK3CD',
    #                               'PIK3CG',
    #                               'FGFR-1',
    #                               'FGFR-2',
    #                               'FGFR-3',
    #                               'FLT3',
    #                               'RET Proto Oncogene',
    #                               '11β-HSD1',
    #                               'Cannabinoid Receptor 1',
    #                               'Cathepsin K',
    #                             #   'Farnesyl Protein Transferase',
    #                               'Janus Kinase 1',
    #                               'Janus Kinase 2',
    #                               'Janus Kinase 3',
    #                               'Renin',
    #                               'Cathepsin S',
    #                               'Dipeptidyl Peptidase 4',
    #                               'MMP-3',
    #                               'MMP-9',
    #                               'MMP-13',
    #                             #   'Nuclear Receptor ROR γ T',
    #                             #   'Phosphodiesterase-4',
    #                               'Spleen Tyrosine Kinase',
    #                               'TNF α',
    #                               'AKT-1',
    #                               'AKT-2',
    #                               'AKT-3',
    #                               'Angiotensin Converting Enzyme',  # 不准
    #                               'BCL-2',
    #                               'CDK-1',
    #                               'CDK-2',  # 不准
    #                               'MAP2K1',
    #                               'Mineralocorticoid Receptor',
    #                               'MMP-14',
    #                               'Nicotinamide Phosphoribosyltransferase',
    #                             #   'PIK-3',
    #                             #   'PKA',
    #                               'PKC θ',
    #                               'C-C Chemokine Receptor Type 2',  # C_C_chemokine_receptor_type_2
    #                             #   'C-C Chemokine Receptor Type 3',  # C_C_chemokine_receptor_type_3
    #                               'MMP-2',
    #                               'ZAP-70',
    #                               'Cathepsin B',
    #                           ],
    #                           default='VEGFR-2'
    #                           )
    target_name = SelectField(u'Target',
                              choices=[
                                  '11β-HSD1',
                                  'AKT1',
                                  'AKT2',
                                  'AKT3',
                                  'ALK',
                                  'ACE',
                                #   'Aurora Kinase A',
                                  'AURKA',
                                #   'Aurora Kinase B',
                                  'AURKB',
                                  'BCL2',
                                  'BRAF',
                                  'BRD4',
                                #   'Bruton\'s tyrosine kinase',
                                  'BTK',
                                  'CCR2',
                                #   'c-Src tyrosine kinase',
                                  'c-Src',
                                  'CDK1',
                                  'CDK2',
                                  'CHK1',
                                #   'Cannabinoid receptor 1',
                                  'CB1R',
                                  'Cathepsin B',
                                  'Cathepsin K',
                                  'Cathepsin S',
                                  'DPP-4',
                                  'EGFR',
                                  'FGFR1',
                                  'FGFR2',
                                  'FGFR3',
                                  'FLT3',
                                  'HDAC1',
                                  'HDAC2',
                                  'HDAC6',
                                  'IGF1R',
                                #   'Janus kinase 1',
                                #   'Janus kinase 2',
                                #   'Janus kinase 3',
                                  'JAK1',
                                  'JAK2',
                                  'JAK3',
                                  'MEK1',
                                  'MMP-2',
                                  'MMP-3',
                                  'MMP-9',
                                  'MMP-13',
                                  'MMP-14',
                                #   'Mineralocorticoid receptor',
                                  'MR',
                                  'NAMPT',
                                  'PDGFR-α',
                                  'PDGFR-β',
                                  'PI3K-α',
                                  'PI3K-β',
                                  'PI3K-δ',
                                  'PI3K-γ',
                                  'PKC-θ',
                                  'Renin',
                                #   'Spleen tyrosine kinase',
                                  'SYK',
                                  'TNF-α',
                                  'VEGFR1',
                                  'VEGFR2',
                                  'VEGFR3',
                                  'ZAP70',
                              ],
                              default='VEGFR-2'
                              )
    accurate_target_name = SelectField(u'Accurate Pipline Target',
                                       choices=[
                                           'None',
                                           'VEGFR-2',
                                           'EGFR',
                                           'PI3K-α',
                                           'Spleen tyrosine kinase',
                                           #    'Janus Kinase 3',               # minus小数据不准似乎只能预测准确到0.01
                                           #    'PDGFR-β',                      # minus小数据不准似乎只能预测准确到0.01
                                           #    'C Src Tyrosine Kinase',        # 都不行
                                           #    'Aurora Kinase A',              # 都不行
                                           #    'C-C Chemokine Receptor Type 3',# 都不行
                                           #    'PIK3CD',                       # 都不行
                                           'Cannabinoid Receptor 1',
                                           'IGF1R',
                                       ],
                                       default=None
                                       )
    uom_flag = SelectField(u'Inhibitory activity unit',
                           choices=['PIC50', 'IC50'],
                           default='PIC50'
                           )
    strategy = SelectField(u'Prediction Stratege',
                           choices=['Best', 'Merge'],
                           default='Best'
                           )
    submit = SubmitField('Submit')


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

_MODEL_REVERSE = {v: k for k, v in list(_MODEL_SHOW2BACKEND.items())}

_TARGET_SHOW2BACKEND = {
    'VEGFR1': 'VEGFR1_str_act',
    'VEGFR2': 'VEGFR2_str_act',
    'VEGFR3': 'VEGFR3_str_act',
    'EGFR': 'EGFR_str_act',
    'ALK': 'Anaplastic_lymphoma_kinase_str_act',
    'c-Src': 'C_src_tyrosine_kinase_str_act',
    'AURKB': 'Aurora_kinase_B_str_act',
    'BRAF': 'BRAF_str_act',
    'BRD4': 'Bromodomain_containing_4_str_act',
    'BTK': 'Brutons_tyrosine_kinase_str_act',
    'AURKA':   'Aurora_kinase_A_str_act',
    'PDGFR-α': 'PDGFR_alpha_str_act',
    'PDGFR-β': 'PDGFR_beta_str_act',
    'HDAC1': 'HDAC1_str_act',
    'HDAC2': 'HDAC2_str_act',
    'HDAC6': 'HDAC6_str_act',
    'IGF1R': 'IGF1R_str_act',
    'CHK1': 'CHK1_str_act',
    'PI3K-α': 'PIK3CA_str_act',
    'PI3K-β': 'PIK3CB_str_act',
    'PI3K-δ': 'PIK3CD_str_act',
    'PI3K-γ': 'PIK3CG_str_act',
    'FGFR1': 'FGFR1_str_act',
    'FGFR2': 'FGFR2_str_act',
    'FGFR3': 'FGFR3_str_act',
    'FLT3': 'FLT3_str_act',
    'RET Proto Oncogene': 'RET_proto_oncogene_str_act',
    '11β-HSD1': '11_beta_HSD1_str_act',
    'CB1R': 'Cannabinoid_receptor_1_str_act',
    'Cathepsin K': 'Cathepsin_K_str_act',
    # 'Farnesyl Protein Transferase': 'Farnesyl_protein_transferase_str_act',
    'JAK1': 'Janus_kinase_1_str_act',
    'JAK2': 'Janus_kinase_2_str_act',
    'Renin': 'Renin_str_act',
    'Cathepsin S': 'Cathepsin_S_str_act',
    'DPP-4': 'Dipeptidyl_peptidase_4_str_act',
    'JAK3': 'Janus_kinase_3_str_act',
    'MMP-13': 'MMP_13_str_act',
    'MMP-3': 'MMP_3_str_act',
    'MMP-9': 'MMP_9_str_act',
    'Nuclear Receptor ROR γ T': 'Nuclear_receptor_ROR_gamma_T_str_act',
    # 'Phosphodiesterase-4': 'Phosphodiesterase_4_str_act',
    'SYK': 'Spleen_tyrosine_kinase_str_act',
    'TNF-α': 'TNF_alpha_str_act',
    'AKT1': 'AKT1_str_act',
    'AKT2': 'AKT2_str_act',
    'AKT3': 'AKT3_str_act',
    'ACE': 'Angiotensin_converting_enzyme_str_act',
    'BCL2': 'BCL2_str_act',
    'CDK1': 'CDK1_str_act',
    'CDK2': 'CDK2_str_act',
    'MEK1': 'MAP2K1_str_act',
    'MR': 'Mineralocorticoid_receptor_str_act',
    'MMP-14': 'MMP_14_str_act',
    'NAMPT': 'Nicotinamide_phosphoribosyltransferase_str_act',
    'PIK-3': 'PIK3_str_act',
    'PKA': 'PKA_str_act',
    'PKC-θ': 'PKC_theta_str_act',
    'CCR2': 'C_C_chemokine_receptor_type_2_str_act',
    # 'C-C Chemokine Receptor Type 3': 'C_C_chemokine_receptor_type_3_str_act',
    'MMP-2': 'MMP_2_str_act',
    'ZAP70': 'ZAP70_str_act',
    'Cathepsin B': 'Cathepsin_B_str_act',
    # 'Mineralocorticoid Receptor': 'Mineralocorticoid_receptor_str_act'
}

def group_info(df):
    data = [row[~pd.isna(row)].tolist() for idx, row in df.iterrows()]
    grouped_data = []
    n_batches = len(data) // 2
    emb_iter = iter(data)
    for i in range(n_batches + 1):
        one_group = []
        batch = list(islice(emb_iter, 2))
        for one_data in batch:
            one_group += one_data
        if one_group:
            grouped_data.append(one_group)
    return grouped_data

@app.route('/')
def index():
    return render_template('main/index.html')


@app.route('/model_info')
def model_info():
    target_info_path = os.path.join('static','target_data')
    bio_info_df = pd.read_csv(os.path.join(target_info_path, 'model_info_Bioactivity_prediction_dataset.csv'))
    grouped_bio_data = group_info(bio_info_df)
    admet_info_df = pd.read_csv(os.path.join(target_info_path, 'model_info_ADMET_info.csv'))
    grouped_admet_data = group_info(admet_info_df)
    model_performance_df = pd.read_csv(os.path.join(target_info_path, 'model_performance_table3.csv'), header=None)
    grouped_model_performance_data = group_info(model_performance_df)

    return render_template('main/model_info.html', ret={
        'grouped_bio_data': grouped_bio_data,
        'grouped_admet_data': grouped_admet_data,
        'grouped_model_performance_data': grouped_model_performance_data,
        

    })


@app.route('/explanation')
def explanation():
    target_info_path = os.path.join('static','target_data')
    target_name_info_df = pd.read_csv(os.path.join(target_info_path, 'explanation_table5.csv'))
    target_name_info_data =  group_info(target_name_info_df)
    lib_info_df = pd.read_csv(os.path.join(target_info_path, 'implementation.csv'))
    lib_info_data = group_info(lib_info_df)
    return render_template('main/explanation.html', ret={
        'target_name_info_data': target_name_info_data,
        'lib_info_data': lib_info_data,
    })


@app.route('/works')
def works():
    return render_template('works/works.html')


@app.route('/works/bioactivity_prediction')
def works_submit_strategy():
    form = MolForm(request.form)
    return render_template('works/bioactivity_prediction.html', form=form)


# @app.route('/works/bioactivity_prediction_old')
# def works_submit_strategy_old():
#     form = MolForm(request.form)
#     return render_template('works/works_gostar_submit.html', form=form)


@app.route('/works/bioactivity_prediction_results', methods=['GET', 'POST'])
def works_result_strategy():
    admet_predictor = ADMET_predictor()
    device = torch.device('cpu')

    all_list = []
    sdf_path = None

    if request.method == 'POST':

        form = MolForm(request.form, request.files)

        print(form.drawn_smiles.data, form.file.data,
              form.strategy.data, form.smiles.data, form.uom_flag.data, form.accurate_target_name.data)

        if form.drawn_smiles.data:
            all_list.append(form.drawn_smiles.data)

        import re

        if form.smiles.data:
            smiles = re.split(': |, |:|,| |。|，|；|；|\r\n', form.smiles.data)
            logger.info(smiles)
            all_list.extend(smiles)

        if request.files['file']:

            file = request.files['file']
            filename = secure_filename(file.filename)
            if filename and filename.split('.')[-1] == 'sdf':
                sdf_path = os.path.join('upload', filename)
                file.save(sdf_path)
            elif request.files.get('file').filename.split('.')[-1] == 'smi':
                for line in request.files.get('file'):
                    print(line.decode('utf-8').strip())
                    all_list.append(line.decode('utf-8').strip())

    else:
        # form = MolForm(request.form)
        # return render_template('works/works_gostar_results.html', form=form)
        return redirect(url_for('works_submit_strategy'))
    try:
        if sdf_path:
            mol_suppl = Chem.SDMolSupplier(sdf_path)
            os.remove(sdf_path)
            all_list = [Chem.MolToSmiles(mol) for mol in mol_suppl]
            # all_smiles = list(set(all_list))
            all_smiles = list(all_list)
        elif all_list:
            all_smiles = list(all_list)
        else:
            raise ValueError

        logger.info(all_smiles)

        msg_list = [[smi, canonicalize_smiles(smi)] for smi in all_smiles]
        all_smiles = [msg[1] for msg in msg_list if msg[1] not in ['', 'C']]

        err_msg1 = 'This input is not a valid molecule!'
        err_msg2 = 'This molecule is too small. Please change it to a larger One!'
        best = True if form.strategy.data == 'Best' else False
        form.accurate_target_name.data = 'None'
        if form.accurate_target_name.data != 'None':
            pred, _ = accurate_pipeline(input_smiles=all_smiles,
                                        target_name=_TARGET_SHOW2BACKEND[form.accurate_target_name.data], best=best, device=device)
            mode_net_flag_return = ['Accurate Pipline' for _ in pred]
            uom_flag = 'IC50 (μM)'
        else:
            uom = 'PIC50'
            uom_flag = uom

            pred, model_net_flag = merge_predict(input_smiles=all_smiles,
                                                 target_name=_TARGET_SHOW2BACKEND[form.target_name.data], best=best, uom=uom, device=device)
            mode_net_flag_return = []
            for model_flag in model_net_flag:
                names = model_flag.split(',')
                names = [_MODEL_REVERSE[x] for x in names]
                mode_net_flag_return.append(', '.join(names))

        drug_props = calculate_drug_prop(all_smiles)
        admet_props_df = admet_predictor.predict(
            all_smiles, device=torch.device('cpu'))
        print(admet_props_df.head())
        used_ademt_col = ['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD', 'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53'] + ['CYP1A2_ContextPred_classification_prob', 'CPY2D6_ContextPred_classification_prob', 'CYP2D6_substrate_ContextPred_classification_prob', 'CYP2C9_ContextPred_classification_prob',
                                                                                                                                                                        'CYP2C19_ContextPred_classification_prob', 'CYP3A4_ContextPred_classification_prob', 'CYP3A4_substrate_CNN_classification_prob', 'HIA_AttrMasking_classification_prob', 'BBB_martins_ContextPred_classification_prob', 'half_life_obach_AttrMasking_classification_prob', 'pgp_broccatelli_AttrMasking_classification_prob', 'LogD_regression_y', 'LogS_regression_y']
        admet_props_df = admet_props_df[used_ademt_col]
        logging.info(admet_props_df.shape)
        valid_index = 0
        for _idx, msg in enumerate(msg_list):
            if msg[0] == 'C':
                msg.append(err_msg2)
                msg.append('')
                for prop_name in drug_props.keys():
                    msg.append('')
            elif msg[1] == '' and msg[0] != 'C':
                msg.append(err_msg1)
                msg.append('')
                for prop_name in drug_props.keys():
                    msg.append('')
            else:
                msg.append('{:.4f}'.format(pred[valid_index]))
                msg.append('{}'.format(mode_net_flag_return[valid_index]))
                for prop_name in drug_props.keys():
                    msg.append('{:.4f}'.format(drug_props[prop_name][valid_index]))
                valid_index += 1
        assert len(pred) == valid_index
        if uom_flag == 'PIC50':
            uom_flag = 'pIC50'
        title_to_csv = [
            'Molecules',
            f'Predicted {uom_flag}',
            # 'Used Models',
            'QED',
            'SAScore',
            'LogP',
            'TPSA',
            'MolWt'
        ]
        admet_props_names = [admet_props_df.columns.tolist()][0]
        title_to_csv += admet_props_names
        result_to_csv = list(zip(
            [msg[0] for msg in msg_list],  # SMILES
            [msg[2] for msg in msg_list],  # predicted value (PIC50 or IC50)
            # [msg[3] for msg in msg_list],  # model msg
            [msg[4] for msg in msg_list],  # QED
            [msg[5] for msg in msg_list],  # SAScore
            [msg[6] for msg in msg_list],  # LogP
            [msg[7] for msg in msg_list],  # TPSA
            [msg[8] for msg in msg_list],  # MolWt

        ))
        def isfloat(x):
            try:
                float(x)
                return True
            except:
                return False
        target_name_to_save = form.target_name.data if form.accurate_target_name.data == 'None' else form.accurate_target_name.data
        valid_index_list = [i for i in range(len(result_to_csv)) if isfloat(result_to_csv[i][1])]
        invalid_index_list = [i for i in range(len(result_to_csv)) if not isfloat(result_to_csv[i][1])]
        admet_props_df.index = valid_index_list
        for invalid_idx in invalid_index_list:
            admet_props_df.loc[invalid_idx] = ['']*admet_props_df.shape[1]
        admet_props_df = admet_props_df.loc[list(range(len(admet_props_df)))]
        admet_props_df = admet_props_df.round(3)
        result_to_csv = np.concatenate([np.asarray(result_to_csv), np.asarray(list(
            admet_props_df[name].tolist() for name in admet_props_names)).T], axis=1).tolist()
        # print(list(zip(admet_props[name] for name in admet_props_names)))
        csv_id = str(random.randint(0, 9999))
        csv_url = os.path.join('static', 'csv', '{}_{}.csv'.format(target_name_to_save, csv_id))
        csv_file = os.path.join(os.path.dirname(__file__), csv_url)
        logger.info('csv_file: ' + csv_file)

        with open(csv_file, 'w', encoding='UTF8') as f:
            # create the csv writer
            writer = csv.writer(f)
            writer.writerow(title_to_csv)
            for r in result_to_csv:
                writer.writerow(r)

        def get_svg_name(x, y):
            if y:
                return x
            else:
                return 'emoji-frown.svg'
        molsvg_path = os.path.join('static', 'molsvg')
        if not os.path.exists(molsvg_path):
            os.makedirs(molsvg_path)
        single_id = [random.randint(0, 9999) for _ in range(len(msg_list))]
        svg_fig_names = [f'{idx}.svg' for idx in single_id]
        svg_fig_names = [get_svg_name(name, smi)
                         for name, smi in zip(svg_fig_names, [msg[1] for msg in msg_list])]
        mol_list = [Chem.MolFromSmiles(x)
                    for x in [msg[1] for msg in msg_list]]
        save_fig_path = [os.path.join(molsvg_path, name)
                         for name in svg_fig_names]
        [mol2svg(mol, path) for mol, path in zip(mol_list, save_fig_path)]
        
        

        
        title_to_show = [
            'SMILES',
            'Molecules',
            f'Predicted {uom_flag}',
            # 'Used Models',
            # 'QED',
            # 'SAScore',
            # 'LogP',
            # 'TPSA',
            # 'MolWt'
        ]
        result_to_show = list(zip(
            # svgmsg_list, # svg msg
            [msg[0] for msg in msg_list],  # SMILES
            svg_fig_names,  # svg name
            [msg[2] for msg in msg_list],  # predicted value (PIC50 or IC50)
            # [msg[3] for msg in msg_list],  # model msg
            # [msg[4] for msg in msg_list],  # QED
            # [msg[5] for msg in msg_list],  # SAScore
            # [msg[6] for msg in msg_list],  # LogP
            # [msg[7] for msg in msg_list],  # TPSA
            # [msg[8] for msg in msg_list],  # MolWt
        ))
        print(len(result_to_show))

        
        detail_table_path = os.path.join('static', 'detail_csv')
        if not os.path.exists(detail_table_path):
            os.makedirs(detail_table_path)
        detail_title = [
            'target_name',
            'svg_path',
            'smiles',
            'uom_flag',
            'pred_value',
            # 'used_model',
            'qed',
            'sascore',
            'logp',
            'tpsa',
            'molwt',
        ]
        detail_title += admet_props_names

        detail_results = list(zip(
            # svgmsg_list, # svg msg
            [target_name_to_save for _ in msg_list],
            svg_fig_names,  # svg name
            [msg[0] for msg in msg_list],  # SMILES
            [uom_flag for _ in msg_list],  # uom flag
            [msg[2] for msg in msg_list],  # predicted value (PIC50 or IC50)
            # [msg[3] for msg in msg_list],  # model msg
            [msg[4] for msg in msg_list],  # QED
            [msg[5] for msg in msg_list],  # SAScore
            [msg[6] for msg in msg_list],  # LogP
            [msg[7] for msg in msg_list],  # TPSA
            [msg[8] for msg in msg_list],  # MolWt
        ))

        detail_results = np.concatenate([np.asarray(detail_results), np.asarray(list(
            admet_props_df[name].tolist() for name in admet_props_names)).T], axis=1).tolist()

        for idx, r in enumerate(detail_results):
            with open(os.path.join(detail_table_path, '{}.csv'.format(single_id[idx])), 'w', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(detail_title)
                writer.writerow(r)

        return render_template("works/bioactivity_prediction_results.html", ret={
            # 'form': form,
            'error': None,
            'output': result_to_show,
            'csv_id': csv_id,
            'csv_url': csv_url,
            # 'n_mol': len(msg_list),
            'target_name': target_name_to_save,
            'title': title_to_show,
            'uom': uom_flag,
            'svg_name': svg_fig_names,
            'single_id': single_id,
        })
    except Exception as e:

        logger.exception(e)

        return render_template("works/bioactivity_prediction_results.html", ret={
            # 'form': form,
            'error': str(e),
            'output': [],
            # 'csv_id': csv_id,
        })


@app.route('/detail/<table_id>')
def works_show_detail(table_id):
    # print(table_id)
    detail_table_path = os.path.join('static', 'detail_csv')
    if not os.path.exists(detail_table_path):
        os.makedirs(detail_table_path)
    try:
        details = pd.read_csv(os.path.join(
            detail_table_path, '{}.csv'.format(table_id)))
        details.to_excel(os.path.join(
            detail_table_path, '{}.xlsx'.format(table_id)), index=False)
        print(details.head())
        detail_dict = details.to_dict()
        return render_template("works/bioactivity_prediction_results_details.html", ret={
            'error': None,
            'target_name': detail_dict['target_name'][0],
            'svg_path': detail_dict['svg_path'][0],
            'smiles': detail_dict['smiles'][0],
            'uom_flag': detail_dict['uom_flag'][0],
            'pred_value': detail_dict['pred_value'][0],
            # 'used_model': detail_dict['used_model'][0],
            'qed': detail_dict['qed'][0],
            'sascore': detail_dict['sascore'][0],
            'logp': detail_dict['logp'][0],
            'tpsa': detail_dict['tpsa'][0],
            'molwt': detail_dict['molwt'][0],
            'logS': detail_dict['LogS_regression_y'][0],
            'logD': detail_dict['LogD_regression_y'][0],

            # ADMET:
            # Absorption:
            'HIA': detail_dict['HIA_AttrMasking_classification_prob'][0],
            'Pgp': detail_dict['pgp_broccatelli_AttrMasking_classification_prob'][0],
            # Distribution
            'BBB': detail_dict['BBB_martins_ContextPred_classification_prob'][0],
            # Metabolism
            'CYP1A2': detail_dict['CYP1A2_ContextPred_classification_prob'][0],
            'CYP2C9': detail_dict['CYP2C9_ContextPred_classification_prob'][0],
            'CYP2D6': detail_dict['CPY2D6_ContextPred_classification_prob'][0],
            'CYP2D6_substrate': detail_dict['CYP2D6_substrate_ContextPred_classification_prob'][0],
            'CYP2C19': detail_dict['CYP2C19_ContextPred_classification_prob'][0],
            'CYP3A4': detail_dict['CYP3A4_ContextPred_classification_prob'][0],
            'CYP3A4_substrate': detail_dict['CYP3A4_substrate_CNN_classification_prob'][0],
            # Excretion:
            'T': detail_dict['half_life_obach_AttrMasking_classification_prob'][0],
            # Toxicity:
            # Tox21
            'NR_AR': detail_dict['NR-AR'][0],
            'NR_AR_LBD': detail_dict['NR-AR-LBD'][0],
            'NR_AhR': detail_dict['NR-AhR'][0],
            'NR_Aromatase': detail_dict['NR-Aromatase'][0],
            'NR_ER': detail_dict['NR-ER'][0],
            'NR_ER_LBD': detail_dict['NR-ER-LBD'][0],
            'NR_PPAR_gamma': detail_dict['NR-PPAR-gamma'][0],
            'SR_ARE': detail_dict['SR-ARE'][0],
            'SR_ATAD5': detail_dict['SR-ATAD5'][0],
            'SR_HSE': detail_dict['SR-HSE'][0],
            'SR_MMP': detail_dict['SR-MMP'][0],
            'SR_p53': detail_dict['SR-p53'][0],
            # csv_url:
            'csv_url': os.path.join(
            detail_table_path, '{}.xlsx'.format(table_id)),
        })
    except Exception as e:

        logger.exception(e)

        return render_template("works/bioactivity_prediction_results_details.html", ret={
            # 'form': form,
            'error': str(e),
            'detail_dict': dict(),
            # 'csv_id': csv_id,
        })


@app.route('/works/target_prediction')
def works_target_prob_submit():
    form = MolForm(request.form)
    return render_template('works/target_prediction.html', form=form)


@app.route('/works/target_prediction_results', methods=['GET', 'POST'])
def works_target_prob_reults():

    all_list = []
    if request.method == 'POST':

        form = MolForm(request.form, request.files)

        print(form.drawn_smiles.data, form.file.data,
              form.strategy.data, form.smiles.data, form.uom_flag.data, form.accurate_target_name.data)

        if form.drawn_smiles.data:
            all_list.append(form.drawn_smiles.data)

    else:
        # form = MolForm(request.form)
        # return render_template('works/works_target_prob_submit.html', form=form)
        return redirect(url_for('works_target_prob_submit'))

    try:
        single_id = random.randint(0, 9999) 
        while True:
            queue_list = os.listdir(queue_path)
            # print(data)
            if len(queue_list) > 2 :
                continue
            else:
                break
  
        with open(os.path.join(queue_path, f'{single_id}.log'), 'w', encoding='utf-8') as f:
            f.write('wait...')
        smiles = Chem.MolToSmiles(Chem.MolFromSmiles(all_list[0]))
        molsvg_path = os.path.join('static', 'molsvg')
        if not os.path.exists(molsvg_path):
            os.makedirs(molsvg_path)

        mol_fig_path = os.path.join(molsvg_path, f't_{single_id}.svg')
        t_mol = Chem.MolFromSmiles(smiles)
        mol2svg(t_mol, mol_fig_path)
        results = app.moltarget_prob_calculator.predict(all_list)
        results = results.round(5)
        results.head()
        target_info_path = os.path.join('static', 'target_data')
        # target_info_path = os.path.join('static', 'target_data', 'std_name2uniprot_id.json')
        # target_class_path = os.path.join('static', 'target_data', 'our_target_classification.csv')
        target_id_df = pd.read_csv(os.path.join(
            target_info_path, 'name2chembl_id.csv'))
        target_class_df = pd.read_csv(os.path.join(
            target_info_path, 'our_target_classification.csv'))
        # std_name2uniprot_id = {k:v for k,v in zip(target_info_df['standard_name'].tolist(),
        #                                           target_info_df['uniprot_id'].tolist())}
        std_name2class = {k: v for k, v in zip(target_class_df['standard_name'].tolist(),
                                               target_class_df['level_1'].tolist())}
        # std_name2chembl_id = {k:v for k,v in zip(target_id_df['standard_name'].tolist(),
        #                                           target_id_df['target_chembl_id'].tolist())}
        with open(os.path.join(target_info_path, 'std_name2uniprot_id.json'), 'r', encoding='utf-8') as f:
            std_name2uniprot_id = json.load(f)
        std_name_list = results.columns.tolist()
        uniport_ids = [std_name2uniprot_id[x] for x in std_name_list]

        def std_name2info_func(x, info_dict):
            if x in info_dict:
                return info_dict[x]
            return 'Unk'
        t_class = [std_name2info_func(x, std_name2class)
                   for x in std_name_list]

        # chembl_ids = [std_name2info_func(x, std_name2chembl_id) for x in std_name_list]
        h_max = 150
        prob_list = [results[key][0] for key in std_name_list]
        h_list = [max(0.001, x) * h_max for x in prob_list]
        target_prob_pair = list(
            zip(std_name_list,  prob_list, uniport_ids, t_class, h_list))
        logging.info('Prob > 0.95: {}'.format(
            len([x for x in target_prob_pair if x[1] > 0.95])))
        logging.info('Prob > 0.9: {}'.format(
            len([x for x in target_prob_pair if x[1] > 0.9])))
        logging.info('Prob > 0.5: {}'.format(
            len([x for x in target_prob_pair if x[1] > 0.5])))
        target_prob_pair.sort(key=lambda x: x[1], reverse=True)
        target_prob_pair = [x for x in target_prob_pair if x[1] > 0.000001]
        # target_prob_pair = target_prob_pair[:15]
        os.remove(os.path.join(queue_path, f'{single_id}.log'))
        return render_template("works/target_prediction_results.html", ret={
            # 'form': form,
            'error': None,
            'smiles': all_list[0],
            'target_prob_pair': target_prob_pair,
            'single_id': single_id,
            'h_max':h_max,
        })
    except Exception as e:

        logger.exception(e)

        return render_template("works/target_prediction_results.html", ret={
            # 'form': form,
            'error': 'Input is not valid!',
            'output': [],
            # 'csv_id': csv_id,
        })
    # return render_template('works/works_target_prob_results.html', ret={})


@app.route('/help')
def help():
    return render_template('main/help.html')


@app.route('/contact')
def contact():
    return render_template('main/contact.html')


@app.route('/test')
def test():
    return render_template('main/copy_from_me.html')


@app.errorhandler(404)
def page_not_found(e):
    # note that we set the 404 status explicitly
    return render_template('main/error.html'), 404


if __name__ == '__main__':
    from pathlib import Path
    cur_file_path = Path(__file__).resolve().parent  # Path.cwd().parent   #
    app.config['UPLOAD_FOLDER'] = cur_file_path/'upload'
    app.config['MAX_CONTENT_PATH'] = 2**10
    Bootstrap(app)
    app.run(host='0.0.0.0', port=8000, debug=True)
