import os
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw, AllChem
from flask import Flask, render_template, request


abs_dir = os.path.dirname(os.path.realpath(__file__))


def svg2file(fname, svg_text):
    with open(fname, 'w') as f:
        f.write(svg_text)


def mol2svg(mol, path):
    # smi = ''.join(smi.split(' '))
    # mol = Chem.MolFromSmiles(smi)
    d = Draw.MolDraw2DSVG(450, 450)
    d.DrawMolecule(mol)
    d.FinishDrawing()
    svg = d.GetDrawingText()
    # svg2 = svg.replace('svg:', '').replace(';fill:#FFFFFF;stroke:none', '').replace('y=\'0.0\'>', 'y=\'0.0\' fill=\'rgb(255,255,255,0)\'>')  # 使用replace将原始白色的svg背景变透明
    # svg2 = svg.replace('svg:', '').replace(';fill:#FFFFFF;stroke:none', 'rgb(255,255,255,0)')
    svg2 = svg.replace('svg:', '')
    svg2file(path, svg2)
    return '\n'.join(svg2.split('\n')[8:-1])


def get_h5_list():
    list_h5_file = []
    return ['None', 'None']
    deepCYP1A2_model_abs_dir = os.path.join(abs_dir, 'ml_model')
    for name in os.listdir(deepCYP1A2_model_abs_dir):
        if '.ckpt' in name:
            list_h5_file.append([name.replace('.ckpt', ''), name])
    return list_h5_file


def generate_smi(file_name, drawn_smiles, smiles):
    target_smi_abs_path = os.path.join(abs_dir, 'upload', file_name)
    with open(target_smi_abs_path, "a+") as f:
        if drawn_smiles:
            f.write(drawn_smiles + "\n")
        if smiles:
            smiles = smiles.replace('\r\n', '\n').replace('\n\n', '\n')
            f.write(smiles)
        f.close()


def handle_file_deepxxx(file_name, xxx='1A2'):
    target_code_abs_path = os.path.join(
        abs_dir, 'ml_model', 'deepxxx', "{}_ecfp4_pred.py".format(xxx))
    target_file_abs_path = os.path.join(abs_dir, 'upload', file_name)
    target_log_abs_path = os.path.join(
        abs_dir, 'ml_model', 'deepxxx', "1A2.log")
    target_output_csv_abs_path = os.path.join(abs_dir, 'ml_model', 'deepxxx', "output",
                                              file_name.replace('.sdf', '').replace('.smi', '') + '.csv')

    os.system('python {} {} {} {}'.format(
        target_code_abs_path,
        target_file_abs_path,
        target_log_abs_path,
        target_output_csv_abs_path
    ))
    output = pd.read_csv(target_output_csv_abs_path, index_col='id')

    os.remove(target_file_abs_path)
    os.remove(target_output_csv_abs_path)

    return output.T.to_dict()


def handle_file_deepames(file_name, model_name):
    target_code_abs_path = os.path.join(
        abs_dir, 'ml_model', 'deepames', "main.py")
    # target_file_abs_path = os.path.join(abs_dir, 'ml_model', 'deepames', "examples", file_name)
    target_file_abs_path = os.path.join(abs_dir, 'upload', file_name)
    target_model_abs_path = os.path.join(
        abs_dir, 'ml_model', 'deepames', 'model', 'DeepAmes.h5')
    target_output_csv_abs_path = os.path.join(abs_dir, 'ml_model', 'deepames', "output",
                                              file_name.replace('.sdf', '').replace('.smi', '') + '.csv')

    cmd = 'python3 {} {} {} {}'.format(
        target_code_abs_path,
        target_file_abs_path,
        target_model_abs_path,
        target_output_csv_abs_path
    )

    print(cmd)
    os.system(cmd)

    output = pd.read_csv(target_output_csv_abs_path, index_col='id')

    os.remove(target_file_abs_path)
    os.remove(target_output_csv_abs_path)

    return output.T.to_dict()


def handle_file_deepames_backup_for_model_name(file_name, model_name):
    target_code_abs_path = os.path.join(
        abs_dir, 'ml_model', 'deepames', "main.py")
    # target_file_abs_path = os.path.join(abs_dir, 'ml_model', 'deepames', "examples", file_name)
    target_file_abs_path = os.path.join(abs_dir, 'upload', file_name)
    target_model_abs_path = os.path.join(
        abs_dir, 'ml_model', 'deepames', "model", model_name+'.h5')
    target_output_csv_abs_path = os.path.join(abs_dir, 'ml_model', 'deepames', "output",
                                              file_name.replace('.sdf', '').replace('.smi', '') + '.csv')

    os.system('python3 {} {} {} {}'.format(
        target_code_abs_path,
        target_file_abs_path,
        target_model_abs_path,
        target_output_csv_abs_path
    ))
    output = pd.read_csv(target_output_csv_abs_path, index_col='id')

    os.remove(target_file_abs_path)
    os.remove(target_output_csv_abs_path)

    return output.T.to_dict()


def main(form):
    pass
