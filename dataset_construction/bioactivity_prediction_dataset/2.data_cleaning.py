import os
from tqdm import tqdm
import pandas as pd
from rdkit import Chem


def uom_to_uM(activity_uom, activity_value, mol_weight):
    import math

    if 'M' in activity_uom:

        if 'p' in activity_uom:
            return math.pow(10, -6) * activity_value
        elif 'n' in activity_uom:
            return math.pow(10, -3) * activity_value
        elif 'm' in activity_uom:
            return math.pow(10, 3) * activity_value
        elif activity_uom == 'M':
            return math.pow(10, 6) * activity_value
        else:
            return activity_value

    elif 'g' in activity_uom:

        if 'u' in activity_uom:
            return math.pow(10, 3) * activity_value / mol_weight
        elif 'n' in activity_uom:
            return activity_value / mol_weight
        else:
            return math.pow(10, 9) * activity_value / mol_weight

    else:

        if activity_uom == 'mol/L':
            return math.pow(10, 6) * activity_value
        elif activity_uom == 'umol/L':
            return activity_value
        elif activity_uom == 'nmol/L':
            return math.pow(10, -3) * activity_value
        elif activity_uom == 'pmol/L':
            return math.pow(10, -6) * activity_value

def canonicalize_smiles(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is not None:
        return Chem.MolToSmiles(mol)
    else:
        return ''

if __name__ == '__main__':

    # Replace the name to clean the bioactivity data of small molecules with different targets
    
    target_name = "11-beta-HSD1_str_act"

    target_data_space = os.path.join('./small_molecule_str_act_dataset', target_name)
    if not os.path.exists(target_data_space):
        os.makedirs(target_data_space)

    structure_activity = pd.read_csv(os.path.join(f'./target_small_molecule_str_act_dataset_org/{target_name}.csv'),
                                     encoding='UTF-8')

    structure_activity_ic50 = structure_activity.loc[structure_activity['activity_type'] == 'IC50']
    structure_activity_ic50 = structure_activity_ic50.loc[pd.isnull(structure_activity_ic50['activity_uom']) == False]
    structure_activity_ic50 = structure_activity_ic50.loc[(structure_activity_ic50['activity_uom'] != '%') & \
                                                          (structure_activity_ic50['activity_uom'] != '1/min') & \
                                                          (structure_activity_ic50['activity_uom'] != '1/(M.s)') & \
                                                          (structure_activity_ic50['activity_uom'] != '1/(uM.s)') & \
                                                          (structure_activity_ic50['activity_uom'] != 'day') & \
                                                          (structure_activity_ic50['activity_uom'] != 'min') & \
                                                          (structure_activity_ic50['activity_uom'] != '%ID/g') & \
                                                          (structure_activity_ic50['activity_uom'] != 'uM.min') & \
                                                          (structure_activity_ic50['activity_uom'] != '1/(M.min)') & \
                                                          (structure_activity_ic50['activity_uom'] != '1/s') & \
                                                          (structure_activity_ic50['activity_uom'] != 'mm3') & \
                                                          (structure_activity_ic50['activity_uom'] != 'mol/min/mol') & \
                                                          (structure_activity_ic50['activity_uom'] != '% dose/mg protein') & \
                                                          (structure_activity_ic50['activity_uom'] != 'kcal/mol') & \
                                                          (structure_activity_ic50['activity_uom'] != '1/ms') & \
                                                          (structure_activity_ic50['activity_uom'] != 'h') & \
                                                          (structure_activity_ic50['activity_uom'] != 'L/mol/min') & \
                                                          (structure_activity_ic50['activity_uom'] != '%ID/mg') & \
                                                          (structure_activity_ic50['activity_uom'] != '%ID') & \
                                                          (structure_activity_ic50['activity_uom'] != 'mg/kg')
                                                          ]

    structure_activity_ic50 = structure_activity_ic50.loc[(structure_activity_ic50['activity_prefix'] != '>') & \
                                                          (structure_activity_ic50['activity_prefix'] != '>=') & \
                                                          (structure_activity_ic50['activity_prefix'] != '~') & \
                                                          (structure_activity_ic50['activity_prefix'] != '>R') & \
                                                          (structure_activity_ic50['activity_prefix'] != '>=R') & \
                                                          (structure_activity_ic50['activity_prefix'] != '<~') & \
                                                          (structure_activity_ic50['activity_prefix'] != '<R') & \
                                                          (structure_activity_ic50['activity_prefix'] != '>~') & \
                                                          (structure_activity_ic50['activity_prefix'] != 'R') & \
                                                          (structure_activity_ic50['activity_prefix'] != '<=C') & \
                                                          (structure_activity_ic50['activity_prefix'] != '<=R')
                                                          ]

    structure_activity_ic50 = structure_activity_ic50.loc[structure_activity_ic50['assay_type'] == 'B']
    print(len(structure_activity_ic50))

    new_activity_values_list = []
    for activity_uom, activity_value, mol_weight in zip(structure_activity_ic50['activity_uom'].tolist(),
                                                        structure_activity_ic50['activity_value'].tolist(),
                                                        structure_activity_ic50['mol_weight'].tolist()):
        new_activity_value = uom_to_uM(activity_uom, activity_value, mol_weight)
        new_activity_values_list.append(new_activity_value)
    assert len(new_activity_values_list) == len(structure_activity_ic50['activity_value'])
    structure_activity_ic50['new_activity_values(uM)'] = new_activity_values_list

    structure_activity_ic50 = structure_activity_ic50.loc[(structure_activity_ic50['new_activity_values(uM)'] <= 500)]

    gvk_id_list = structure_activity_ic50['gvk_id'].tolist()
    str_id_list = structure_activity_ic50['str_id'].tolist()
    smiles_list = structure_activity_ic50['smiles'].tolist()
    ic50_values = structure_activity_ic50['new_activity_values(uM)'].tolist()

    new_gvk_id_list = []
    new_str_id_list = []
    new_smiles_list = []
    new_ic50_values = []

    for gvk_id, str_id, smiles, ic50_value in tqdm(zip(gvk_id_list, str_id_list, smiles_list, ic50_values), total=len(smiles_list)):
        if not pd.isna(smiles):
            smiles_can = canonicalize_smiles(smiles)

            if smiles_can is not '':
                if smiles_can in new_smiles_list:
                    idx = new_smiles_list.index(smiles_can)
                    ic50_saved = new_ic50_values[idx]
                    if ic50_value < ic50_saved:
                        new_ic50_values[idx] = ic50_value
                    else:
                        continue
                else:
                    new_gvk_id_list.append(gvk_id)
                    new_str_id_list.append(str_id)
                    new_smiles_list.append(smiles_can)
                    new_ic50_values.append(ic50_value)
            else:
                continue

    assert len(new_smiles_list) == len(new_ic50_values)
    assert len(new_smiles_list) == len(new_gvk_id_list)
    assert len(new_smiles_list) == len(new_str_id_list)

    print(len(new_smiles_list))
    new_dataset = {
        'gvk_id': new_gvk_id_list,
        'str_id': new_str_id_list,
        'smiles': new_smiles_list,
        'new_activity_values(uM)': new_ic50_values
    }

    new_dataset = pd.DataFrame(new_dataset)
    new_dataset.to_csv(os.path.join(target_data_space, f'{target_name}_um_can.csv'), index=False)