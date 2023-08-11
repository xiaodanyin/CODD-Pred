import pandas as pd
import os
from collections import defaultdict
import numpy as np
from rdkit import Chem
from tqdm import tqdm


def canonicalize_smiles(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is not None:
        return Chem.MolToSmiles(mol)
    else:
        return ''

def get_one_target_dataset(dataset_df, threshold):
    # dataset_df = pd.read_csv(path)
    target_name = dataset_df['common_name'][0]
    canonical_dataset = {
        'common_name': [],
        'standard_name': [],
        'canonical_smiles': [],
        'mol_weight': [],
        'label': [],
        'new_activity_values(uM)': [],
        'act_id': [],
        'assay_id': [],
        'str_id': [],
        'gvk_id': [],
        'target_class': [],
        'level_1': [],
        'level_2': [],
        'level_3': []
    }
    canonical_smi = [canonicalize_smiles(smiles) for smiles in dataset_df['sub_smiles'].tolist()]
    canonical_smi_to_data = defaultdict(dict)
    for i, can_smi in enumerate(canonical_smi):
        if can_smi == '': 
            continue
        common_name = dataset_df['common_name'][i]
        standard_name = dataset_df['standard_name'][i]
        mol_weight = dataset_df['mol_weight'][i]
        values = dataset_df['new_activity_values(uM)'][i]
        act_id = dataset_df['act_id'][i]
        assay_id = dataset_df['assay_id'][i]
        str_id = dataset_df['str_id'][i]
        gvk_id = dataset_df['gvk_id'][i]
        target_class = dataset_df['target_class'][i]
        level_1 = dataset_df['level_1'][i]
        level_2 = dataset_df['level_2'][i]
        level_3 = dataset_df['level_3'][i]

        if 'common_name' in canonical_smi_to_data[can_smi]:
            canonical_smi_to_data[can_smi]['common_name'].append(common_name)
            canonical_smi_to_data[can_smi]['standard_name'].append(standard_name)
            canonical_smi_to_data[can_smi]['mol_weight'].append(mol_weight)
            canonical_smi_to_data[can_smi]['new_activity_values(uM)'].append(values)
            canonical_smi_to_data[can_smi]['act_id'].append(act_id)
            canonical_smi_to_data[can_smi]['assay_id'].append(assay_id)
            canonical_smi_to_data[can_smi]['str_id'].append(str_id)
            canonical_smi_to_data[can_smi]['gvk_id'].append(gvk_id)
            canonical_smi_to_data[can_smi]['target_class'].append(target_class)
            canonical_smi_to_data[can_smi]['level_1'].append(level_1)
            canonical_smi_to_data[can_smi]['level_2'].append(level_2)
            canonical_smi_to_data[can_smi]['level_3'].append(level_3)

        else:
            canonical_smi_to_data[can_smi] = {
                'common_name': [common_name],
                'standard_name': [standard_name],
                'mol_weight': [mol_weight],
                'new_activity_values(uM)': [values],
                'act_id': [act_id],
                'assay_id': [assay_id],
                'str_id': [str_id],
                'gvk_id': [gvk_id],
                'target_class': [target_class],
                'level_1': [level_1],
                'level_2': [level_2],
                'level_3': [level_3]                
            }

    # print(canonical_smi_to_data)
    new_canonical_smi_to_data = {}

    len_neg = 0
    len_pos = 0
    for can_smi in canonical_smi_to_data:
        data = canonical_smi_to_data[can_smi]
        values = data['new_activity_values(uM)']
        flags = []
        for v in values:
            if v > threshold:
                flags.append(0)
            elif v <= threshold:
                flags.append(1)
        flags = np.array(flags)
        if flags.sum() == len(values):
            label = 1
            len_pos += 1
        elif flags.sum() == 0:
            label = 0
            len_neg += 1
        else:
            label = -100

        if label != -100:
            data['label'] = label
            # new_canonical_smi_to_data[can_smi] = data
            canonical_dataset['common_name'].append(data['common_name'][0])
            canonical_dataset['standard_name'].append(data['standard_name'][0])
            canonical_dataset['mol_weight'].append(data['mol_weight'][0])
            canonical_dataset['canonical_smiles'].append(can_smi)
            canonical_dataset['label'].append(label)
            canonical_dataset['act_id'].append(';'.join([str(x) for x in data['act_id']]))
            canonical_dataset['assay_id'].append(';'.join([str(x) for x in data['assay_id']]))
            canonical_dataset['str_id'].append(';'.join([str(x) for x in data['str_id']]))
            canonical_dataset['gvk_id'].append(';'.join([str(x) for x in data['gvk_id']]))
            canonical_dataset['target_class'].append(';'.join([str(x) for x in data['target_class']]))
            canonical_dataset['level_1'].append(';'.join([str(x) for x in data['level_1']]))
            canonical_dataset['level_2'].append(';'.join([str(x) for x in data['level_2']]))
            canonical_dataset['level_3'].append(';'.join([str(x) for x in data['level_3']]))
            canonical_dataset['new_activity_values(uM)'].append(';'.join([str(x) for x in data['new_activity_values(uM)']]))

    # print(new_canonical_smi_to_data)
    print(f'{target_name}: Positive / Negative = {len_pos}/{len_neg}')
    canonical_dataset = pd.DataFrame.from_dict(canonical_dataset)
    return canonical_dataset, len_pos, len_neg, target_name

if __name__ == "__main__":
    '''
    IDG values
    protein kinases: <= 30nM   0.03uM
    GPCRs: <= 100nM   0.1uM
    Nuclear Receptors: <= 100nM   0.1uM
    Ion Channels:   <= 10μM
    Non-IDG Family Targets:  <= 1μM
    '''

    split_data_path = os.path.join('./target_data', 'split_data')
    file_names = os.listdir(split_data_path)

    for fname in tqdm(file_names):
        if fname.split('.')[-1] != 'csv':
            print(f'!!!!!!!!!!!!!!!!!!!!!!!!{fname}!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            continue

        df = pd.read_csv(os.path.join(split_data_path, fname))

        if df['level_1'][0] == 'Ion channel':
            threshold = 10
        elif df['level_2'][0] == 'Family A G protein-coupled receptor':
            threshold = 0.1
        elif df['level_2'][0] == 'Nuclear receptor':
            threshold = 0.1
        elif df['level_3'][0] == 'Protein Kinase':
            threshold = 0.03 
        else:
            threshold = 1

        print(f'threshold: {threshold}')

        split_can_data_path = os.path.join('./target_data_pos_40_neg_30', f'split_can_data_threshold_{threshold}')
        if not os.path.exists(split_can_data_path):
            os.makedirs(split_can_data_path)    

        canonical_dataset, len_pos, len_neg, target_name = get_one_target_dataset(df, threshold=threshold)

        if len_pos >= 40 and len_neg >= 30:
            canonical_dataset.to_csv(os.path.join(split_can_data_path, fname), index=False)
        else:
            print(f'Remove dataset {target_name}')            


