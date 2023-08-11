import json
import os
from collections import defaultdict
import pandas as pd
from tqdm import tqdm

if __name__ == '__main__':
    merge_dataset_path = os.path.join('./merge_dataset')
    if not os.path.exists(merge_dataset_path):
        os.makedirs(merge_dataset_path)
    # threshold = 10
    merge_table_name = 'merge_data_pos_40_neg_30.csv'

    split_can_data_path = os.path.join('./target_data_pos_40_neg_30_all_delete_62_target')
    file_names = os.listdir(split_can_data_path)

    smiles2label = defaultdict(dict)
    smiles_set = set()
    for fname in tqdm(file_names):
        canonical_dataset = pd.read_csv(os.path.join(split_can_data_path, fname))
        target_name = canonical_dataset['standard_name'][0]
        # data_dict[target_name] = canonical_dataset
        for smi, label in zip(canonical_dataset['canonical_smiles'].tolist(), canonical_dataset['label'].tolist()):
            smiles_set.add(smi)
            smiles2label[target_name][smi] = label
            
    merge_dataset_dict = {}
    labels = list(smiles2label.keys())
    with open(os.path.join(merge_dataset_path, merge_table_name.replace('.csv', '_tasks.json')), 'w') as f:
        json.dump(labels, f)
        
    smiles_list = []
    data_m_dict = defaultdict(list)
    for smi in tqdm(smiles_set):
        smiles_list.append(smi)
        for i, target_name in enumerate(labels):
            if smi in smiles2label[target_name].keys():
                data_m_dict[target_name].append(smiles2label[target_name][smi])
            else:
                data_m_dict[target_name].append(None)

    merge_dataset_df = pd.DataFrame.from_dict(data_m_dict)
    assert len(smiles_list) == len(merge_dataset_df)
    merge_dataset_df['smiles'] = smiles_list
    print(merge_dataset_df.shape)
    merge_dataset_df.to_csv(os.path.join(merge_dataset_path, merge_table_name), index=False)
