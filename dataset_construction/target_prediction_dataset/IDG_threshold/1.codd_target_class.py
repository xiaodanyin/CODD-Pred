import pandas as pd
import os
import json


if __name__ == "__main__":
    chembl_class = pd.read_csv('./chembl_target_classification1.csv')

    with open('./std_name2uniprot_id.json', 'r', encoding='utf-8') as f:
        std_name2uniprot_id_dict = json.load(f)
    uniprot_id2std_name_dict = {v: k for k, v in list(std_name2uniprot_id_dict.items())}

    our_target_class_dict = {
        'standard_name': [],
        'uniprot_id':[],
        'level_1':[],
        'level_2':[],
        'level_3':[],
    }    

    for uniprot_id, level_1, level_2, level_3 in zip(
        chembl_class['accession'].tolist(),
        chembl_class['l1'].tolist(),
        chembl_class['l2'].tolist(),
        chembl_class['l3'].tolist(),
    ):
        if uniprot_id in our_target_class_dict['uniprot_id']:
            continue
        our_target_class_dict['standard_name'].append(uniprot_id2std_name_dict[uniprot_id])
        our_target_class_dict['uniprot_id'].append(uniprot_id)
        our_target_class_dict['level_1'].append(level_1)
        our_target_class_dict['level_2'].append(level_2)
        our_target_class_dict['level_3'].append(level_3)

    our_target_class_df = pd.DataFrame.from_dict(our_target_class_dict)
    our_target_class_df.to_csv('./our_target_classification_level1_3.csv', index=False)
