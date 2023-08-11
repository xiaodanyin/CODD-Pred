import pandas as pd
import os
from tqdm import tqdm

if __name__ == "__main__":
    split_dataset_path = os.path.join('./target_data', 'split_data')
    if not os.path.exists(split_dataset_path):
        os.makedirs(split_dataset_path)

    target_df = pd.read_csv('../target_prediction_data/human_target_100p_uom_with_target_class.csv')    
    common_name_set = set(target_df['common_name'].tolist())

    for name in tqdm(common_name_set):
        if pd.isna(name): 
            continue
        df = target_df.loc[target_df['common_name'] == name]
        df.dropna(axis=0, subset=['act_id', 'activity_prefix', 'activity_type', 'activity_uom', 'activity_value', 'assay_id', 'assay_type', 'common_name', 'source', 'standard_name', 'sub_smiles', 'mol_weight', 'str_id', 'gvk_id', 'new_activity_values(uM)'], inplace=True)

        if len(df) < 100: 
            continue

        save_name = name.replace(' ', '_').replace('/', 'or').replace('\\', '_')
        df.to_csv(os.path.join(split_dataset_path, f'human_{save_name}_dataset.csv'), index=False)    