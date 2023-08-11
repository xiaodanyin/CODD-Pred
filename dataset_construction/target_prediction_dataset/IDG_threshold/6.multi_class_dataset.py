import os
import pandas as pd
from tqdm import tqdm

if __name__ == '__main__':
    dataset = pd.read_csv('./merge_dataset/merge_data_pos_40_neg_30.csv')
    multi_classification_dataset = {
        'smiles': [],
        'class': [],
        'label': [],
    }
    dataset_col = dataset.columns.tolist()
    for idx, column in tqdm(enumerate(dataset_col[:640])):
        sub_dataset = dataset.loc[dataset[column] == 1.0]
        sub_smiles = sub_dataset['smiles'].tolist()
        if len(sub_smiles) == 0:
            print(f'{column} Positive data is 0')
        multi_classification_dataset['smiles'].extend(sub_smiles)
        multi_classification_dataset['class'].extend([column]*len(sub_smiles))
        multi_classification_dataset['label'].extend([idx] * len(sub_smiles))

    multi_classification_dataset_df = pd.DataFrame.from_dict(multi_classification_dataset)
    print(len(multi_classification_dataset_df))
    multi_classification_dataset_path = os.path.join('./merge_dataset/', 'multi_classification_dataset')
    if not os.path.exists(multi_classification_dataset_path):
        os.makedirs(multi_classification_dataset_path)
    multi_classification_dataset_df.to_csv(os.path.join(multi_classification_dataset_path, 'merge_data_pos_40_neg_30_pos_class.csv'), index=False)
