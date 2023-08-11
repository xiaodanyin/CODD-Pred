import os

import pandas as pd

if __name__ == '__main__':
    std_name2label = {}
    chemprop_data_path = os.path.join(
        './merge_dataset/multi_classification_dataset/merge_data_pos_40_neg_30_chemprop')
    if not os.path.exists(chemprop_data_path):
        os.makedirs(chemprop_data_path)
    for dataset_name in ['train', 'val', 'test']:
        dataset_df = pd.read_csv(
            os.path.join('./merge_dataset/multi_classification_dataset/merge_data_pos_40_neg_30/',
                         f'{dataset_name}.csv'))
        std_name_list = dataset_df['class'].tolist()
        label_list = dataset_df['label'].tolist()
        smiles_list = dataset_df['smiles'].tolist()
        for std_name, label in zip(std_name_list, label_list):
            std_name2label[std_name] = label

        chemprop_dataset = pd.DataFrame({
            'smiles': smiles_list,
            'label': label_list,
        })
        print(len(chemprop_dataset))
        chemprop_dataset.to_csv(os.path.join(chemprop_data_path, f'{dataset_name}.csv'), index=False)
    std_name2label_item = list(std_name2label.items())
    std_name2label_item.sort(key=lambda x: x[1])
    all_std_name_list = [x[0] for x in std_name2label_item]
    all_label_list = [x[1] for x in std_name2label_item]
    std_2label_df = pd.DataFrame({
        'std_name': all_std_name_list,
        'label': all_label_list
    })
    std_2label_df.to_csv(os.path.join(chemprop_data_path, 'std_name2label.txt'), index=False, header=None)
