import os
from collections import defaultdict

import numpy as np

import pandas as pd
from rdkit.Chem.Scaffolds import MurckoScaffold
from tqdm import tqdm


def generate_scaffold(smiles, include_chirality=False):
    """
    Obtain Bemis-Murcko scaffold from smiles
    :param smiles:
    :param include_chirality:
    :return: smiles of scaffold
    """
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(
        smiles=smiles, includeChirality=include_chirality)
    return scaffold


def random_scaffold_split(dataframe, frac_train=0.8, frac_valid=0.1, frac_test=0.1, seed=0):
    np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.0)
    rng = np.random.RandomState(seed)
    all_label = list(set(dataframe['label']))
    all_label.sort()
    train, val, test = [], [], []
    for label in tqdm(all_label):
        sub_dataset = dataframe.loc[dataframe['label'] == label].reset_index()
        smiles_list = sub_dataset['smiles'].tolist()
        scaffolds = defaultdict(list)
        for ind, smiles in enumerate(smiles_list):
            scaffold = generate_scaffold(smiles, include_chirality=True)
            scaffolds[scaffold].append(ind)

        # scaffold_sets = rng.permutation(list(scaffolds.values()))
        scaffolds_value_ls = list(scaffolds.values())
        max_length = max(len(sublist) for sublist in scaffolds_value_ls)
        scaffold_sets = rng.permutation([sublist + [None] * (max_length - len(sublist)) for sublist in scaffolds_value_ls])

        n_total_valid = int(np.floor(frac_valid * len(sub_dataset)))   
        n_total_test = int(np.floor(frac_test * len(sub_dataset)))

        train_idx = []
        valid_idx = []
        test_idx = []

        for scaffold_set in scaffold_sets:

            scaffold_set = scaffold_set[scaffold_set != None]
            scaffold_set_len = len([x for x in scaffold_set if x != None])

            # if len(valid_idx) + len(scaffold_set) <= n_total_valid:
            if len(valid_idx) + scaffold_set_len <= n_total_valid:
                valid_idx.extend(scaffold_set)
            # elif len(test_idx) + len(scaffold_set) <= n_total_test:
            elif len(test_idx) + scaffold_set_len <= n_total_test:
                test_idx.extend(scaffold_set)
            else:
                train_idx.extend(scaffold_set)

        sub_train_dataset = sub_dataset.loc[train_idx]
        sub_valid_dataset = sub_dataset.loc[valid_idx]
        sub_test_dataset = sub_dataset.loc[test_idx]

        train.append(sub_train_dataset)
        val.append(sub_valid_dataset)
        test.append(sub_test_dataset)
    train_dataset = pd.concat(train)
    valid_dataset = pd.concat(val)
    test_dataset = pd.concat(test)

    return train_dataset, valid_dataset, test_dataset


if __name__ == '__main__':
    multi_cls_dataset = pd.read_csv('./merge_dataset/multi_classification_dataset/merge_data_pos_40_neg_30_pos_class.csv')
    print(len(multi_cls_dataset))
    
    train_dataset, valid_dataset, test_dataset =  random_scaffold_split(multi_cls_dataset, frac_train=0.8,
                                                                       frac_valid=0.1, frac_test=0.1, seed=0)
    split_dataset_path = os.path.join('./merge_dataset/multi_classification_dataset', 'merge_data_pos_40_neg_30')
    if not os.path.exists(split_dataset_path):
        os.makedirs(split_dataset_path)
    train_dataset = train_dataset.sample(frac=1)
    valid_dataset = valid_dataset.sample(frac=1)
    test_dataset = test_dataset.sample(frac=1)
    print(len(train_dataset))
    print(len(valid_dataset))
    print(len(test_dataset))
    
    train_dataset.to_csv(os.path.join(split_dataset_path, 'train.csv'), index=False)
    valid_dataset.to_csv(os.path.join(split_dataset_path, 'val.csv'), index=False)
    test_dataset.to_csv(os.path.join(split_dataset_path, 'test.csv'), index=False)