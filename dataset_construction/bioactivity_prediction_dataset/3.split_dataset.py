import math
import os
import numpy as np
import pandas as pd

def split_value(values, min, max):
    # 指数
    values = np.asarray(values)
    ind = ((values > math.pow(10, min)) & (values <= math.pow(10, max)))
    return ind


def show_dist(values):
    exponents = [-99] + list(range(-3, 4))
    log_str = ''
    for i in range(len(exponents) - 1):
        print('10^{} ~ 10^{} um: {:.3f}%, cnt:{}'.format(exponents[i], exponents[i + 1],
                                                         100 * split_value(values, exponents[i],
                                                         exponents[i + 1]).sum() / len(values),
                                                         split_value(values, exponents[i], exponents[i + 1]).sum()))
        log_str += '10^{} ~ 10^{} um: {:.3f}%, cnt:{}\n'.format(exponents[i], exponents[i + 1],
                                                                100 * split_value(values, exponents[i],
                                                                exponents[i + 1]).sum() / len(values),
                                                                split_value(values, exponents[i],exponents[i + 1]).sum())
    return [split_value(values, exponents[i], exponents[i + 1]) for i in range(len(exponents) - 1)], log_str


def split_train_val_test(df):
    split_1 = int(0.1 * len(df))
    split_2 = int(0.2 * len(df))
    test = df[:split_1]
    val = df[split_1:split_2]
    train = df[split_2:]

    return train, val, test


def shuffle_data(dataset):
    _index = [i for i in range(len(dataset))]
    random.shuffle(_index)
    return dataset.loc[_index].reset_index(drop=True)


if __name__ == '__main__':
    import random
    random.seed(123)

    target_name = "Aurora kinase B_str_act"

    target_data_space = os.path.join('./small_molecule_str_act_dataset', target_name)

    df = pd.read_csv(os.path.join(target_data_space, f'{target_name}_um_can.csv'))
    # print(len(df))

    df = shuffle_data(df)
    df.to_csv(os.path.join(target_data_space, f'{target_name}_um_can_shuffle.csv'), index=False)
    df = pd.read_csv(os.path.join(target_data_space, f'{target_name}_um_can_shuffle.csv'))

    activity_value = df['new_activity_values(uM)']
    index_list, log_str = show_dist(activity_value)

    df_split_list = [df[index] for index in index_list]

    dataset_split_list = [split_train_val_test(df) for df in df_split_list]

    train_dataset = pd.DataFrame()
    val_dataset = pd.DataFrame()
    test_dataset = pd.DataFrame()
    for train, val, test in dataset_split_list:
        train_dataset = train_dataset.append(train)
        val_dataset = val_dataset.append(val)
        test_dataset = test_dataset.append(test)

    if not os.path.exists(os.path.join(target_data_space, f'{target_name}')):
        os.makedirs(os.path.join(target_data_space, f'{target_name}'))

    train_dataset.to_csv(os.path.join(target_data_space, f'{target_name}', 'train.csv'), index=False)
    val_dataset.to_csv(os.path.join(target_data_space, f'{target_name}', 'val.csv'), index=False)
    test_dataset.to_csv(os.path.join(target_data_space, f'{target_name}', 'test.csv'), index=False)

    train_dataset = shuffle_data(pd.read_csv(os.path.join(target_data_space, f'{target_name}', 'train.csv')))
    val_dataset = shuffle_data(pd.read_csv(os.path.join(target_data_space, f'{target_name}', 'val.csv')))
    test_dataset = shuffle_data(pd.read_csv(os.path.join(target_data_space, f'{target_name}', 'test.csv')))

    train_dataset.to_csv(os.path.join(target_data_space, f'{target_name}', 'train.csv'), index=False)
    val_dataset.to_csv(os.path.join(target_data_space, f'{target_name}', 'val.csv'), index=False)
    test_dataset.to_csv(os.path.join(target_data_space, f'{target_name}', 'test.csv'), index=False)

    # with open(os.path.join(target_data_space, f'{target_name}', 'split_log.txt'), 'w', encoding='utf-8') as f:
    #     f.write(log_str)
