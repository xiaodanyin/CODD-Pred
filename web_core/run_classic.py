import os
import os.path as osp

import torch
from tqdm import tqdm

try:
    from utils import get_fps_from_dataset, regression_metrics, cal_r2_score, seed_torch
    from target_dataset import TargetDataset
except:
    from web_core.utils import get_fps_from_dataset, regression_metrics
    from web_core.target_dataset import TargetDataset
from sklearn.ensemble import RandomForestRegressor

from sklearn.svm import SVR

import xgboost as xgb


def get_fps_from_dataset(pyg_dataset):
    fps, y = [], []
    for data in tqdm(pyg_dataset):
        fps.append(data.fp)
        y.append(data.y)

    fps = torch.cat(fps)
    y = torch.cat(y)

    return fps.numpy(), y.numpy()


def cal_r2_score_xgb(y_predicted, y_true):
    results = cal_r2_score(torch.tensor(y_true.get_label(), dtype=torch.float),
                           torch.tensor(y_predicted, dtype=torch.float))
    return 'r2', results


if __name__ == '__main__':


    seed_torch(123)
    debug = False

    target_data_space_name = 'MMP_2_str_act'
    model = 'xgboost'
    classic_model_path = './model_classic/'

    to_pic50 = False
    data_layered_flag = 'minus'  # > 1 um : plus, <= 1um : minus  to_pic50 = False 生效

    if to_pic50:
        target_name = f'{target_data_space_name}'
        data_path = 'dataset_path'

    else:
        target_name = f'{target_data_space_name}_{data_layered_flag}'
        data_path = 'dataset_path_layered'

    path = osp.join(osp.dirname(osp.realpath(__file__)), data_path)
    dataset = TargetDataset(root=path,
                            target_data_space_name=target_data_space_name,
                            target_name=target_name,
                            to_pic50=to_pic50,
                            # target_col umns_name='new_activity_values(uM)' if not us_nm else 'new_activity_values(nM)',
                            target_columns_name='new_activity_values(uM)',
                            transform=None)
    y_max = dataset.data.y.max()
    y_min = dataset.data.y.min()
    y_std = dataset.data.y.std()
    y_mean = dataset.data.y.mean()

    if not to_pic50:
        maxy_miny_gap = y_max - y_min
        dataset.data.y = (dataset.data.y - y_min) / maxy_miny_gap
        dataset.train.data.y = (
                                           dataset.train.data.y - y_min) / maxy_miny_gap  ####只需要在train中归一化y train val test的data类是同一个

    test_dataset = dataset.test
    val_dataset = dataset.val
    train_dataset = dataset.train

    if debug:
        test_dataset = dataset.test[:500]
        val_dataset = dataset.val[:500]
        train_dataset = dataset.train[:1000]

    trainfp_dataset = get_fps_from_dataset(train_dataset)
    valfp_dataset = get_fps_from_dataset(val_dataset)
    testfp_dataset = get_fps_from_dataset(test_dataset)

    if not os.path.exists(classic_model_path):
        os.mkdir(classic_model_path)

    uom_flag = 'IC50' if not to_pic50 else 'PIC50'
    model_save_fname = os.path.join(classic_model_path, f'{target_name}_{uom_flag}_{model}_train.pkl')

    if not os.path.exists(model_save_fname):
        print('Training..')

        if model == 'rf':
            model = RandomForestRegressor(n_estimators=200, random_state=0)
            model.fit(trainfp_dataset[0], trainfp_dataset[1])
        elif model == 'svm':
            model = SVR(kernel='rbf', verbose=True, C=1.0, epsilon=0.01, gamma=0.01)
            model.fit(X=trainfp_dataset[0], y=trainfp_dataset[1])
        elif model == 'xgboost':
            model = xgb.XGBRegressor(n_estimators=1000, learning_rate=0.2, min_child_weight=5, max_depth=4)
            model.fit(X=trainfp_dataset[0], y=trainfp_dataset[1],
                      # early_stopping_rounds=10,
                      eval_metric=cal_r2_score_xgb,
                      eval_set=[(valfp_dataset[0], valfp_dataset[1])],
                      verbose=True)
        torch.save(model, model_save_fname)
    else:
        print(f'Loading model from {model_save_fname}.')
        model = torch.load(model_save_fname)

    print(regression_metrics(torch.tensor(testfp_dataset[1], dtype=torch.float),
                             torch.tensor(model.predict(testfp_dataset[0]), dtype=torch.float)))
