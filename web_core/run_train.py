import os
import os.path as osp
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from collections import defaultdict

try:
    from model import GCN_net, MPNN_net, GAT_net, GIN_net, AttentiveFP_net, DMPNN_Change
    from target_dataset import TargetDataset
    from utils import regression_metrics, cooked_data_to_csv, seed_torch
except:
    from web_core.model import GCN_net, MPNN_net, GAT_net, GIN_net, AttentiveFP_net, DMPNN_Change
    from web_core.target_dataset import TargetDataset
    from web_core.utils import regression_metrics, cooked_data_to_csv, seed_torch


def train(epoch):
    model.train()
    loss_all = 0

    for data in tqdm(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        loss = F.mse_loss(model(data), data.y)
        loss.backward()
        loss_all += loss.item() * data.num_graphs
        optimizer.step()
    return loss_all / len(train_loader.dataset)


def test_model(loader):
    model.eval()
    # error = 0
    ys_true, ys_pred = [], []

    for data in tqdm(loader):
        data = data.to(device)
        # error += (model(data) - data.y).abs().sum().item() # MAE
        output = model(data).view(-1)
        ys_true.append(data.y.cpu())
        ys_pred.append(output.data.cpu())

    result = regression_metrics(torch.cat(ys_true), torch.cat(ys_pred))
    return result


if __name__ == '__main__':

    seed_torch(123)
    debug = True
    save_to_split_csv = True

    target_data_space_name = 'MMP_2_str_act'

    to_pic50 = True
    data_layered_flag = 'minus'  # > 1 um : plus, <= 1um : minus  to_pic50 = False 生效
    model_net = 'MPNN_net'
    train_try_num = '0'
    if to_pic50:
        target_name = f'{target_data_space_name}_{data_layered_flag}'
        uom_flag = 'PIC50'
        data_path = 'dataset_path'
        task_mark = f'{target_name}_{uom_flag}_model_{model_net}_{train_try_num}'
        model_save_path = osp.join(osp.dirname(osp.abspath(__file__)), 'model_pic50')
    else:
        target_name = f'{target_data_space_name}_{data_layered_flag}'
        uom_flag = 'IC50'
        data_path = 'dataset_path_layered'
        task_mark = f'{target_name}_{uom_flag}_model_{model_net}_{train_try_num}'
        model_save_path = osp.join(osp.dirname(osp.abspath(__file__)), 'model_ic50')
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)

    # 模型初始化参数
    hidden_dim = 256
    mess_nn_dim = 128
    out_dim = 1
    gat_heads = 8
    massage_depth = 1
    dropout_rate = 0.3
    use_gru = False
    AttentiveFP_num_layers = 5

    # 模型训练参数
    lr = 0.00005
    batch_size = 128
    epochs = 300

    # 初始化任务标记
    if debug:
        batch_size = 8
        epochs = 10

    print('epochs: {}\n'
          'batch_size: {}'.format(epochs, batch_size))

    path = osp.join(osp.dirname(osp.realpath(__file__)), data_path)
    # dataset_df = SelfDatasetPic50(path, transform=None).shuffle()
    dataset = TargetDataset(root=path,
                            target_data_space_name=target_data_space_name,
                            target_name=target_name,
                            to_pic50=to_pic50,
                            # target_columns_name='new_activity_values(uM)' if not us_nm else 'new_activity_values(nM)',
                            target_columns_name='new_activity_values(uM)',
                            transform=None)

    y_max = dataset.data.y.max()
    y_min = dataset.data.y.min()
    y_std = dataset.data.y.std()
    y_mean = dataset.data.y.mean()

    if not to_pic50:
        maxy_miny_gap = y_max - y_min
        dataset.data.y = (dataset.data.y - y_min) / maxy_miny_gap
        dataset.train.data.y = (dataset.train.data.y - y_min) / maxy_miny_gap  # 只需要在train中归一化y train val test的data类是同一个

    # 储存任务超参数
    train_parm_dic = {
        'task_mark': task_mark,
        'hidden_dim': hidden_dim,
        'mess_nn_dim': mess_nn_dim,
        'mol_in_dim': dataset.self_num_node_features,
        'mol_in_edge_dim': dataset.self_num_edge_features,
        'out_dim': out_dim,
        'gat_heads': gat_heads,
        'massage_depth': massage_depth,
        'dropout_rate': dropout_rate,
        'use_gru': use_gru,
        'AttentiveFP_num_layers': AttentiveFP_num_layers,
        'lr': lr,
        'batch_size': batch_size,
        'epochs': epochs,
        'data_max': y_max.item(),
        'data_min': y_min.item(),
        'data_mean': y_mean.item(),
        'data_std': y_std.item(),
        # 'us_nm': us_nm,
    }

    if save_to_split_csv:
        print('Saving normalize data to csv...')
        cooked_data_to_csv(dataset.test, os.path.join(dataset.target_data_split, f'test_cooked.csv'))
        cooked_data_to_csv(dataset.val, os.path.join(dataset.target_data_split, f'val_cooked.csv'))
        cooked_data_to_csv(dataset.train, os.path.join(dataset.target_data_split, f'train_cooked.csv'))

    test_loader = DataLoader(dataset.test, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(dataset.val, batch_size=batch_size, shuffle=False)
    train_loader = DataLoader(dataset.train, batch_size=batch_size, shuffle=True)

    if debug:
        test_loader = DataLoader(dataset.test[:250], batch_size=batch_size, shuffle=False)
        val_loader = DataLoader(dataset.val[:250], batch_size=batch_size, shuffle=False)
        train_loader = DataLoader(dataset.train[:250], batch_size=batch_size, shuffle=True)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    if model_net == 'GCN_net':
        model = GCN_net(mol_in_dim=dataset.self_num_node_features, out_dim=out_dim, dim=hidden_dim)
    elif model_net == 'MPNN_net':
        model = MPNN_net(mol_in_node_dim=dataset.self_num_node_features, mol_in_edge_dim=dataset.self_num_edge_features,
                         out_dim=out_dim, dim=hidden_dim, mess_nn_dim=mess_nn_dim)
    elif model_net == 'GAT_net':
        model = GAT_net(mol_in_dim=dataset.self_num_node_features, out_dim=out_dim, gat_heads=gat_heads, dim=hidden_dim)
    elif model_net == 'GIN_net':
        model = GIN_net(mol_in_dim=dataset.self_num_node_features, out_dim=out_dim, dim=hidden_dim)
    elif model_net == 'AttentiveFP_net':
        model = AttentiveFP_net(mol_in_dim=dataset.self_num_node_features,
                                mol_in_edge_dim=dataset.self_num_edge_features, out_dim=out_dim, dim=hidden_dim,
                                num_layers=AttentiveFP_num_layers)
    elif model_net == 'DMPNN_Change':
        model = DMPNN_Change(mol_in_dim=dataset.self_num_node_features, out_dim=out_dim, dim=hidden_dim,
                             mol_in_edge_dim=dataset.self_num_edge_features, use_gru=use_gru,
                             massage_depth=massage_depth,
                             dropout_rate=dropout_rate)
    else:
        raise ValueError

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                           factor=0.7, patience=5,
                                                           min_lr=0.00001)

    torch.save(train_parm_dic, os.path.join(model_save_path, f'{task_mark}_train_parm.pkl'))

    best_val_error = float('inf')

    if not os.path.exists(model_save_path):
        os.mkdir(model_save_path)
    results = []
    draw_output = defaultdict(list)

    for epoch in tqdm(range(1, epochs + 1)):
        lr = scheduler.optimizer.param_groups[0]['lr']
        loss = train(epoch)
        result = test_model(val_loader)
        val_error = result['mae']
        scheduler.step(val_error)
        results.append(result)

        if val_error <= best_val_error:
            best_val_error = val_error

            test_results = test_model(test_loader)
            test_error = test_results['mae']
            model_state = model.state_dict()
            torch.save(model_state, os.path.join(model_save_path, f'{task_mark}_state.pkl'))
        draw_output['Epoch'].append(epoch)
        draw_output['Val MAE'].append(val_error)
        draw_output['Val MSE'].append(result['mse'])
        draw_output['Val RMSE'].append(result['rmse'])
        draw_output['Val R2'].append(result['r2'])
        draw_output['Test MAE'].append(test_error)
        draw_output['Test MSE'].append(test_results['mse'])
        draw_output['Test RMSE'].append(test_results['rmse'])
        draw_output['Test R2'].append(test_results['r2'])

        print(
            'Epoch: {:03d}, LR: {:.7f}, Loss: {:.7f}, Val MAE: {:.7f}, Val MSE: {:.7f},'
            ' Val RMSE: {:.7f}'
            ', Val R2: {:.7f},'
            'Test MAE: {:.7f}, Test MSE: {:.7f}, '
            'Test RMSE: {:.7f}, Test R2: {:.7f}'.format(epoch,
                                                        lr,
                                                        loss,
                                                        val_error,
                                                        result['mse'],
                                                        result['rmse'],

                                                        result['r2'],
                                                        test_error,
                                                        test_results[
                                                            'mse'],
                                                        test_results[
                                                            'rmse'],
                                                        test_results[
                                                            'r2']
                                                        ))

        torch.save(draw_output, os.path.join(model_save_path, f'{task_mark}_draw_output_dic.pkl'))
