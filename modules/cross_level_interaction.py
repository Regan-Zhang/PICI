import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
import torch

def make_pseudo_label(loader, model,class_num):
    model.eval()
    scaler = MinMaxScaler()
    zi_list ,zj_list= [],[]
    for step, ((x_i, x_j), _) in enumerate(loader):
        x_i = x_i.to('cuda')
        x_j = x_j.to('cuda')
        with torch.no_grad():
            (z_i, z_j, _, _), _, _ = model.forward(x_i,x_j)
        z_i = z_i.cpu().detach().numpy()
        z_i = scaler.fit_transform(z_i)
        zi_list.extend(z_i)
        z_j = z_j.cpu().detach().numpy()
        z_j = scaler.fit_transform(z_j)
        zj_list.extend(z_j)

    kmeans = KMeans(n_clusters=class_num, n_init=100)
    new_pseudo_label = []

    Pseudo_label = kmeans.fit_predict(zi_list)
    Pseudo_label = Pseudo_label.reshape(len(zi_list), 1)
    Pseudo_label = torch.from_numpy(Pseudo_label)
    new_pseudo_label.append(Pseudo_label)

    Pseudo_label = kmeans.fit_predict(zj_list)
    Pseudo_label = Pseudo_label.reshape(len(zj_list), 1)
    Pseudo_label = torch.from_numpy(Pseudo_label)
    new_pseudo_label.append(Pseudo_label)

    return new_pseudo_label


def match(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    new_y = np.zeros(y_true.shape[0])
    for i in range(y_pred.size):
        for j in row_ind:
            if y_true[i] == col_ind[j]:
                new_y[i] = row_ind[j]
    new_y = torch.from_numpy(new_y).long().to('cuda')
    new_y = new_y.view(new_y.size()[0])
    return new_y
