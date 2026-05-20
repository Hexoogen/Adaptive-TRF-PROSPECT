import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import matplotlib.pyplot as plt
import torch
from func import *
import time
#%%
X=np.load('spec/49637.npy')[:5000]
#%%
p0   = torch.tensor([1.5, 40, 10, 0, 0.01, 5e-3, 0, 0.3, 1.5],device='cuda')
low  = torch.tensor([1,  0,  0,  0, 0, 0, 0, 0.01, 0.6],device='cuda')
high = torch.tensor([3.5, 120, 30, 40, 0.06, 0.05, 1, 1,   3.5],device='cuda')

# p0   = torch.tensor([1.5, 40, 10, 0, 0.01, 5e-3, 0, 1, 1.5],device='cuda')
# low  = torch.tensor([1,  0,  0,  0, 0, 0, 0, 0, 1],device='cuda')
# high = torch.tensor([3.5, 120, 30, 40, 0.06, 0.05, 1, 30,   5],device='cuda')

lm = AdaptiveLM(forward_model=full_torch,low=low,high=high,n_params=9,device='cuda',)
t0 = time.time()
paras, converged = batch_solve_with_retry(X, lm, p0, low, high, batch_size=5000,target=0.9999)
time_lbfgs = time.time() - t0
# np.save('full_962.npy', paras)
#%%
# params_r=np.load('Kf_pig.npy')
obs=np.load('plot/opt/D.npz')['cab'][:49637]

plt.scatter(obs,paras[:, 1])
plt.show()
from sklearn.metrics import root_mean_squared_error
def rmse(y_true, y_pred):
    y_pred=y_pred[~np.isnan(y_true)]
    y_true=y_true[~np.isnan(y_true)]
    y_true=y_true[~np.isnan(y_pred)]
    y_pred=y_pred[~np.isnan(y_pred)]
    return root_mean_squared_error(y_true, y_pred)
print(rmse(obs,paras[:, 1]))

