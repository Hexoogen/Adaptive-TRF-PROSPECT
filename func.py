import numpy as np
import prospect

from prospect import prospect_pro,prospect_d
from prosail import spectral_lib
from prospect import calctav
import pandas as pd
import math
import logging

def prospecular(para):
    n = para[0]
    cab = para[1]
    car = para[2]
    cb = para[3]
    cw = para[4]
    cm = para[5]
    ant = para[6]
    prot = para[7]
    cbc = para[8]
    rough = para[9]
    fs= para[10]
    sza = np.deg2rad(para[11])
    vza = np.deg2rad(para[12])
    raa = np.deg2rad(para[13])

    _, DHR_diff= prospect_pro(n, cab, car, cb, cw, cm, ant, prot, cbc)

    ph = np.arccos(np.cos(sza)* np.cos(vza) + np.sin(sza)* np.sin(vza)*np.cos(raa))
    pha = np.arccos((np.cos(sza) + np.cos(vza)) / (2 * np.cos(ph/2)))

    E1 = 2 * np.cos(pha) *np.cos(sza) / np.cos(ph / 2)
    E2 = 2 * np.cos(pha) *np.cos(vza) / np.cos(ph / 2)
    G = np.minimum(1, np.minimum(E1, E2))

    nr = np.load('lut/DLM_nr.npy')
    g = np.sqrt(np.abs((fs*nr) ** 2 + np.cos(ph / 2) ** 2 - 1))  # g
    beckman=np.exp(-np.tan(pha) ** 2 / rough ** 2)/(np.pi*(rough ** 2) * np.cos(pha) ** 4)
    fnq = 0.5 * ((g - np.cos(ph / 2)) / (g + np.cos(ph / 2))) ** 2  # The first half of Fresnel factor
    fnh = 1 + ((np.cos(ph / 2) * (g + np.cos(ph / 2)) - 1) /
               (np.cos(ph / 2) * (g - np.cos(ph / 2)) + 1)) ** 2  # The second half of Fresnel factor
    Fn = fnq * fnh  # Fresnel factor
    fmq = 4 * np.cos(sza)* np.cos(vza)
    brdfj = beckman * Fn * G / fmq  # BRDF of specular component
    brf=DHR_diff+brdfj*np.pi
    return brf

def prospeculard(para):
    n = para[0]
    cab = para[1]
    car = para[2]
    ant = para[3]
    cw = para[4]
    cm = para[5]
    cb = para[6]
    rough = para[7]
    fs= para[8]
    sza = np.deg2rad(para[9])
    vza = np.deg2rad(para[10])
    raa = np.deg2rad(para[11])

    _, DHR_diff= prospect_d(n, cab, car, ant, cw, cm, cb)

    ph = np.arccos(np.cos(sza)* np.cos(vza) + np.sin(sza)* np.sin(vza)*np.cos(raa))
    pha = np.arccos((np.cos(sza) + np.cos(vza)) / (2 * np.cos(ph/2)))

    E1 = 2 * np.cos(pha) *np.cos(sza) / np.cos(ph / 2)
    E2 = 2 * np.cos(pha) *np.cos(vza) / np.cos(ph / 2)
    G = np.minimum(1, np.minimum(E1, E2))

    nr = np.load('lut/DLM_nr.npy')
    g = np.sqrt(np.abs((fs*nr) ** 2 + np.cos(ph / 2) ** 2 - 1))  # g
    beckman=np.exp(-np.tan(pha) ** 2 / rough ** 2)/(np.pi*(rough ** 2) * np.cos(pha) ** 4)
    fnq = 0.5 * ((g - np.cos(ph / 2)) / (g + np.cos(ph / 2))) ** 2  # The first half of Fresnel factor
    fnh = 1 + ((np.cos(ph / 2) * (g + np.cos(ph / 2)) - 1) /
               (np.cos(ph / 2) * (g - np.cos(ph / 2)) + 1)) ** 2  # The second half of Fresnel factor
    Fn = fnq * fnh  # Fresnel factor
    fmq = 4 * np.cos(sza)* np.cos(vza)
    brdfj = beckman * Fn * G / fmq  # BRDF of specular component
    brf=DHR_diff+brdfj*np.pi
    return brf

def pbrfull(para):
    n = para[0]
    cab = para[1]
    car = para[2]
    ant = para[3]
    cw = para[4]
    cm = para[5]
    cb = para[6]
    rough = para[7]
    fs= para[8]
    sza = np.deg2rad(para[9])
    vza = np.deg2rad(para[10])
    raa = np.deg2rad(para[11])

    DHR, _= prospect_d(n, cab, car, ant, cw, cm, cb)

    rough2=rough**2
    ph = np.arccos(np.cos(sza) * np.cos(vza) + np.sin(sza) * np.sin(vza) * np.cos(raa))
    vh = np.cos(ph / 2)
    nm = (np.cos(sza) + np.cos(vza)) / (2 * vh)
    G = 2 / (np.sqrt(1 + np.tan(sza) ** 2 * rough2) + np.sqrt(1 + np.tan(vza) ** 2 * rough2))
    ggx = rough2 / (np.pi * (nm ** 2 * (rough2 - 1) + 1) ** 2)

    nr = np.load('lut/DLM_nr.npy')
    F0 = ((1 - fs*nr) / (1 + fs*nr)) ** 2  # Fresnel factor
    Fn = F0 + (1 - F0) * (1 - vh) ** 5  # Fresnel factor
    brf_spec = np.pi * G * ggx * Fn / (4 * np.cos(sza) * np.cos(vza))
    brf=DHR- dhr_spec(rough)+brf_spec
    return brf

def pbrfullf(para):
    n = para[0]
    cab = para[1]
    car = para[2]
    ant = para[3]
    cw = para[4]
    cm = para[5]
    cb = para[6]
    rough = para[7]
    fs= para[8]
    sza = np.deg2rad(para[9])
    vza = np.deg2rad(para[10])
    raa = np.deg2rad(para[11])

    DHR, _= prospect_d(n, cab, car, ant, cw, cm, cb)

    rough2=rough**2
    ph = np.arccos(np.cos(sza) * np.cos(vza) + np.sin(sza) * np.sin(vza) * np.cos(raa))
    vh = np.cos(ph / 2)
    nm = (np.cos(sza) + np.cos(vza)) / (2 * vh)
    G = 2 / (np.sqrt(1 + np.tan(sza) ** 2 * rough2) + np.sqrt(1 + np.tan(vza) ** 2 * rough2))
    ggx = rough2 / (np.pi * (nm ** 2 * (rough2 - 1) + 1) ** 2)

    nr = np.load('lut/DLM_nr.npy')
    g = np.sqrt(np.abs((fs * nr) ** 2 + np.cos(ph / 2) ** 2 - 1))  # g
    fnq = 0.5 * ((g - np.cos(ph / 2)) / (g + np.cos(ph / 2))) ** 2  # The first half of Fresnel factor
    fnh = 1 + ((np.cos(ph / 2) * (g + np.cos(ph / 2)) - 1) /
               (np.cos(ph / 2) * (g - np.cos(ph / 2)) + 1)) ** 2  # The second half of Fresnel factor
    Fn = fnq * fnh  # Fresnel factor

    brf_spec = np.pi * G * ggx * Fn / (4 * np.cos(sza) * np.cos(vza))
    brf=DHR- dhr_spec(rough)+brf_spec
    return brf

def dhr_spec(rough):
    nr=spectral_lib.prospectd.nr
    F0= ((1 - nr) / (1 + nr)) ** 2
    return F0*np.exp(-1.26 * rough**1.79)

def Kf_lut(lut,ph,rough):
    pidx0 = int(np.floor(ph) - 1)
    pidx1 = int(np.floor(ph))
    pper = ph - np.floor(ph)

    ridx0=int(np.floor(rough*100)-1)
    ridx1=int(np.floor(rough*100))
    rper=rough*100-np.floor(rough*100)

    f00,f01,f10,f11=lut[pidx0,ridx0],lut[pidx0,ridx1],lut[pidx1,ridx0],lut[pidx1,ridx1]
    Kf=(1 - pper) * ((1 - rper) * f00 + rper * f01) + pper * ((1 - rper) * f10 + rper * f11)
    return Kf

def diff_lut(h,h2,dhr,ph,rough):

    pidx0 = int(np.floor(ph) - 1)
    pidx1 = int(np.floor(ph))
    pper = ph - np.floor(ph)

    ridx0=int(np.floor(rough*100)-1)
    ridx1=int(np.floor(rough*100))
    rper=rough*100-np.floor(rough*100)

    f00,f01,f10,f11=h[pidx0,ridx0],h[pidx0,ridx1],h[pidx1,ridx0],h[pidx1,ridx1]
    k=(1 - pper) * ((1 - rper) * f00 + rper * f01) + pper * ((1 - rper) * f10 + rper * f11)

    f00,f01,f10,f11=h2[pidx0,ridx0],h2[pidx0,ridx1],h2[pidx1,ridx0],h2[pidx1,ridx1]
    b=(1 - pper) * ((1 - rper) * f00 + rper * f01) + pper * ((1 - rper) * f10 + rper * f11)
    return k*(dhr-b)

#%%
def resample_to_10nm(refl):
    wl=np.arange(400,2501)
    bins = np.arange(400, 2501, 10)
    resampled = []
    new_waves = []

    for i in range(len(bins) - 1):
        mask = (wl >= bins[i]) & (wl < bins[i+1])
        if mask.sum() == 0:
            continue
        resampled.append(refl[:, mask].mean(axis=1))
        new_waves.append((bins[i] + bins[i+1]) / 2)

    return np.vstack(resampled).T, np.array(new_waves)
#%% data
def get_global_best():
    D = pd.read_csv('spec/global.csv')
    D = D[D['Foreoptic type'] != 'Integrating sphere']
    X = D.iloc[:, :196].values
    X = X[X[:, 55] > 0.05]
    return X

def get_SVC():
    resampler = np.load('lut/Resampler.npy')
    D = pd.read_csv('spec/All_SVC.csv')
    wl = [str(x) for x in range(400, 2501)]
    return np.dot(D[wl], resampler.T)

def get_ga():
    X1=get_global_best()
    X2=get_SVC()
    return np.vstack((X1,X2))

def get_global_no():
    D = pd.read_csv('spec/global.csv')
    D1 = D[D['Foreoptic type'] == 'Integrating sphere']
    D2 = D[D.iloc[:, 55] < 0.05]
    X = pd.concat([D1, D2], axis=0).iloc[:, :196].values
    return X

def get_all():
    D1 = pd.read_csv('spec/global.csv').iloc[:, :196]
    resampler = np.load('lut/Resampler.npy')
    D2 = pd.read_csv('spec/All_SVC.csv')
    wl = [str(x) for x in range(400, 2501)]
    D2 = np.dot(D2[wl], resampler.T)
    D2 = pd.DataFrame(D2,columns=D1.columns)
    D=pd.concat([D1,D2], axis=0)
    return D.values

def get_all_sim():
    D1 = pd.read_csv('spec/global.csv').iloc[:, :196].values
    D2 = pd.read_csv('spec/All_SVC.csv')
    wl = [str(x) for x in range(400, 2501)]
    D2 = D2[wl].values
    return D1,D2
def get_all_simr():
    D1 = pd.read_csv('spec/global.csv').iloc[:, :196].values.ravel()
    D2 = pd.read_csv('spec/All_SVC.csv')
    wl = [str(x) for x in range(400, 2501)]
    D2 = D2[wl].values.ravel()
    D=np.hstack((D1,D2))
    return D

def get_hsi():
    D1 = pd.read_csv('spec/ApMV900.csv')
    D2 = pd.read_csv('spec/Grape.csv')
    X1 = D1.values[:, :115]
    X2 = D2.values[:, :115]
    X = np.vstack((X1, X2))
    return X

#%%
import numpy as np
import torch
import torch.nn.functional as F
from prosailt.prospect_d import prospect_dt,prospect_dt1
from prosailt import spectral_lib as pt_lib
from dataclasses import dataclass
from typing import Callable, Optional
lut = torch.tensor(np.load('lut/brf_spec.npy'),dtype=torch.float32,device='cuda').unsqueeze(0).unsqueeze(0)
dlm_nr = torch.tensor(np.load('lut/DLM_nr.npy'),dtype=torch.float32,device='cuda').unsqueeze(0)

def bilinear_grid_sample(input, grid, align_corners=True):
    """
    可微双线性 grid_sample，兼容 jacfwd / vmap / CUDA
    """
    N, C, H, W = input.shape
    _, Ho, Wo, _ = grid.shape

    # 网格坐标 [-1, 1] → 像素坐标
    if align_corners:
        x = (grid[..., 0] + 1) / 2 * (W - 1)
        y = (grid[..., 1] + 1) / 2 * (H - 1)
    else:
        x = ((grid[..., 0] + 1) * W - 1) / 2
        y = ((grid[..., 1] + 1) * H - 1) / 2

    # ══════════════════════════════════════════════════════
    #  关键修复：NaN 保护 + 严格 clamp 确保索引不越界
    # ══════════════════════════════════════════════════════
    x = torch.nan_to_num(x, nan=0.0).clamp(0, W - 1 - 1e-7)
    y = torch.nan_to_num(y, nan=0.0).clamp(0, H - 1 - 1e-7)
    #                                   上界 -1-1e-7 保证 x0+1 ≤ W-1

    # 四个最近像素的整数坐标
    x0 = x.long()
    x1 = x0 + 1                          # 现在保证 ≤ W-1
    y0 = y.long()
    y1 = y0 + 1                          # 现在保证 ≤ H-1

    # 最终安全 clamp（防御性编程）
    x0 = x0.clamp(0, W - 1)
    x1 = x1.clamp(0, W - 1)
    y0 = y0.clamp(0, H - 1)
    y1 = y1.clamp(0, H - 1)

    # 插值权重
    wx1 = x - x0.float()
    wx0 = 1 - wx1
    wy1 = y - y0.float()
    wy0 = 1 - wy1

    # gather 函数
    def gather_pixel(y_idx, x_idx):
        y_exp = y_idx.unsqueeze(1).expand(-1, C, -1, -1)
        x_exp = x_idx.unsqueeze(1).expand(-1, C, -1, -1)

        idx = y_exp * W + x_exp
        idx = idx.clamp(0, H * W - 1)        # 最终兜底
        idx_flat = idx.reshape(N, C, -1)

        input_flat = input.reshape(N, C, -1)
        gathered = torch.gather(input_flat, 2, idx_flat)
        return gathered.reshape(N, C, Ho, Wo)

    v00 = gather_pixel(y0, x0)
    v01 = gather_pixel(y0, x1)
    v10 = gather_pixel(y1, x0)
    v11 = gather_pixel(y1, x1)

    # 双线性插值
    w00 = (wx0 * wy0).unsqueeze(1)
    w01 = (wx1 * wy0).unsqueeze(1)
    w10 = (wx0 * wy1).unsqueeze(1)
    w11 = (wx1 * wy1).unsqueeze(1)

    output = w00 * v00 + w01 * v01 + w10 * v10 + w11 * v11
    return output


def Kf_sampler(phase, rough):
    # phase, rough=phase.squeeze(-1), rough.squeeze(-1)


    phase_norm = phase / 90.0
    phase_norm = phase_norm * 2 - 1

    # rough = rough.clamp(min=1e-6)# ← 保护：log(0) = -inf
    # rough_norm = (torch.log(rough) - np.log(0.001)) / (np.log(1.0) - np.log(0.001))
    # rough_norm = rough_norm * 2 - 1
    rough_norm = rough / 100.0
    rough_norm = rough_norm * 2 - 1
    grid = torch.stack([rough_norm, phase_norm],dim=-1)
    grid = grid.unsqueeze(1)
    # kf = F.grid_sample(lut.expand(len(phase), -1, -1, -1),grid,mode='bilinear',padding_mode='border',align_corners=True)
    kf = bilinear_grid_sample(
        lut.expand(len(phase), -1, -1, -1),  # (N, 1, 91, 500)
        grid,  # (N, 1, 1, 2)
        align_corners=True,
    )
    return kf.squeeze()

def rough_torch(params):
    N = params[:,0].reshape(-1,1)
    cab = params[:,1].reshape(-1,1)
    car = params[:,2].reshape(-1,1)
    ant = params[:,3].reshape(-1,1)
    cw = params[:,4].reshape(-1,1)
    cm = params[:,5].reshape(-1,1)
    cb = params[:,6].reshape(-1,1)
    rough = params[:,7].reshape(-1,1)
    fs = params[:,8].reshape(-1,1)
    phase=torch.ones_like(rough)*35

    # rough = torch.tensor([0.01, 0.05, 0.3],dtype=torch.float32,device='cuda').reshape(-1,1)
    # fs = torch.tensor([1, 1.5, 2],dtype=torch.float32,device='cuda').reshape(-1,1)

    dhr, _ = prospect_dt(N,cab,car,ant,cw,cm,cb)
    kf = Kf_sampler(phase, rough).reshape(-1,1)
    F0 = ((1 - fs * dlm_nr) / (1 + fs * dlm_nr)) ** 2
    # vh = torch.cos(torch.deg2rad(torch.tensor(35 / 2,device='cuda')))
    # Fn = F0+(1-F0)*(1-np.cos(np.deg2rad(35/2)))**5
    brf=dhr+kf*F0
    return brf

def full_torch(params):
    N = params[:,0].reshape(-1,1)
    cab = params[:,1].reshape(-1,1)
    car = params[:,2].reshape(-1,1)
    ant = params[:,3].reshape(-1,1)
    cw = params[:,4].reshape(-1,1)
    cm = params[:,5].reshape(-1,1)
    cb = params[:,6].reshape(-1,1)
    rough = params[:,7].reshape(-1,1)
    fs = params[:,8].reshape(-1,1)
    pha=torch.ones_like(rough)*np.deg2rad(35)

    # params = torch.rand(1000, 9) * torch.tensor([2.5, 120, 30, 40,0.06, 0.05,1,1,3])
    # params[:, 0] += 1  # N ∈ [1, 3.5]
    # params[:, -1] += 0.5  # N ∈ [1, 3.5]
    # params = params.to('cuda')
    # rough = torch.tensor([0.01, 0.05, 0.3],dtype=torch.float32,device='cuda').reshape(-1,1)
    # fs = torch.tensor([1, 1.5, 2],dtype=torch.float32,device='cuda').reshape(-1,1)

    dhr, _ = prospect_dt(N,cab,car,ant,cw,cm,cb)
    rough2 = rough ** 2
    G = 2 / (1 + torch.sqrt(1 + torch.tan(pha) ** 2 * rough2))
    ggx = rough2 / (torch.pi * (torch.cos(pha / 2) ** 2 * (rough2 - 1) + 1) ** 2)
    kf = G * ggx / (4 * torch.cos(pha))
    F0 = ((1 - fs * dlm_nr) / (1 + fs * dlm_nr)) ** 2
    # vh = torch.cos(torch.deg2rad(torch.tensor(35 / 2,device='cuda')))
    # Fn = F0+(1-F0)*(1-np.cos(np.deg2rad(35/2)))**5
    brf=dhr+kf*F0
    return brf
    # resampler=torch.tensor(np.load('lut/Resampler.npy')[:36], dtype=torch.float32, device='cuda')
    # return brf @ resampler.T
def Kf_torch(params):
    N = params[:,0].reshape(-1,1)
    cab = params[:,1].reshape(-1,1)
    car = params[:,2].reshape(-1,1)
    ant = params[:,3].reshape(-1,1)
    cw = params[:,4].reshape(-1,1)
    cm = params[:,5].reshape(-1,1)
    cb = params[:,6].reshape(-1,1)
    Kf = params[:,7].reshape(-1,1)
    fs = params[:,8].reshape(-1,1)

    _, dhr_diff = prospect_dt(N,cab,car,ant,cw,cm,cb)
    F0 = ((1 - fs * dlm_nr) / (1 + fs * dlm_nr)) ** 2
    # vh = torch.cos(torch.deg2rad(torch.tensor(35 / 2,device='cuda')))
    # Fn = F0+(1-F0)*(1-np.cos(np.deg2rad(35/2)))**5
    brf=dhr_diff+Kf*F0
    return brf
    # resampler=torch.tensor(np.load('lut/Resampler.npy')[:36], dtype=torch.float32, device='cuda')
    # return brf @ resampler.T


#%%
#%%
import torch
from torch.func import jacfwd, vmap


class AdaptiveLM:
    """
    批量自适应 Levenberg-Marquardt 优化器
    - sigmoid reparameterization 实现有界约束
    - torch.func.jacfwd + vmap 计算批量 Jacobian
    - 逐样本自适应阻尼 + 早停
    """

    def __init__(
        self,
        forward_model,
        low,
        high,
        n_params,
        device='cuda',
    ):
        self.forward_model = forward_model
        self.device = device
        self.P = n_params

        # 边界
        self.low  = low.to(device).float()
        self.high = high.to(device).float()
        self.scale = self.high - self.low

        # Jacobian 函数
        self._build_jacobian_fn()

    def _build_jacobian_fn(self):
        """构建 vmap(jacfwd) 的 Jacobian 函数"""
        def residual_single(param, single_obs):
            pred = self.forward_model(param.unsqueeze(0)).squeeze(0)
            return pred - single_obs

        self.jacobian_fn = vmap(jacfwd(residual_single), in_dims=(0, 0))

    def _to_params(self, z):
        """z (无约束) → 物理参数 (有界)，纯线性映射"""
        s = torch.sigmoid(z)
        return s * self.scale + self.low

    def _dp_dz(self, z):
        """∂params/∂z，用于链式法则转换 Jacobian"""
        s = torch.sigmoid(z)
        return s * (1 - s) * self.scale

    def _to_norm(self, params):
        """物理参数 → 归一化 (0, 1)，纯线性"""
        return ((params - self.low) / self.scale).clamp(0.02, 0.98)

    def solve(
        self,
        obs,
        init_params,
        n_iters=100,
        lam_init=1e-2,
        lam_up=10.0,
        lam_down=10.0,
        max_lam=1e8,
        min_lam=1e-12,
        tol_loss=1e-12,
        tol_param=1e-10,
        patience=15,
        grad_clip=10.0,
        verbose=False,
    ):
        B, L = obs.shape
        P = self.P
        device = self.device

        # 初值映射到无约束空间
        p_norm = self._to_norm(init_params)
        z = torch.logit(p_norm)

        obs_gpu = obs.to(device).float()

        # 逐样本状态
        lam         = torch.full((B, 1), lam_init, device=device)
        prev_loss   = torch.full((B,), float('inf'), device=device)
        stall_count = torch.zeros(B, dtype=torch.long, device=device)
        converged   = torch.zeros(B, dtype=torch.bool, device=device)

        loss_history = []

        for it in range(n_iters):
            if converged.all():
                break

            with torch.no_grad():
                params = self._to_params(z)
                pred = self.forward_model(params)
                r = pred - obs_gpu
                loss = (r ** 2).sum(dim=1)

            loss_mean = loss[~converged].mean().item()

            if verbose and it % 10 == 0:
                print(f'  iter={it:3d}  loss={loss_mean:.10f}  '
                      f'λ={lam.mean().item():.2e}  '
                      f'conv={converged.sum().item()}/{B}')

            with torch.no_grad():
                J_params = self.jacobian_fn(params, obs_gpu)
                dpdz = self._dp_dz(z)
                J = J_params * dpdz.unsqueeze(1)

                JT = J.transpose(1, 2)
                H = JT @ J
                g = JT @ r.unsqueeze(-1)

                # 梯度裁剪
                g_norm = g.squeeze(-1).norm(dim=1, keepdim=True)
                g_scale = torch.where(
                    g_norm > grad_clip,
                    grad_clip / (g_norm + 1e-15),
                    torch.ones_like(g_norm),
                )
                g = g * g_scale.unsqueeze(-1)

                # 阻尼
                H_diag = H.diagonal(dim1=1, dim2=2)
                H_damped = H + torch.diag_embed(lam * (H_diag + 1e-8))

                # 求解
                try:
                    delta = torch.linalg.solve(
                        H_damped, g
                    ).squeeze(-1).reshape(B, P)
                except Exception:
                    delta = torch.linalg.lstsq(
                        H_damped, g
                    ).solution.squeeze(-1).reshape(B, P)

                # 预测下降
                Jd = (J @ delta.unsqueeze(-1)).squeeze(-1)
                Hd = (H @ delta.unsqueeze(-1)).squeeze(-1)
                predicted = (Jd ** 2).sum(dim=1) + (delta * Hd).sum(dim=1)

                # 试探
                z_new = z - delta
                new_params = self._to_params(z_new)
                new_pred = self.forward_model(new_params)
                new_loss = ((new_pred - obs_gpu) ** 2).sum(dim=1)

                # 增益比
                rho = (loss - new_loss) / (predicted.abs() + 1e-15)

                # 逐样本分类
                active     = ~converged
                very_good  = (rho > 0.75)  & active
                good       = (rho > 0.25)  & (rho <= 0.75) & active
                poor       = (rho > 0.0)   & (rho <= 0.25) & active
                bad        = (rho <= 0.0)   & active

                accept = very_good | good | poor

                # z 更新
                z = torch.where(accept.unsqueeze(1), z_new, z)

                # λ 更新
                lam = torch.where(
                    very_good.unsqueeze(1), lam / lam_down,
                    torch.where(
                        poor.unsqueeze(1), lam * (lam_up ** 0.5),
                        torch.where(
                            bad.unsqueeze(1), lam * lam_up,
                            lam,
                        )
                    )
                )
                lam = lam.clamp(min=min_lam, max=max_lam)

                # loss 更新
                loss_accepted = torch.where(accept, new_loss, loss)

                # 收敛判断
                loss_unchanged = (prev_loss - loss_accepted).abs() < tol_loss
                param_changed  = delta.abs().max(dim=1).values > tol_param
                grad_small     = g.squeeze(-1).norm(dim=1) < 1e-8

                stalled     = (loss_unchanged & ~param_changed) | grad_small
                stall_count = torch.where(stalled, stall_count + 1, stall_count * 0)
                newly_done  = stall_count >= patience
                converged   = converged | newly_done

                prev_loss = loss_accepted

        # 最终参数
        with torch.no_grad():
            final_params = self._to_params(z)
            final_params = torch.clamp(
                final_params, self.low.unsqueeze(0), self.high.unsqueeze(0)
            )

        if verbose:
            print(f'\n  Done: {converged.sum().item()}/{B} converged')

        return final_params, converged, loss_history


def batch_solve_with_retry(obs_all, lm, p0, low, high,
                           target=0.95, max_rounds=10, batch_size=2000):
    N = obs_all.shape[0]
    P = len(p0)
    all_params = np.zeros((N, P))
    all_conv   = np.zeros(N, dtype=bool)

    round_cfgs = [
        {'n_iters': 80,  'patience': 15, 'lam_init': 1e-2},
        {'n_iters': 120, 'patience': 25, 'lam_init': 1e-3},
        {'n_iters': 160, 'patience': 35, 'lam_init': 1e-3},
        {'n_iters': 200, 'patience': 50, 'lam_init': 1e-4},
        {'n_iters': 300, 'patience': 60, 'lam_init': 1e-4},
    ]

    for rnd in range(max_rounds):
        mask = ~all_conv
        rate = all_conv.mean()

        if rate >= target and rnd > 0:
            break
        if mask.sum() == 0:
            break

        cfg = round_cfgs[min(rnd, len(round_cfgs) - 1)]
        n_todo = mask.sum()
        print(f'\n{"="*60}')
        print(f'Round {rnd+1}: {n_todo} remaining, '
              f'convergence={rate*100:.1f}%')
        print(f'{"="*60}')

        lo = low.to('cuda')
        hi = high.to('cuda')

        # 生成初值
        if rnd == 0:
            init = p0.unsqueeze(0).expand(n_todo, -1).to('cuda')

        elif rnd <= 2:
            init = torch.tensor(all_params[mask], dtype=torch.float32, device='cuda')
            noise = torch.randn_like(init) * 0.05 * (hi - lo)
            init = (init + noise).clamp(lo, hi)

        else:
            init = lo + torch.rand(n_todo, P, device='cuda') * (hi - lo)

        # 逐批反演
        obs_todo = obs_all[mask]

        for start in range(0, n_todo, batch_size):
            end = min(start + batch_size, n_todo)

            obs_b  = torch.tensor(obs_todo[start:end], dtype=torch.float32)
            init_b = init[start:end]

            params_b, conv_b, _ = lm.solve(
                obs=obs_b, init_params=init_b, verbose=False, **cfg
            )

            with torch.no_grad():
                pred   = lm.forward_model(params_b.to(lm.device))
                obs_t  = obs_b.to(lm.device)
                new_loss = ((pred - obs_t) ** 2).mean(dim=1).cpu().numpy()

            todo_idx = np.where(mask)[0][start:end]

            if rnd > 0:
                old_loss = np.full(end - start, np.inf)
                has_old  = all_conv[todo_idx]
                if has_old.any():
                    old_p    = torch.tensor(all_params[todo_idx],
                                            dtype=torch.float32, device=lm.device)
                    old_pred = lm.forward_model(old_p)
                    old_loss = ((old_pred - obs_t) ** 2).mean(dim=1).cpu().numpy()
                improved = new_loss < old_loss
            else:
                improved = np.ones(end - start, dtype=bool)

            all_params[todo_idx[improved]] = params_b[improved].cpu().numpy()
            all_conv[todo_idx[improved]]   = conv_b[improved].cpu().numpy()

            torch.cuda.empty_cache()

    print(f'\n{"="*60}')
    print(f'Final: {all_conv.sum()}/{N} ({all_conv.mean()*100:.1f}%)')
    print(f'{"="*60}')

    return all_params, all_conv

