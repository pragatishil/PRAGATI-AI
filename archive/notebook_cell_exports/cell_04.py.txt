#!/usr/bin/env python3
"""
================================================================
AGRO-DOCTOR (KRISHI-AI) — CROSS-DOMAIN COFFEE EVALUATION
================================================================
Purpose:
  Evaluates the trained PrognosisNet V10.1 model on the unseen
  Coffee sequence dataset to test cross-domain generalization.
  
Fixes Applied:
  - Corrected SpearmanR calculation to compute globally across
    flattened outputs rather than averaging unreliable 3-point
    per-sample correlations.
================================================================
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import spearmanr
import scipy.stats as stats

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import warnings
warnings.filterwarnings("ignore")

import math

# ── Paths ──
INPUT_DIR    = "/kaggle/input/datasets/hriitk2026/weight-version-v10"
COFFEE_SEQ   = "/kaggle/working/coffee_sequences_v10.csv"
PROPOSED_PTH = f"{INPUT_DIR}/best_prognosis_net_v10_seed42.pth"  
CONFORMAL    = f"{INPUT_DIR}/conformal_quantiles_v10.pkl"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# ── Dimensions & Hyperparameters ──
EMBED_DIM   = 512
MORPH_DIM   = 16
STAGE_EMB   = 8
CURVE_DIM   = 6
VEL_DIM     = 1
CROP_EMB    = 32
DIS_EMB     = 32
SEQ_L       = 5
SEQ_P       = 3
NUM_LV      = 6
CORAL_K     = 5
CONCEPT_DIM = CROP_EMB + DIS_EMB

VIS_DELTA_DIM = 128
SEQ_FEAT_V10  = EMBED_DIM + VIS_DELTA_DIM + MORPH_DIM + 1 + VEL_DIM + STAGE_EMB + CURVE_DIM

# Determine vocab sizes (fallback to safe max sizes if needed)
try:
    state_dict = torch.load(PROPOSED_PTH, map_location="cpu")
    NUM_CROPS = state_dict['crop_emb.weight'].shape[0]
    NUM_DISEASES = state_dict['dis_emb.weight'].shape[0]
except FileNotFoundError:
    print(f"Error: Could not find model at {PROPOSED_PTH}")
    sys.exit(1)

MORPH_COLS = [
    "lesion_fraction_image", "lesion_fraction_bbox", "bbox_fraction",
    "bbox_aspect_ratio", "bbox_diag_ratio", "lesion_count", "mean_area",
    "max_area", "std_area", "median_area", "area_cv", "dispersion",
    "density", "compactness", "entropy", "convex_fraction"
]
mnorm = [f"m_{c}" for c in MORPH_COLS]

# ================================================================
# MODEL DEFINITION (PrognosisNet V10.1)
# ================================================================

class SinPE(nn.Module):
    def __init__(self,d,mx=64):
        super().__init__()
        pe=torch.zeros(mx,d); pos=torch.arange(mx).unsqueeze(1).float()
        div=torch.exp(torch.arange(0,d,2).float()*(-math.log(10000.)/d))
        pe[:,0::2]=torch.sin(pos*div); pe[:,1::2]=torch.cos(pos*div)
        self.register_buffer("pe",pe.unsqueeze(0))
    def forward(self,x): return x+self.pe[:,:x.size(1),:]

class ARM(nn.Module):
    def __init__(self,d,r=4):
        super().__init__(); r=max(d//r,8)
        self.fc=nn.Sequential(nn.Linear(d,r),nn.ReLU(),nn.Linear(r,d),nn.Sigmoid())
    def forward(self,x): return x*self.fc(x)

class CBAM1D(nn.Module):
    def __init__(self,d,r=8):
        super().__init__(); r=max(d//r,8)
        self.mlp=nn.Sequential(nn.Linear(1,r),nn.ReLU(),nn.Linear(r,d))
        self.sig=nn.Sigmoid()
    def forward(self,x):
        avg=x.mean(dim=1,keepdim=True); mx=x.max(dim=1,keepdim=True)[0]
        return x*self.sig(self.mlp(avg)+self.mlp(mx))

class eSTA(nn.Module):
    def __init__(self,d):
        super().__init__()
        self.W=nn.Linear(d,d); self.v=nn.Linear(d,1,bias=False)
    def forward(self,h):
        a=torch.softmax(self.v(torch.tanh(self.W(h))),dim=1)
        return (h*a).sum(1),a.squeeze(-1)

class CORALHead(nn.Module):
    def __init__(self,in_dim,K=CORAL_K):
        super().__init__()
        self.trunk=nn.Sequential(nn.Linear(in_dim,64),nn.GELU())
        self.fc=nn.Linear(64,1,bias=False); self.bias=nn.Parameter(torch.zeros(K)); self.K=K
    def forward(self,x):
        return torch.sigmoid(self.fc(self.trunk(x))+self.bias.unsqueeze(0))

PROJ_DIM=256; TF_HEADS=4; LSTM_H=128; FB_ITER=2

class PrognosisNetV10(nn.Module):
    def __init__(self):
        super().__init__()
        self.crop_emb = nn.Embedding(NUM_CROPS,    CROP_EMB)
        self.dis_emb  = nn.Embedding(NUM_DISEASES, DIS_EMB)
        self.lv_emb   = nn.Embedding(NUM_LV,       STAGE_EMB)
        
        self.delta_proj = nn.Sequential(
            nn.Linear(EMBED_DIM, VIS_DELTA_DIM), nn.LayerNorm(VIS_DELTA_DIM), nn.GELU()
        )
        self.concept_proj = nn.Sequential(nn.Linear(CONCEPT_DIM, PROJ_DIM), nn.LayerNorm(PROJ_DIM))
        self.inp_proj     = nn.Sequential(nn.Linear(SEQ_FEAT_V10, PROJ_DIM), nn.LayerNorm(PROJ_DIM))
        self.pe           = SinPE(PROJ_DIM)
        self.cross_attn   = nn.TransformerDecoderLayer(
            d_model=PROJ_DIM, nhead=TF_HEADS, dim_feedforward=PROJ_DIM*4, 
            dropout=0.1, batch_first=True, norm_first=True)
                
        self.bilstm   = nn.LSTM(PROJ_DIM, LSTM_H, 1, batch_first=True, bidirectional=True)
        lstm_out = LSTM_H*2; fused = lstm_out + CONCEPT_DIM
        self.esta     = eSTA(lstm_out)
        self.arm      = ARM(fused)
        self.cbam     = CBAM1D(fused)
        self.bn_fused = nn.BatchNorm1d(fused)
        self.fb_proj  = nn.Linear(SEQ_P, fused)
        self.fb_gate  = nn.Linear(fused*2, fused)
        self.mu_head  = nn.Sequential(nn.Linear(fused,128), nn.GELU(), nn.Linear(128, SEQ_P), nn.Sigmoid())
        self.lv_head  = nn.Sequential(nn.Linear(fused,128), nn.GELU(), nn.Linear(128, SEQ_P))
        self.horizon_scale = nn.Parameter(torch.zeros(SEQ_P))
        self.coral_heads = nn.ModuleList([CORALHead(fused) for _ in range(SEQ_P)])
        
        # Heteroscedastic Uncertainty Head
        self.logvar_head = nn.Sequential(nn.Linear(fused, 128), nn.GELU(), nn.Linear(128, SEQ_P))

    def _project(self, vis, vis_delta, morph, sev, vel, lvl, curve, ctx):
        le = self.lv_emb(lvl)
        vis_delta_comp = self.delta_proj(vis_delta)
        x = torch.cat([vis, vis_delta_comp, morph, sev, vel, le, curve], dim=-1)
        x = self.inp_proj(x); x = self.pe(x)
        return x + self.concept_proj(ctx).unsqueeze(1)

    def forward(self, vis, vis_delta, morph, sev, vel, lvl, curve, ci, di):
        c = self.crop_emb(ci); d = self.dis_emb(di); ctx = torch.cat([c, d], dim=1)
        x = self._project(vis, vis_delta, morph, sev, vel, lvl, curve, ctx)
        
        last     = x[:, -1:, :]
        attended = self.cross_attn(tgt=last, memory=x)
        lstm_out, _ = self.bilstm(x)
        lstm_out = lstm_out + attended.expand(-1, SEQ_L, -1)[:, :, :lstm_out.size(-1)]
        
        z_ctx, attn_w = self.esta(lstm_out)
        z = torch.cat([z_ctx, ctx], dim=1)
        z = self.bn_fused(z); z = self.arm(z); z = self.cbam(z)
        
        mu = self.mu_head(z)
        for _ in range(FB_ITER):
            fb = self.fb_proj(mu); gate = torch.sigmoid(self.fb_gate(torch.cat([z, fb], 1)))
            z = z + gate * fb; mu = self.mu_head(z)
            
        lv_pred = self.lv_head(z) + self.horizon_scale.unsqueeze(0)
        logv = self.logvar_head(z)
        coral_probs = [self.coral_heads[p](z) for p in range(SEQ_P)]
        
        return mu, logv, lv_pred, coral_probs, attn_w

# ================================================================
# DATASET WRAPPER
# ================================================================

class AblationDS(Dataset):
    def __init__(self, df_):
        self.df = df_.reset_index(drop=True)
        T = SEQ_L
        self.vis_k = [f"t{t}_vis_{i}" for t in range(T) for i in range(EMBED_DIM)]
        self.mph_k = [f"t{t}_{c}"     for t in range(T) for c in mnorm]
        self.sev_k = [f"t{t}_severity"  for t in range(T)]
        self.vel_k = [f"t{t}_vel"       for t in range(T)]
        self.lvl_k = [f"t{t}_level_idx" for t in range(T)]
        self.crv_k = [f"t{t}_curve_{i}" for t in range(T) for i in range(CURVE_DIM)]
        self.tgt_k = [f"tgt_sev_{p}"    for p in range(SEQ_P)]

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]; T = SEQ_L
        vis = row[self.vis_k].values.astype(np.float32).reshape(T, EMBED_DIM)
        mph = row[self.mph_k].values.astype(np.float32).reshape(T, MORPH_DIM)
        sev = row[self.sev_k].values.astype(np.float32).reshape(T, 1)
        vel = row[self.vel_k].values.astype(np.float32).reshape(T, 1)
        lvl = row[self.lvl_k].values.astype(np.int64)
        crv = row[self.crv_k].values.astype(np.float32).reshape(T, CURVE_DIM)
        ci  = int(row.crop_idx); di = int(row.disease_idx)
        tgt = row[self.tgt_k].values.astype(np.float32)
        
        vis_delta = np.zeros_like(vis)
        vis_delta[1:] = vis[1:] - vis[:-1]
        
        return (torch.tensor(vis, dtype=torch.float32),
                torch.tensor(vis_delta, dtype=torch.float32),
                torch.tensor(mph, dtype=torch.float32),
                torch.tensor(sev, dtype=torch.float32),
                torch.tensor(vel, dtype=torch.float32),
                torch.tensor(lvl, dtype=torch.long),
                torch.tensor(crv, dtype=torch.float32),
                torch.tensor(ci,  dtype=torch.long),
                torch.tensor(di,  dtype=torch.long),
                torch.tensor(tgt, dtype=torch.float32))

# ================================================================
# EVALUATION METRICS
# ================================================================

def _to_lv(t):
    thr = torch.tensor([0.05, 0.20, 0.40, 0.60, 0.80], device=t.device)
    return torch.bucketize(t, thr).clamp(0, 5)

def compute_picp_mpiw(mu, tgt, sigmas):
    lower = mu - 1.96 * sigmas
    upper = mu + 1.96 * sigmas
    picp = ((tgt >= lower) & (tgt <= upper)).astype(np.float32).mean()
    mpiw = (upper - lower).mean()
    return round(float(picp), 4), round(float(mpiw), 4)

def compute_ece(errors, sigmas, n_bins=10):
    per_step_ece = []
    for p in range(errors.shape[1]):
        e = errors[:, p]; s = sigmas[:, p]
        norm_e = e / (s + 1e-8)
        bins = np.linspace(0, np.percentile(s, 98), n_bins + 1)
        ece = 0.
        for b in range(n_bins):
            mask = (s >= bins[b]) & (s < bins[b+1])
            if mask.sum() == 0: continue
            avg_conf = 1. / (s[mask].mean() + 1e-8)
            avg_err  = e[mask].mean()
            ece += abs(avg_conf - 1./(avg_err + 1e-8)) * mask.sum() / len(e)
        per_step_ece.append(round(float(ece), 5))
    return round(float(np.mean(per_step_ece)), 5)

def compute_uncertainty_error_corr(errors, sigmas):
    per_step = []
    for p in range(errors.shape[1]):
        try:
            r, _ = spearmanr(sigmas[:, p], errors[:, p])
            per_step.append(round(float(r), 4))
        except:
            per_step.append(0.)
    return round(float(np.nanmean(per_step)), 4)

# ================================================================
# MAIN EXECUTION
# ================================================================

if __name__ == "__main__":
    if not os.path.exists(COFFEE_SEQ):
        print(f"Error: Cannot find {COFFEE_SEQ}. Please run the sequence generator first.")
        sys.exit(1)

    print("\nLoading Coffee Sequences...")
    coffee_seq_df = pd.read_csv(COFFEE_SEQ)
    print(f"Found {len(coffee_seq_df)} coffee sequences.")

    print(f"\nLoading Proposed Model from {PROPOSED_PTH}...")
    model = PrognosisNetV10().to(DEVICE)
    model.load_state_dict(state_dict)
    model.eval()

    print("Loading Conformal Quantiles...")
    try:
        with open(CONFORMAL, "rb") as f:
            conf_q = pickle.load(f)
    except FileNotFoundError:
        print(f"Warning: {CONFORMAL} not found. Fallback to 0.1 for all steps.")
        conf_q = {0: 0.1, 1: 0.1, 2: 0.1}

    ds = AblationDS(coffee_seq_df)
    ld = DataLoader(ds, 64, shuffle=False, num_workers=2)

    all_mu=[]; all_tgt=[]; all_sig=[]

    with torch.no_grad():
        for batch in tqdm(ld, desc="  Coffee Inference"):
            vis, vd, mph, sv, vel, lvl, crv, ci, di, tgt = [b.to(DEVICE) for b in batch]
            mu, logv, _, _, _ = model(vis, vd, mph, sv, vel, lvl, crv, ci, di)
            sigmas = torch.exp(0.5 * logv)
            
            all_mu.append(mu.cpu())
            all_tgt.append(tgt.cpu())
            all_sig.append(sigmas.cpu())

    mu  = torch.cat(all_mu)
    tgt = torch.cat(all_tgt)
    sig = torch.cat(all_sig)
    
    errors = (mu - tgt).abs().numpy()

    # Calculate Metrics
    mae   = float(errors.mean())
    rmse  = float(((mu - tgt)**2).mean().sqrt())
    lacc  = float((_to_lv(mu)==_to_lv(tgt)).float().mean())
    nmono = float(F.relu(mu[:,:-1]-mu[:,1:]).mean()) if mu.size(1)>1 else 0.
    
    # FIXED: Proper global Spearman correlation over flattened outputs
    try:
        msr = spearmanr(mu.numpy().flatten(), tgt.numpy().flatten()).correlation
        if np.isnan(msr): msr = 0.
    except:
        msr = 0.
    
    picp, mpiw = compute_picp_mpiw(mu.numpy(), tgt.numpy(), sig.numpy())
    ece = compute_ece(errors, sig.numpy())
    ue_corr = compute_uncertainty_error_corr(errors, sig.numpy())

    conf_cov_list = []
    for p in range(SEQ_P):
        cov = float(((tgt[:,p]-mu[:,p]).abs()<=conf_q[p]).float().mean())
        conf_cov_list.append(round(cov, 4))
        
    print("\n" + "="*60)
    print("CROSS-DOMAIN COFFEE EVALUATION RESULTS")
    print("="*60)
    print(f"  MAE          : {mae:.4f}")
    print(f"  RMSE         : {rmse:.4f}")
    print(f"  SpearmanR    : {msr:.4f}")
    print(f"  LevelAcc     : {lacc:.4f}")
    print(f"  NonMonoRate  : {nmono:.4f}")
    print(f"  ECE          : {ece:.4f}")
    print(f"  UE-ErrCorr   : {ue_corr:.4f}")
    print(f"  PICP         : {picp:.4f}")
    print(f"  MPIW         : {mpiw:.4f}")
    print(f"  Conf Cov     : {conf_cov_list}")
    print("="*60)
    
    if mae < 0.08 and msr > 0.5:
        print("\n[Scientific Conclusion]:")
        print("Despite being evaluated on an entirely unseen crop/disease domain,")
        print("the prognosis model maintains strong forecasting accuracy (MAE) and")
        print("excellent progression correlation (SpearmanR).")
        print("This strongly supports the hypothesis that the learned morphology-temporal")
        print("manifold transfers effectively without requiring fine-tuning.")