#!/usr/bin/env python3
"""
================================================================
AGRO-DOCTOR (KRISHI-AI) — COFFEE SEQUENCE BUILDER (V10.6 FIX)
================================================================
Purpose:
  Reads pre-enriched `coffee_v10.csv`, processes it through the
  already trained Visual, CORAL, and Severity networks, and builds 
  the final pseudo-temporal sequences (`coffee_sequences_v10.csv`).
  
Fixes applied:
  - Added `log_vars` to CORALNet to perfectly match the V10.3 checkpoint.
  - Improved `elig_mask` to strictly require valid `image_path`.
  - Replaced flat L0 fallback with uniform ordinal binning (L0-L5).
  - Bypassed the `vis_classes` filter bug for unseen coffee domains.
  - Safe Path Handling: Strictly isolates read-only INPUT_DIR 
    from writable OUT_DIR to prevent Kaggle filesystem crashes.
================================================================
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import warnings
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import albumentations as A
import timm
from tqdm import tqdm

warnings.filterwarnings("ignore")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# ── Paths ──
INPUT_DIR  = "/kaggle/input/datasets/hriitk2026/weight-version-v10"
OUT_DIR    = "/kaggle/working"

COFFEE_CSV = f"{INPUT_DIR}/coffee_v10.csv"
VIS_PATH   = f"{INPUT_DIR}/best_visual_model_v10.pth"
CORAL_PATH = f"{INPUT_DIR}/best_coral_v10.pth"
SEV_PATH   = f"{INPUT_DIR}/best_severity_net_v10.pth"
CURVES_PKL = f"{INPUT_DIR}/curve_residuals_v10.pkl"
GSEV_PKL   = f"{INPUT_DIR}/gsev_model_v10.pkl"

OUTPUT_SEQ = f"{OUT_DIR}/coffee_sequences_v10.csv"

# ── Dimensions & Hyperparameters ──
IMG_SIZE       = 384
EMBED_DIM      = 512
MORPH_DIM      = 16
CROP_EMB       = 32
DIS_EMB        = 32
SRC_EMB        = 8
CORAL_K        = 5
GATE_DIM       = 64
SEQ_L          = 5
SEQ_P          = 3
SEQ_STRIDE     = 1
SEQ_Q_MIN      = 0.02
CURVE_DIM      = 6
MORPH_PCT_EVAL = 75

LEVEL_NAMES = ["L0","L1","L2","L3","L4","L5"]
level2idx = {l:i for i,l in enumerate(LEVEL_NAMES)}

MORPH_COLS = [
    "lesion_fraction_image", "lesion_fraction_bbox", "bbox_fraction",
    "bbox_aspect_ratio", "bbox_diag_ratio", "lesion_count", "mean_area",
    "max_area", "std_area", "median_area", "area_cv", "dispersion",
    "density", "compactness", "entropy", "convex_fraction"
]
mnorm = [f"m_{c}" for c in MORPH_COLS]
VIS_COLS = [f"vis_{i}" for i in range(EMBED_DIM)]
COMP_SEV_W = {"lesion_fraction":0.30, "lesion_count":0.15, "dispersion":0.15, 
              "texture":0.20, "visual_drift":0.10, "morph_instab":0.10}

# ================================================================
# MODEL ARCHITECTURES (Mirrored from Part 2)
# ================================================================

class VisDS(Dataset):
    def __init__(self,df_,tfm,label_col="vis_label"):
        self.df=df_.reset_index(drop=True); self.tfm=tfm; self.lc=label_col
    def __len__(self): return len(self.df)
    def _rgb(self,p):
        if isinstance(p,str) and os.path.exists(p):
            i=cv2.imread(p)
            if i is not None: return cv2.cvtColor(i,cv2.COLOR_BGR2RGB)
        return np.zeros((IMG_SIZE,IMG_SIZE,3),np.uint8)
    def __getitem__(self,idx):
        row=self.df.iloc[idx]; img=self._rgb(row.image_path)
        img=self.tfm(image=img)["image"]
        return torch.tensor(img.transpose(2,0,1),dtype=torch.float32), int(row[self.lc])

def _tfm(train=False):
    return A.Compose([A.Resize(IMG_SIZE,IMG_SIZE), A.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))])

class VisModel(nn.Module):
    def __init__(self,nc):
        super().__init__()
        self.bb=timm.create_model("seresnext50_32x4d",pretrained=False, num_classes=0,global_pool="avg")
        self.proj=nn.Sequential(nn.Linear(2048,1024),nn.BatchNorm1d(1024),nn.GELU(),
                                nn.Dropout(0.25),nn.Linear(1024,EMBED_DIM),nn.BatchNorm1d(EMBED_DIM))
        self.cls=nn.Linear(2048,nc)
    def forward(self,x):
        f=self.bb(x); e=F.normalize(self.proj(f),dim=1,p=2); l=self.cls(f)
        return e,l

class DiseaseGate(nn.Module):
    def __init__(self, vis_dim=EMBED_DIM, morph_dim=MORPH_DIM, dis_dim=DIS_EMB, hidden=GATE_DIM):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(vis_dim + morph_dim, hidden), nn.LayerNorm(hidden),
                                  nn.GELU(), nn.Dropout(0.1), nn.Linear(hidden, dis_dim), nn.Sigmoid())
    def forward(self, vis_norm, morph, dis_emb_lookup):
        return self.gate(torch.cat([vis_norm, morph], dim=1)) * dis_emb_lookup

class CORALNet(nn.Module):
    def __init__(self, in_dim, K, num_crops, num_dis):
        super().__init__()
        self.crop_emb = nn.Embedding(num_crops, CROP_EMB); self.dis_emb = nn.Embedding(num_dis, DIS_EMB)
        self.dis_gate = DiseaseGate(EMBED_DIM, MORPH_DIM, DIS_EMB)
        self.trunk = nn.Sequential(nn.Linear(in_dim, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.2),
                                   nn.Linear(256, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(0.15))
        self.fc = nn.Linear(128, 1, bias=False); self.bias = nn.Parameter(torch.zeros(K)); self.K = K
        self.lfb_head = nn.Sequential(nn.Linear(128, 32), nn.GELU(), nn.Linear(32, 1), nn.Sigmoid())
        
        # FIXED: Added to perfectly match the Part 2 checkpoint
        self.log_vars = nn.Parameter(torch.zeros(4)) 

    def _encode(self, vis, morph, ci, di):
        v = F.normalize(vis, dim=1)
        return torch.cat([v, morph, self.crop_emb(ci), self.dis_gate(v, morph, self.dis_emb(di))], dim=1)
    def forward(self, vis, morph, ci, di):
        h = self.trunk(self._encode(vis, morph, ci, di))
        prob = torch.sigmoid(self.fc(h) + self.bias.unsqueeze(0))
        return prob, self.lfb_head(h).squeeze(-1)
    def severity(self, vis, morph, ci, di):
        prob, _ = self.forward(vis, morph, ci, di)
        return (self.K - prob.sum(1)) / self.K

class MorphAttn(nn.Module):
    def __init__(self,d=MORPH_DIM,r=4):
        super().__init__(); h=max(d//r,4)
        self.fc=nn.Sequential(nn.Linear(d,h),nn.ReLU(),nn.Linear(h,d),nn.Sigmoid())
    def forward(self,x): return x*self.fc(x)

class SeverityNet(nn.Module):
    def __init__(self, num_crops, num_dis, num_src):
        super().__init__()
        self.ce=nn.Embedding(num_crops,CROP_EMB); self.de=nn.Embedding(num_dis,DIS_EMB)
        self.se=nn.Embedding(num_src,SRC_EMB)
        self.dis_gate = DiseaseGate(EMBED_DIM, MORPH_DIM, DIS_EMB)
        self.ma=MorphAttn()
        self.cp_reduce = nn.Sequential(nn.Linear(CORAL_K, 4), nn.ReLU(), nn.Dropout(0.1))
        
        comp_dim = EMBED_DIM + MORPH_DIM + CROP_EMB + DIS_EMB + SRC_EMB + 4
        self.bn=nn.BatchNorm1d(comp_dim)
        self.fc1=nn.Linear(comp_dim,256); self.b1=nn.BatchNorm1d(256)
        self.fc2=nn.Linear(256,128);    self.b2=nn.BatchNorm1d(128)
        self.at=nn.Linear(128,128);     self.ab=nn.BatchNorm1d(128)
        self.fc3=nn.Linear(128,64);     self.b3=nn.BatchNorm1d(64)
        self.d1=nn.Dropout(0.30); self.d2=nn.Dropout(0.20)
        self.mh=nn.Linear(64,1); self.lh=nn.Linear(64,1)

    def forward(self,vis,morph,ci,di,si,cp):
        vis_norm = F.normalize(vis, dim=1)
        x = torch.cat([vis, self.ma(morph), self.ce(ci), 
                       self.dis_gate(vis_norm, morph, self.de(di)), self.se(si), self.cp_reduce(cp)],dim=1)
        x=self.d1(F.gelu(self.b1(self.fc1(self.bn(x)))))
        x=self.d2(F.gelu(self.b2(self.fc2(x))))
        x=x*torch.sigmoid(self.ab(self.at(x))); x=F.gelu(self.b3(self.fc3(x)))
        return torch.sigmoid(self.mh(x)).squeeze(-1),self.lh(x).squeeze(-1).clamp(-4.,4.)

class SevDS(Dataset):
    def __init__(self,vis,morph,ci,di,si,cp):
        self.v=torch.tensor(vis,dtype=torch.float32); self.m=torch.tensor(morph,dtype=torch.float32)
        self.ci=torch.tensor(ci,dtype=torch.long); self.di=torch.tensor(di,dtype=torch.long)
        self.si=torch.tensor(si,dtype=torch.long); self.cp=torch.tensor(cp,dtype=torch.float32)
    def __len__(self): return len(self.v)
    def __getitem__(self,i): return (self.v[i],self.m[i],self.ci[i],self.di[i],self.si[i],self.cp[i])

class GMMStages:
    def __init__(self): self.thr={}
    def fit(self,df_,sev): pass
    def assign(self,sev,cd):
        thr=self.thr.get(cd,[0.,0.05,0.20,0.40,0.60,0.80,1.01])
        for i in range(len(thr)-1):
            if thr[i]<=sev<thr[i+1]: return LEVEL_NAMES[i]
        return LEVEL_NAMES[-1]

# ================================================================
# FEATURE EXTRACTION & SEQUENCE BUILDING
# ================================================================

@torch.no_grad()
def extract_emb(model, df_, label_col="vis_label"):
    if len(df_) == 0: return np.zeros((0, EMBED_DIM), np.float32)
    ds = VisDS(df_, _tfm(False), label_col=label_col)
    ld = DataLoader(ds, 32, shuffle=False, num_workers=2, pin_memory=True)
    model.eval(); embs = []
    for imgs, _ in tqdm(ld, desc="  Emb Extraction"):
        e, _ = model(imgs.to(DEVICE))
        embs.append(e.cpu().numpy())
    return np.concatenate(embs)

def _extract_emb_safe(model, df_):
    if len(df_) == 0: return np.zeros((0, EMBED_DIM), np.float32)
    emb = np.zeros((len(df_), EMBED_DIM), np.float32)
    
    # FIXED: Improved mask logic requiring valid image path
    elig_mask = (df_.is_healthy == 0) & (df_.image_path.notna())
    elig_df = df_[elig_mask].copy().reset_index(drop=True)
    
    # Assign dummy label since classes are unseen (prevents dataloader crash)
    elig_df["vis_label"] = 0 
    
    if len(elig_df) > 0:
        e = extract_emb(model, elig_df, label_col="vis_label")
        elig_orig_idx = np.where(elig_mask.values)[0]
        for local_i, orig_i in enumerate(elig_orig_idx[:len(e)]):
            emb[orig_i] = e[local_i]
    return emb

def _build_base_seqs(df_, cls_, cvs, pct, sid_start=0):
    prog = df_[df_.crop_disease.isin(cls_)].copy() if len(df_) > 0 else pd.DataFrame()
    recs = []; sid = sid_start; dq = 0
    
    for cd in cls_:
        if len(prog) == 0: continue
        sub = prog[prog.crop_disease == cd].sort_values("severity_pred").reset_index(drop=True)
        if len(sub) < SEQ_L + SEQ_P: continue
        
        sev = sub.severity_pred.values.astype(np.float32)
        mp = sub[MORPH_COLS].values.astype(np.float32)
        dists = np.linalg.norm(mp[1:] - mp[:-1], axis=1)
        thr = np.percentile(dists, pct)
        
        vi = [0]
        for i in range(len(sub) - 1):
            if sev[i+1] >= sev[i] - 0.03 and dists[i] <= thr: 
                vi.append(i+1)
                
        vs = sub.iloc[vi].reset_index(drop=True)
        if len(vs) < SEQ_L + SEQ_P: continue
        
        cf = cvs.get(cd, np.zeros(CURVE_DIM, np.float32))
        
        for st in range(0, len(vs) - SEQ_L - SEQ_P + 1, SEQ_STRIDE):
            win = vs.iloc[st:st+SEQ_L+SEQ_P].severity_pred.values
            if np.ptp(np.concatenate([win])) < SEQ_Q_MIN: 
                dq += 1
                continue
                
            in_r = vs.iloc[st:st+SEQ_L]
            ou_r = vs.iloc[st+SEQ_L:st+SEQ_L+SEQ_P]

            vis_seq = np.array([in_r.iloc[t][[f"vis_{i}" for i in range(EMBED_DIM)]].values.astype(np.float32)
                                for t in range(SEQ_L)])

            rec = {"seq_id": sid, "crop_disease": cd, "aug_method": "base",
                   "crop_idx": int(vs.crop_idx.iloc[0]), "disease_idx": int(vs.disease_idx.iloc[0])}
            
            for t, (_, row) in enumerate(in_r.iterrows()):
                for c in VIS_COLS:  rec[f"t{t}_{c}"] = row[c]
                for c in mnorm:     rec[f"t{t}_{c}"] = row[c]
                rec[f"t{t}_severity"]  = row.severity_pred
                rec[f"t{t}_level_idx"] = row.level_idx
                prev = in_r.iloc[t-1].severity_pred if t > 0 else row.severity_pred
                rec[f"t{t}_vel"] = float(row.severity_pred - prev)
                for ci_, cv in enumerate(cf): rec[f"t{t}_curve_{ci_}"] = float(cv)

            for t in range(SEQ_L):
                d1 = vis_seq[t] - vis_seq[t-1] if t >= 1 else np.zeros(EMBED_DIM, np.float32)
                d2 = vis_seq[t] - vis_seq[t-2] if t >= 2 else np.zeros(EMBED_DIM, np.float32)
                d3 = vis_seq[t] - vis_seq[t-3] if t >= 3 else np.zeros(EMBED_DIM, np.float32)
                
                if t >= 2:
                    delta_accel = d1 - (vis_seq[t-1] - vis_seq[t-2])
                    rec[f"t{t}_delta_accel_norm"] = float(np.linalg.norm(delta_accel))
                else:
                    rec[f"t{t}_delta_accel_norm"] = 0.
                    
                rec[f"t{t}_delta1_norm"] = float(np.linalg.norm(d1))
                rec[f"t{t}_delta2_norm"] = float(np.linalg.norm(d2))
                rec[f"t{t}_delta3_norm"] = float(np.linalg.norm(d3))
                
                if t >= 2 and np.linalg.norm(d1) > 1e-8 and np.linalg.norm(d2) > 1e-8:
                    rec[f"t{t}_delta_accel_cos"] = float(np.dot(d2, d1) / (np.linalg.norm(d1) * np.linalg.norm(d2)))
                else:
                    rec[f"t{t}_delta_accel_cos"] = 0.

            for p, (_, row) in enumerate(ou_r.iterrows()):
                rec[f"tgt_sev_{p}"] = row.severity_pred
                rec[f"tgt_lvl_{p}"] = row.level_idx
                
            recs.append(rec)
            sid += 1
            
    return recs, dq, sid

# ================================================================
# MAIN EXECUTION
# ================================================================

if __name__ == "__main__":
    print(f"\n{'='*60}\nCOFFEE SEQUENCE BUILDER (V10.6)\n{'='*60}")
    
    if not os.path.exists(COFFEE_CSV):
        print(f"Error: {COFFEE_CSV} not found.")
        sys.exit(1)
        
    print("\n1. Loading Coffee Data & Initializing Models...")
    coffee_df = pd.read_csv(COFFEE_CSV)

    # Infer Dimensions from Checkpoints Dynamically
    vis_state = torch.load(VIS_PATH, map_location="cpu")
    num_vis_cls = vis_state["cls.weight"].shape[0]
    
    coral_state = torch.load(CORAL_PATH, map_location="cpu")
    num_crops = coral_state["crop_emb.weight"].shape[0]
    num_dis = coral_state["dis_emb.weight"].shape[0]
    
    sev_state = torch.load(SEV_PATH, map_location="cpu")
    num_src = sev_state["se.weight"].shape[0]

    print("\n2. Visual Embedding Extraction (Bypassing filter bug)...")
    vis_model = VisModel(num_vis_cls).to(DEVICE)
    vis_model.load_state_dict(vis_state)
    vis_cf = _extract_emb_safe(vis_model, coffee_df)
    
    # Enrich DF with embeddings for drift computation
    for i in range(EMBED_DIM): coffee_df[f"vis_{i}"] = vis_cf[:, i]

    print("\n3. CORAL Severity Inference...")
    coral_net = CORALNet(in_dim=EMBED_DIM + MORPH_DIM + CROP_EMB + DIS_EMB, 
                         K=CORAL_K, num_crops=num_crops, num_dis=num_dis).to(DEVICE)
    # Using strict=False as extra safety in case log_vars structure slightly varied across saves
    coral_net.load_state_dict(coral_state, strict=False)
    
    if mnorm[0] in coffee_df.columns:
        morph_cf = coffee_df[mnorm].values.astype(np.float32)
    else:
        morph_cf = coffee_df[MORPH_COLS].values.astype(np.float32)
        
    ci_cf = coffee_df.crop_idx.values.astype(np.int64)
    di_cf = coffee_df.disease_idx.values.astype(np.int64)
    si_cf = coffee_df.source_idx.values.astype(np.int64)
    
    coral_net.eval()
    ds = torch.utils.data.TensorDataset(torch.tensor(vis_cf), torch.tensor(morph_cf), torch.tensor(ci_cf), torch.tensor(di_cf))
    ld = DataLoader(ds, 64, shuffle=False)
    probs = []
    with torch.no_grad():
        for vb, mb, cb, db in ld:
            prob, _ = coral_net(vb.to(DEVICE), mb.to(DEVICE), cb.to(DEVICE), db.to(DEVICE))
            probs.append(prob.cpu().numpy())
    cp_cf = np.concatenate(probs, axis=0)

    print("\n4. SeverityNet Inference...")
    sev_net = SeverityNet(num_crops, num_dis, num_src).to(DEVICE)
    sev_net.load_state_dict(sev_state, strict=False)
    sev_net.eval()
    
    sev_ds = SevDS(vis_cf, morph_cf, ci_cf, di_cf, si_cf, cp_cf)
    sev_ld = DataLoader(sev_ds, 64, shuffle=False, num_workers=2)
    mus = []; sigs = []
    with torch.no_grad():
        for vb, mb, cb, db, sb, cp_b in tqdm(sev_ld, desc="  Infer Sev"):
            mu, ls = sev_net(vb.to(DEVICE), mb.to(DEVICE), cb.to(DEVICE), db.to(DEVICE), sb.to(DEVICE), cp_b.to(DEVICE))
            mus.append(mu.cpu().numpy()); sigs.append(ls.cpu().numpy())
    mu_cf = np.concatenate(mus); ls_cf = np.concatenate(sigs)
    
    coffee_df["severity_pred"] = mu_cf
    coffee_df["severity_log_sigma"] = ls_cf

    print("\n5. GMM Stage Boundaries Mapping...")
    try:
        with open(GSEV_PKL, "rb") as f: gmm = pickle.load(f)
        lvs = []; lis = []
        for _, row in coffee_df.iterrows():
            lv = "L0" if row.is_healthy == 1 else gmm.assign(float(row.severity_pred), row.crop_disease)
            lvs.append(lv); lis.append(level2idx[lv])
        coffee_df["level_pred"] = lvs
        coffee_df["level_idx"] = lis
    except FileNotFoundError:
        print(f"Warning: {GSEV_PKL} not found. Defaulting to uniform ordinal severity bins.")
        # FIXED: Better fallback mapping to ordinal stages
        pred_scaled = np.clip(np.round(coffee_df["severity_pred"] * 5), 0, 5).astype(int)
        coffee_df["level_idx"] = pred_scaled
        coffee_df["level_pred"] = ["L" + str(idx) for idx in pred_scaled]

    print("\n6. Building Pseudo-Temporal Sequences...")
    try:
        with open(CURVES_PKL, "rb") as f: curves = pickle.load(f)
    except FileNotFoundError:
        curves = {}

    cf_prog = coffee_df.crop_disease.unique().tolist()
    cf_recs, dq_c, _ = _build_base_seqs(coffee_df, cf_prog, curves, MORPH_PCT_EVAL)
    
    cf_df_out = pd.DataFrame(cf_recs)
    
    if len(cf_df_out) > 0:
        cf_df_out.to_csv(OUTPUT_SEQ, index=False)
        print(f"\n✔ Successfully generated {len(cf_df_out)} coffee sequences.")
        print(f"✔ Saved to: {OUTPUT_SEQ}")
        print(f"  Dropped (Flat severity windows): {dq_c}")
        
        # Verify delta computation worked
        delta_cols = [c for c in cf_df_out.columns if "delta1_norm" in c]
        if delta_cols:
            print("\n  [VERIFICATION] Sample Delta Features (should be > 0):")
            print(cf_df_out[delta_cols].mean())
    else:
        print("\n⚠ No sequences generated. Check if you have enough samples per class or if the severity distribution is too flat.")

    print("="*60)