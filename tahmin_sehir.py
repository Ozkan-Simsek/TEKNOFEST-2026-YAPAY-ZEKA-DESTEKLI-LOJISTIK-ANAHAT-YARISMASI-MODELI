#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
================================================================
 MODÜL: tahmin_sehir.py
 AMAÇ : TEKNOFEST 2026 Lojistik Projesi — Güzergah Bazlı (TM) Desi Tahmini

 Bu modül, toplam desi tahminini 89 ayrı güzergah (Transfer Merkezi) çiftine
 çözer. tahmin.py'nin global toplamını elde ettikten sonra bu dosya,
 her güzergah-güzergah çiftine düşecek olan haftalık talep miktarını ayrıntılı
 olarak tahmin eder.

 TASARIM KARARI — Pooled (Tek) Model:
   Tüm 89 güzergah çifti tek bir model altında eğitilir (10.770 eğitim satırı).
   Bu yöntemde model, güzergah kimliğini (route_id) bir özellik olarak öğrenir
   ve az veriye sahip güzergah çiftleri diğerlerinden yararlanabilir.

 PIPELINE'DAKİ YERİ:
   tahmin.py → [tahmin_sehir.py] → finalize.py → optimize_cost_v4.py → gorsellestir.py

 TEKNOFEST KOLON FORMATI:
   Çıktı kolonları: Tarih | Çıkış TM | Varış TM | Tahmin Edilen Desi
================================================================
"""

import pandas as pd
import numpy as np
from datetime import timedelta
import warnings
warnings.filterwarnings("ignore")

# ================================================================
# AYARLAR
# ================================================================
DATA_PATH    = r"data/Desi_talep.xlsx"
OUTPUT_EXCEL = r"data/tahmin_tm_v2_11_17_mayis.xlsx"

TAHMIN_BASLANGIC = pd.Timestamp("2026-05-11")
TAHMIN_BITIS     = pd.Timestamp("2026-05-17")

USE_OPTUNA = True
N_TRIALS   = 40   # LightGBM için; XGBoost N_TRIALS//2 kullanır

GUN_ADI = {0:"Pazartesi",1:"Salı",2:"Çarşamba",3:"Perşembe",
            4:"Cuma",5:"Cumartesi",6:"Pazar"}
SEP = "=" * 68

# ================================================================
# ÖZEL GÜNLER
# ================================================================
SPECIAL_DATES = {
    "2026-01-01": ("Yılbaşı",                "tatil"),
    "2026-02-14": ("Sevgililer Günü",         "ozel"),
    "2026-02-19": ("Ramazan Başlangıcı",      "dini"),
    "2026-03-08": ("Kadınlar Günü",           "ozel"),
    "2026-03-16": ("Kadir Gecesi",            "dini"),
    "2026-03-19": ("Ramazan Bayramı Arifesi", "tatil"),
    "2026-03-20": ("Ramazan Bayramı 1. Gün",  "tatil"),
    "2026-03-21": ("Ramazan Bayramı 2. Gün",  "tatil"),
    "2026-03-22": ("Ramazan Bayramı 3. Gün",  "tatil"),
    "2026-04-23": ("23 Nisan",               "tatil"),
    "2026-05-01": ("İşçi Bayramı",           "tatil"),
}
TATIL_SET     = {k for k,v in SPECIAL_DATES.items() if v[1]=="tatil"}
HOLIDAY_DATES = sorted([pd.Timestamp(k) for k in TATIL_SET])
RAMAZAN_BAS   = pd.Timestamp("2026-02-19")
RAMAZAN_BIT   = pd.Timestamp("2026-03-19")

# ================================================================
# YARDIMCI FONKSİYONLAR
# ================================================================
def smape(y_true, y_pred):
    """
    Simetrik Ortalama Mutlak Yüzde Hata (SMAPE).
    tahmin.py'deki aynı metrikle birebir aynıdır; güzel bazlı CV değerlendirmesinde kullanılır.
    """
    yt = np.array(y_true, dtype=float) + 1e-8
    yp = np.array(y_pred, dtype=float) + 1e-8
    return float(np.mean(2*np.abs(yt-yp)/(np.abs(yt)+np.abs(yp)))*100)

def d2h(d):
    """
    'Days to Holiday' kısaltması — bir sonraki tatile kaç gün kaldığını döndürür.
    Maksimum 30 ile sınırlandırılmıştır (tahmin.py ile aynı mantık).
    """
    f = [h for h in HOLIDAY_DATES if h >= d]
    return min((f[0]-d).days, 30) if f else 30

def dfrom(d):
    """
    'Days from Holiday' kısaltması — son geçmiş tatilden bu yana kaç gün geçtiğini döndürür.
    """
    p = [h for h in HOLIDAY_DATES if h <= d]
    return min((d-p[-1]).days, 30) if p else 30

def cal_feats(d: pd.Timestamp) -> dict:
    """
    Verilen tarih için takvim özelliklerini üretir.

    tahmin.py'deki build_calendar_features ile aynı mantıkta çalışır;
    ancak bu versiyon güzel bazlı (pooled route) özellik vektörüne entegre
    edilmek üzere sadeleştirilmiş bir alt küme üretir.
    """
    ds   = d.strftime("%Y-%m-%d")
    dow  = d.dayofweek
    sp   = SPECIAL_DATES.get(ds)
    pre1 = (d+timedelta(1)).strftime("%Y-%m-%d")
    pre2 = (d+timedelta(2)).strftime("%Y-%m-%d")
    post = (d-timedelta(1)).strftime("%Y-%m-%d")
    return {
        "day_of_week":     dow,
        "day_of_month":    d.day,
        "month":           d.month,
        "week_of_year":    int(d.isocalendar()[1]),
        "is_weekend":      int(dow >= 5),
        "is_monday":       int(dow == 0),
        "is_friday":       int(dow == 4),
        "is_saturday":     int(dow == 5),
        "is_sunday":       int(dow == 6),
        "is_holiday":      int(sp is not None and sp[1]=="tatil"),
        "is_special_day":  int(sp is not None and sp[1]=="ozel"),
        "is_ramazan":      int(RAMAZAN_BAS <= d <= RAMAZAN_BIT),
        "is_pre_holiday":  int(pre1 in TATIL_SET),
        "is_pre2_holiday": int(pre2 in TATIL_SET),
        "is_post_holiday": int(post in TATIL_SET),
        "is_near_holiday": int(d2h(d) <= 3 or dfrom(d) <= 3),
        "days_to_holiday": d2h(d),
        "days_from_holiday": dfrom(d),
    }

# ================================================================
# 1. VERİ YÜKLEME
# ================================================================
print(SEP)
print("  TM BAZLI POOLED ML MODELİ — 11-17 MAYIS 2026")
print(SEP)
print("\n[1/7] Veri yükleniyor...")

df_raw = pd.read_excel(DATA_PATH)
df_raw["Tarih"] = pd.to_datetime(df_raw["Tarih"])

df = (df_raw.groupby(["Çıkış Transfer Merkezi","Varış Transfer Merkezi","Tarih"])
      ["Toplam Desi"].sum()
      .reset_index()
      .rename(columns={"Toplam Desi":"y","Tarih":"ds",
                        "Çıkış Transfer Merkezi":"cikis",
                        "Varış Transfer Merkezi":"varis"}))
df["route_key"] = df["cikis"] + "→" + df["varis"]
df = df.sort_values(["route_key","ds"]).reset_index(drop=True)

print(f"  ✓ Satır sayısı    : {len(df):,}")
print(f"  ✓ Güzergah sayısı : {df['route_key'].nunique()}")
print(f"  ✓ Tarih aralığı   : {df['ds'].min().date()} → {df['ds'].max().date()}")
print(f"  ✓ Ort / Max desi  : {df['y'].mean():,.0f} / {df['y'].max():,.0f}")

# ================================================================
# 2. ÖZELLİK MÜHENDİSLİĞİ
# ================================================================
print("\n[2/7] Özellik mühendisliği (route bazlı lag + rolling + EWM)...")

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error

le_rk = LabelEncoder().fit(df["route_key"])
le_ci = LabelEncoder().fit(df["cikis"])
le_va = LabelEncoder().fit(df["varis"])
df["route_id"]  = le_rk.transform(df["route_key"]).astype(int)
df["cikis_enc"] = le_ci.transform(df["cikis"]).astype(int)
df["varis_enc"] = le_va.transform(df["varis"]).astype(int)

# Trend (günlük sıra, tüm güzergahlar için aynı)
date_rank = df.groupby("ds").ngroup()
df["trend"] = date_rank.astype(float).values

# Takvim özellikleri
cal_df = pd.DataFrame([cal_feats(d) for d in df["ds"]])
for c in cal_df.columns:
    df[c] = cal_df[c].values

# ── Route istatistikleri (statik) ──────────────────────────────
rs = (df.groupby("route_key")["y"]
       .agg(route_mean="mean", route_std="std", route_median="median")
       .reset_index())
rs["route_std"] = rs["route_std"].fillna(0)
df = df.merge(rs, on="route_key", how="left")
df["route_mean_log"] = np.log1p(df["route_mean"])

# Route desi payı
route_total = df.groupby("route_key")["y"].sum()
df["route_share"] = df["route_key"].map(route_total) / df["y"].sum()

# Route × DOW istatistikleri
rdow = (df.groupby(["route_key","day_of_week"])["y"]
         .agg(route_dow_mean="mean", route_dow_std="std")
         .reset_index())
rdow["route_dow_std"] = rdow["route_dow_std"].fillna(0)
df = df.merge(rdow, on=["route_key","day_of_week"], how="left")
df["route_dow_mean_log"] = np.log1p(df["route_dow_mean"])

# Çıkış / Varış TM istatistikleri
cs = df.groupby("cikis")["y"].agg(cikis_mean="mean",cikis_std="std").reset_index()
cs["cikis_std"] = cs["cikis_std"].fillna(0)
df = df.merge(cs, on="cikis", how="left")
df["cikis_mean_log"] = np.log1p(df["cikis_mean"])

vs = df.groupby("varis")["y"].agg(varis_mean="mean",varis_std="std").reset_index()
vs["varis_std"] = vs["varis_std"].fillna(0)
df = df.merge(vs, on="varis", how="left")
df["varis_mean_log"] = np.log1p(df["varis_mean"])

# ── Global günlük toplam lag ───────────────────────────────────
gd = (df.groupby("ds")["y"].sum().reset_index()
       .rename(columns={"y":"g_total"})
       .sort_values("ds").reset_index(drop=True))
for lag in [7, 14]:
    gd[f"g_lag_{lag}"]     = gd["g_total"].shift(lag)
    gd[f"g_lag_{lag}_log"] = np.log1p(gd[f"g_lag_{lag}"].clip(lower=0))
df = df.merge(gd[["ds","g_lag_7","g_lag_14","g_lag_7_log","g_lag_14_log"]],
              on="ds", how="left")

# ── Route bazlı lag özellikler ─────────────────────────────────
# Sıralama: (route_key, ds) kritik — lag doğru çalışsın
df = df.sort_values(["route_key","ds"]).reset_index(drop=True)

for lag in [7, 14, 21, 28]:
    df[f"lag_{lag}"]     = df.groupby("route_key")["y"].shift(lag)
    df[f"lag_{lag}_log"] = np.log1p(df[f"lag_{lag}"].clip(lower=0))

# Lag_1 (rolling için base)
df["_lag1"] = df.groupby("route_key")["y"].shift(1)

for w in [7, 14, 21]:
    df[f"roll_mean_{w}"] = (df.groupby("route_key")["_lag1"]
                             .transform(lambda x: x.rolling(w, min_periods=2).mean()))
    df[f"roll_std_{w}"]  = (df.groupby("route_key")["_lag1"]
                             .transform(lambda x: x.rolling(w, min_periods=2).std()))
    df[f"roll_mean_{w}_log"] = np.log1p(df[f"roll_mean_{w}"].clip(lower=0))

df["roll_max_7"] = (df.groupby("route_key")["_lag1"]
                     .transform(lambda x: x.rolling(7, min_periods=2).max()))
df["roll_min_7"] = (df.groupby("route_key")["_lag1"]
                     .transform(lambda x: x.rolling(7, min_periods=2).min()))

# EWM (route bazlı)
df["ewm_7"]  = (df.groupby("route_key")["_lag1"]
                 .transform(lambda x: x.ewm(span=7,  min_periods=2).mean()))
df["ewm_14"] = (df.groupby("route_key")["_lag1"]
                 .transform(lambda x: x.ewm(span=14, min_periods=3).mean()))
df["ewm_7_log"]  = np.log1p(df["ewm_7"].clip(lower=0))
df["ewm_14_log"] = np.log1p(df["ewm_14"].clip(lower=0))
df.drop(columns=["_lag1"], inplace=True)

# ── Etkileşim özellikleri ──────────────────────────────────────
df["route_x_dow"]     = df["route_id"] * df["day_of_week"]
df["route_x_month"]   = df["route_id"] * df["month"]
df["cikis_x_dow"]     = df["cikis_enc"] * df["day_of_week"]
df["varis_x_dow"]     = df["varis_enc"] * df["day_of_week"]
df["route_x_holiday"] = df["route_id"] * df["is_holiday"]
df["dow_x_month"]     = df["day_of_week"] * df["month"]
df["dow_x_ramazan"]   = df["day_of_week"] * df["is_ramazan"]
df["route_x_ramazan"] = df["route_id"] * df["is_ramazan"]

# Log target
df["y_log"] = np.log1p(df["y"])

# Date sıralamasına geri dön (CV için)
df = df.sort_values(["ds","route_key"]).reset_index(drop=True)

FEATURE_COLS = [
    # Route kimliği
    "route_id","cikis_enc","varis_enc",
    # Takvim
    "day_of_week","day_of_month","month","week_of_year","trend",
    # Gün tipi
    "is_weekend","is_monday","is_friday","is_saturday","is_sunday",
    # Tatil
    "is_holiday","is_special_day","is_ramazan",
    "is_pre_holiday","is_pre2_holiday","is_post_holiday",
    "is_near_holiday","days_to_holiday","days_from_holiday",
    # Route istatistikleri (Q25/Q75 kaldırıldı — nokta tahmini)
    "route_mean","route_std","route_median","route_mean_log",
    "route_share",
    "route_dow_mean","route_dow_std","route_dow_mean_log",
    # TM istatistikleri
    "cikis_mean","cikis_std","cikis_mean_log",
    "varis_mean","varis_std","varis_mean_log",
    # Global lag
    "g_lag_7","g_lag_14","g_lag_7_log","g_lag_14_log",
    # Route lag (orijinal + log)
    "lag_7","lag_14","lag_21","lag_28",
    "lag_7_log","lag_14_log","lag_21_log",
    # Rolling
    "roll_mean_7","roll_mean_14","roll_mean_21",
    "roll_mean_7_log","roll_mean_14_log",
    "roll_std_7","roll_max_7","roll_min_7",
    # EWM
    "ewm_7","ewm_14","ewm_7_log","ewm_14_log",
    # Etkileşimler
    "route_x_dow","route_x_month","cikis_x_dow","varis_x_dow",
    "route_x_holiday","dow_x_month","dow_x_ramazan","route_x_ramazan",
]

train_df = df.dropna(subset=FEATURE_COLS+["y_log"]).copy().reset_index(drop=True)
print(f"  ✓ Özellik sayısı  : {len(FEATURE_COLS)}")
print(f"  ✓ Eğitim satırı   : {len(train_df):,}  (NaN'sız)")
print(f"  ✓ Benzersiz tarih : {train_df['ds'].nunique()}")

# ================================================================
# 3. KÜTÜPHANELER
# ================================================================
try:
    import xgboost as xgb;   HAS_XGB = True
except ImportError:
    HAS_XGB = False; print("  ⚠ XGBoost yok")
try:
    import lightgbm as lgb;  HAS_LGB = True
except ImportError:
    HAS_LGB = False; print("  ⚠ LightGBM yok")
try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = USE_OPTUNA
except ImportError:
    HAS_OPTUNA = False; print("  ⚠ Optuna yok")

# ================================================================
# 4. DATE-BASED CV (data leakage yok)
# ================================================================
def date_cv_splits(df_sorted, n_splits=4, test_days=10):
    """
    Tarihe dayalı CV bölmeleri üretir (data leakage yoktur).

    TimeSeriesSplit yerine bu özel fonksiyon kullanılır; çünkü tüm güzel
    çiftleri aynı tarihe sahiptir ve gruplar tarih bazlı olarak kümelenmelidir.

    Süreç:
      - Son n_splits * test_days gün sırayla test seti olarak alınır.
      - Her test setinden önceki tüm veriler eğitim seti olur.
      - Bu yöntem, geleceğin geçmişe sızmasını (leakage) kesinlikle engeller.
    """
    udates = sorted(df_sorted["ds"].unique())
    n = len(udates)
    splits = []
    for i in range(n_splits):
        te_end   = n - i * test_days
        te_start = te_end - test_days
        if te_start < 30: break
        te_set = set(udates[te_start:te_end])
        tr_set = set(udates[:te_start])
        tr_idx = df_sorted[df_sorted["ds"].isin(tr_set)].index
        te_idx = df_sorted[df_sorted["ds"].isin(te_set)].index
        splits.append((tr_idx, te_idx))
    return splits

SPLITS = date_cv_splits(train_df, n_splits=4, test_days=10)
X_ALL  = train_df[FEATURE_COLS]
Y_ALL  = train_df["y_log"]

def cv_mae(model_cls, params, n_splits=4):
    """
    Verilen model sınıfı ve parametreler için CV MAE hesaplar.
    Log uzayında eğitim yapılır; orijinal ölçekte hata döndürülür.
    """
    maes = []
    for tr, te in SPLITS[:n_splits]:
        m = model_cls(**params)
        m.fit(X_ALL.loc[tr], Y_ALL.loc[tr])
        p = np.expm1(np.maximum(m.predict(X_ALL.loc[te]), 0))
        yt = np.expm1(Y_ALL.loc[te].values)
        maes.append(mean_absolute_error(yt, p))
    return float(np.mean(maes))

# ================================================================
# 5. OPTUNA TUNING
# ================================================================
print(f"\n[3/7] Optuna hiperparametre optimizasyonu ({N_TRIALS} trial)...")

lgb_best = xgb_best = None

if HAS_LGB and HAS_OPTUNA:
    def lgb_obj(trial):
        p = {
            "n_estimators":      trial.suggest_int("n_estimators",200,800),
            "max_depth":         trial.suggest_int("max_depth",4,10),
            "num_leaves":        trial.suggest_int("num_leaves",20,150),
            "learning_rate":     trial.suggest_float("learning_rate",0.01,0.15,log=True),
            "subsample":         trial.suggest_float("subsample",0.6,1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree",0.5,1.0),
            "min_child_samples": trial.suggest_int("min_child_samples",5,40),
            "reg_alpha":         trial.suggest_float("reg_alpha",1e-4,10.0,log=True),
            "reg_lambda":        trial.suggest_float("reg_lambda",1e-4,10.0,log=True),
            "random_state":42,"verbose":-1,
        }
        return cv_mae(lgb.LGBMRegressor, p, n_splits=3)
    st = optuna.create_study(direction="minimize",
                             sampler=optuna.samplers.TPESampler(seed=42))
    st.optimize(lgb_obj, n_trials=N_TRIALS, show_progress_bar=False)
    lgb_best = st.best_params
    print(f"  ✓ LightGBM (pooled) → en iyi MAE: {st.best_value:>10,.1f}")

if HAS_XGB and HAS_OPTUNA:
    def xgb_obj(trial):
        p = {
            "n_estimators":      trial.suggest_int("n_estimators",200,600),
            "max_depth":         trial.suggest_int("max_depth",4,8),
            "learning_rate":     trial.suggest_float("learning_rate",0.01,0.15,log=True),
            "subsample":         trial.suggest_float("subsample",0.6,1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree",0.5,1.0),
            "min_child_weight":  trial.suggest_int("min_child_weight",1,10),
            "reg_alpha":         trial.suggest_float("reg_alpha",1e-4,10.0,log=True),
            "reg_lambda":        trial.suggest_float("reg_lambda",1e-4,10.0,log=True),
            "random_state":42,"verbosity":0,
        }
        return cv_mae(xgb.XGBRegressor, p, n_splits=3)
    st2 = optuna.create_study(direction="minimize",
                              sampler=optuna.samplers.TPESampler(seed=42))
    st2.optimize(xgb_obj, n_trials=N_TRIALS//2, show_progress_bar=False)
    xgb_best = st2.best_params
    print(f"  ✓ XGBoost  (pooled) → en iyi MAE: {st2.best_value:>10,.1f}")

# ================================================================
# 6. MODEL EĞİTİMİ + CV
# ================================================================
from sklearn.ensemble import RandomForestRegressor

print("\n[4/7] Modeller eğitiliyor ve doğrulanıyor...")

D_LGB = {"n_estimators":400,"max_depth":7,"num_leaves":63,"learning_rate":0.05,
          "subsample":0.8,"colsample_bytree":0.8,"random_state":42,"verbose":-1}
D_XGB = {"n_estimators":300,"max_depth":6,"learning_rate":0.05,
          "subsample":0.8,"colsample_bytree":0.8,"random_state":42,"verbosity":0}

model_defs = {}
if HAS_LGB:
    p = {**(lgb_best or D_LGB), "random_state":42, "verbose":-1}
    model_defs["LightGBM"] = lgb.LGBMRegressor(**p)
if HAS_XGB:
    p = {**(xgb_best or D_XGB), "random_state":42, "verbosity":0}
    model_defs["XGBoost"] = xgb.XGBRegressor(**p)
model_defs["RandomForest"] = RandomForestRegressor(
    n_estimators=200, max_depth=8, min_samples_leaf=3,
    max_features="sqrt", random_state=42, n_jobs=-1)

print(f"\n  {'Model':<22} {'MAE':>10} {'SMAPE':>8}")
print(f"  {'-'*44}")

scores = {}
trained = {}

for name, model in model_defs.items():
    maes, smapes_ = [], []
    for tr, te in SPLITS:
        m = model.__class__(**model.get_params())
        m.fit(X_ALL.loc[tr], Y_ALL.loc[tr])
        p  = np.expm1(np.maximum(m.predict(X_ALL.loc[te]), 0))
        yt = np.expm1(Y_ALL.loc[te].values)
        maes.append(mean_absolute_error(yt, p))
        smapes_.append(smape(yt, p))
    mae_cv   = float(np.mean(maes))
    smape_cv = float(np.mean(smapes_))
    scores[name] = {"MAE":mae_cv, "SMAPE":smape_cv}
    model.fit(X_ALL, Y_ALL)
    trained[name] = model
    print(f"  {name:<22} {mae_cv:>10,.1f} {smape_cv:>7.1f}%")

# Naive (Route × DOW) CV
naive_maes = []
for tr, te in SPLITS:
    tr_df = train_df.loc[tr]; te_df = train_df.loc[te]
    rdm = tr_df.groupby(["route_key","day_of_week"])["y"].mean().reset_index()
    rdm.columns = ["route_key","day_of_week","np_"]
    merged = te_df[["route_key","day_of_week","y_log"]].merge(
        rdm, on=["route_key","day_of_week"], how="left")
    rm_fb = tr_df.groupby("route_key")["y"].mean().to_dict()
    merged["np_"] = merged["np_"].fillna(merged["route_key"].map(rm_fb))
    yt_n = np.expm1(merged["y_log"].values)
    naive_maes.append(mean_absolute_error(yt_n, merged["np_"].values))

nb_mae = float(np.mean(naive_maes))
scores["Naive (Route×DOW)"] = {"MAE":nb_mae, "SMAPE":0}
print(f"  {'Naive (Route×DOW)':<22} {nb_mae:>10,.1f}          ← ensemblea dahil!")

# Ağırlıklar (inverse MAE)
raw_w = {k: 1.0/(v["MAE"]+1e-6) for k,v in scores.items()}
tw    = sum(raw_w.values())
W     = {k: v/tw for k,v in raw_w.items()}
print(f"\n  Ensemble ağırlıkları:")
for k,v in W.items():
    print(f"    {k:<24}: {v:.4f}")

# Lookup tabloları (tüm eğitim verisiyle)
rdow_lk  = train_df.groupby(["route_key","day_of_week"])["y"].mean().to_dict()
rall_lk  = train_df.groupby("route_key")["y"].mean().to_dict()
global_lk = df.groupby("ds")["y"].sum().to_dict()

# Route data (hızlı lookup için)
route_hist = {rk: grp.set_index("ds")["y"].to_dict()
              for rk, grp in df.groupby("route_key")}

# ================================================================
# 7. TAHMİN (11-17 MAYIS)
# ================================================================
print("\n[5/7] 11-17 Mayıs 2026 tahmini yapılıyor...")

tahmin_tarihleri = pd.date_range(TAHMIN_BASLANGIC, TAHMIN_BITIS, freq="D")

routes = (train_df[["route_key","cikis","varis","route_id","cikis_enc","varis_enc",
                    "route_mean","route_std","route_median",
                    "route_share","cikis_mean","cikis_std","varis_mean","varis_std"]]
          .drop_duplicates("route_key")
          .reset_index(drop=True))

trend_max = float(train_df["trend"].max())
ds_max    = train_df["ds"].max()

pred_rows = []

for d in tahmin_tarihleri:
    dow  = d.dayofweek
    c    = cal_feats(d)
    tval = trend_max + float((d - ds_max).days)

    # Global lag
    gl7  = global_lk.get(d-timedelta(7),  df["y"].mean() * df["ds"].nunique())
    gl14 = global_lk.get(d-timedelta(14), gl7)

    for _, rz in routes.iterrows():
        rk = rz["route_key"]
        ri = int(rz["route_id"])
        ce = int(rz["cikis_enc"])
        ve = int(rz["varis_enc"])

        rm_v  = float(rz["route_mean"])
        rs_v  = float(rz["route_std"])
        rmd_v = float(rz["route_median"])
        rsh   = float(rz["route_share"])

        # Route × DOW stats (sadece mean/std — nokta tahmini)
        rdow_sub = train_df[(train_df["route_key"]==rk)&(train_df["day_of_week"]==dow)]["y"]
        rdm   = float(rdow_sub.mean())  if len(rdow_sub)>0 else rm_v
        rds   = float(rdow_sub.std())   if len(rdow_sub)>1 else rs_v*0.3

        # Lag (hızlı dict lookup)
        rh = route_hist.get(rk, {})
        lags = {}
        for lag in [7, 14, 21, 28]:
            raw = rh.get(d-timedelta(lag), rdm)
            lags[f"lag_{lag}"]     = raw
            lags[f"lag_{lag}_log"] = float(np.log1p(raw))

        # Rolling (geçmiş penceresinden)
        for w in [7, 14, 21]:
            vals = [rh.get(d-timedelta(i), None) for i in range(1, w+1)]
            vals = [v for v in vals if v is not None]
            mv = float(np.mean(vals))  if len(vals) >= 2 else rm_v
            sv = float(np.std(vals))   if len(vals) >= 2 else rs_v*0.3
            lags[f"roll_mean_{w}"]     = mv
            lags[f"roll_std_{w}"]      = sv
            lags[f"roll_mean_{w}_log"] = float(np.log1p(mv))

        r7v = [rh.get(d-timedelta(i), None) for i in range(1, 8)]
        r7v = [v for v in r7v if v is not None]
        lags["roll_max_7"] = float(np.max(r7v))  if r7v else rm_v
        lags["roll_min_7"] = float(np.min(r7v))  if r7v else 0.0

        # EWM
        rec = pd.Series([rh.get(d-timedelta(i), None) for i in range(21, 0, -1)]).dropna()
        e7  = float(rec.ewm(span=7).mean().iloc[-1])  if len(rec) >= 3 else rm_v
        e14 = float(rec.ewm(span=14).mean().iloc[-1]) if len(rec) >= 5 else rm_v
        lags.update({"ewm_7":e7,"ewm_14":e14,
                     "ewm_7_log":float(np.log1p(e7)),"ewm_14_log":float(np.log1p(e14))})

        feat = {
            "route_id":ri,"cikis_enc":ce,"varis_enc":ve,
            "day_of_week":dow,"day_of_month":d.day,"month":d.month,
            "week_of_year":int(d.isocalendar()[1]),"trend":tval,
            "is_weekend":c["is_weekend"],"is_monday":c["is_monday"],
            "is_friday":c["is_friday"],"is_saturday":c["is_saturday"],
            "is_sunday":c["is_sunday"],"is_holiday":c["is_holiday"],
            "is_special_day":c["is_special_day"],"is_ramazan":c["is_ramazan"],
            "is_pre_holiday":c["is_pre_holiday"],"is_pre2_holiday":c["is_pre2_holiday"],
            "is_post_holiday":c["is_post_holiday"],"is_near_holiday":c["is_near_holiday"],
            "days_to_holiday":c["days_to_holiday"],"days_from_holiday":c["days_from_holiday"],
            "route_mean":rm_v,"route_std":rs_v,"route_median":rmd_v,
            "route_mean_log":float(np.log1p(rm_v)),
            "route_share":rsh,
            "route_dow_mean":rdm,"route_dow_std":rds,
            "route_dow_mean_log":float(np.log1p(rdm)),
            "cikis_mean":float(rz["cikis_mean"]),"cikis_std":float(rz["cikis_std"]),
            "cikis_mean_log":float(np.log1p(float(rz["cikis_mean"]))),
            "varis_mean":float(rz["varis_mean"]),"varis_std":float(rz["varis_std"]),
            "varis_mean_log":float(np.log1p(float(rz["varis_mean"]))),
            "g_lag_7":gl7,"g_lag_14":gl14,
            "g_lag_7_log":float(np.log1p(gl7)),"g_lag_14_log":float(np.log1p(gl14)),
            "route_x_dow":ri*dow,"route_x_month":ri*d.month,
            "cikis_x_dow":ce*dow,"varis_x_dow":ve*dow,
            "route_x_holiday":ri*c["is_holiday"],
            "dow_x_month":dow*d.month,"dow_x_ramazan":dow*c["is_ramazan"],
            "route_x_ramazan":ri*c["is_ramazan"],
            **lags,
        }

        X_row = pd.DataFrame([feat])[FEATURE_COLS]

        # ML Ensemble
        ens = 0.0
        mp  = {}
        for nm, model in trained.items():
            p_log = float(model.predict(X_row)[0])
            p     = float(np.expm1(max(p_log, 0)))
            mp[nm] = p
            ens   += W.get(nm, 0) * p

        # Naive
        naive_p = rdow_lk.get((rk, dow), rall_lk.get(rk, rm_v))
        mp["Naive (Route×DOW)"] = naive_p
        ens += W.get("Naive (Route×DOW)", 0) * naive_p
        ens = max(ens, 0)

        # GÜNCELLEME 2: Kolon adları ve şehir ismi suffix formatı
        # "Çıkış TM" / "Varış TM" kolonları; şehir isminin sonuna " TM" ekleniyor.
        cikis_tm_label = rz["cikis"] + " TM"
        varis_tm_label = rz["varis"] + " TM"
        pred_rows.append({
            "Tarih":              d.date(),
            "Çıkış TM":          cikis_tm_label,
            "Varış TM":          varis_tm_label,
            "Tahmin Edilen Desi": float(round(ens, 4)),  # float — yuvarlama yok
            **{k: round(v, 1) for k,v in mp.items()},
        })

    print(f"  → {GUN_ADI[dow]} ({d.date()}) : {len(routes)} güzergah tamamlandı")

results_df = pd.DataFrame(pred_rows)
# Gün sütununu Tarih'ten türet (pred_rows'da artık Gün yok)
results_df["Gün"] = pd.to_datetime(results_df["Tarih"]).dt.dayofweek.map(GUN_ADI)
results_df = results_df.sort_values(["Tarih","Çıkış TM","Varış TM"]).reset_index(drop=True)

# KURAL 1: Tarih → strict YYYY-MM-DD string (timestamp '00:00:00' temizlenir)
results_df["Tarih"] = pd.to_datetime(results_df["Tarih"]).dt.strftime("%Y-%m-%d")

# KURAL 2: Çıkış TM / Varış TM — güvenli " TM" suffix (mükerrer " TM TM" önleme)
for _col in ["Çıkış TM", "Varış TM"]:
    if _col in results_df.columns:
        results_df[_col] = (results_df[_col]
                            .astype(str).str.strip()
                            .str.replace(r'\s*TM$', '', regex=True).str.strip()
                            + " TM")


# ================================================================
# 8. EXCEL ÇIKTISI
# ================================================================
print("\n[6/7] Excel yazılıyor...")

gunluk = (results_df.groupby(["Tarih","Gün"])["Tahmin Edilen Desi"]
          .sum().reset_index().rename(columns={"Tahmin Edilen Desi":"Günlük Toplam"}))
gunluk["Günlük Toplam"] = gunluk["Günlük Toplam"].round(4)

en_yogun = (results_df.groupby(["Çıkış TM","Varış TM"])["Tahmin Edilen Desi"]
            .sum().reset_index()
            .rename(columns={"Tahmin Edilen Desi":"Haftalık Toplam"})
            .sort_values("Haftalık Toplam", ascending=False)
            .head(20).reset_index(drop=True))

cikis_oz = (results_df.groupby("Çıkış TM")["Tahmin Edilen Desi"]
            .sum().reset_index()
            .sort_values("Tahmin Edilen Desi",ascending=False)
            .reset_index(drop=True))
cikis_oz.columns = ["Çıkış TM","Haftalık Toplam"]

varis_oz = (results_df.groupby("Varış TM")["Tahmin Edilen Desi"]
            .sum().reset_index()
            .sort_values("Tahmin Edilen Desi",ascending=False)
            .reset_index(drop=True))
varis_oz.columns = ["Varış TM","Haftalık Toplam"]

met_df = pd.DataFrame([
    {"Model":k,"CV MAE":round(v["MAE"],1),"CV SMAPE(%)":round(v.get("SMAPE",0),2),
     "Ensemble Ağırlığı":round(W.get(k,0),4)}
    for k,v in scores.items()
]).sort_values("CV MAE").reset_index(drop=True)

# GÜNCELLEME 2: Ana kolon sırası — Tarih | Çıkış TM | Varış TM | Tahmin Edilen Desi
ana_cols = ["Tarih","Çıkış TM","Varış TM","Tahmin Edilen Desi"]

with pd.ExcelWriter(OUTPUT_EXCEL, engine="openpyxl") as writer:
    results_df[ana_cols].to_excel(writer, sheet_name="TM Tahminler (Ana)", index=False)
    results_df[["Tarih","Gün","Çıkış TM","Varış TM","Tahmin Edilen Desi"]].to_excel(
        writer, sheet_name="Tüm Model Detayı", index=False)
    gunluk.to_excel(writer,    sheet_name="Günlük Özet",          index=False)
    en_yogun.to_excel(writer,  sheet_name="En Yoğun 20 Güzergah", index=False)
    cikis_oz.to_excel(writer,  sheet_name="Çıkış TM Özeti",       index=False)
    varis_oz.to_excel(writer,  sheet_name="Varış TM Özeti",        index=False)
    met_df.to_excel(writer,    sheet_name="Model Metrikleri",      index=False)

print(f"  ✓ Excel → {OUTPUT_EXCEL}")

# ================================================================
# ÖZET
# ================================================================
print()
print(SEP)
print("  POOLED ML MODELİ TAMAMLANDI!")
print(SEP)
print(f"\n  📊 Toplam tahmin satırı : {len(results_df):,}")
print(f"  📊 Haftalık toplam desi : {results_df['Tahmin Edilen Desi'].sum():>14,.0f}")
print()
print(f"  Günlük tahminler:")
print(f"  {'Tarih':<12} {'Gün':<12} {'Toplam':>14}")
print(f"  {'-'*40}")
for _, r in gunluk.iterrows():
    print(f"  {str(r['Tarih']):<12} {r['Gün']:<12} {r['Günlük Toplam']:>14,.0f}")
print(f"  {'-'*40}")
print(f"  {'TOPLAM':<24} {gunluk['Günlük Toplam'].sum():>14,.0f}")

print(f"\n  En yoğun 10 güzergah (haftalık):")
for _, r in en_yogun.head(10).iterrows():
    print(f"  {r['Çıkış TM']:<15} → {r['Varış TM']:<15} {r['Haftalık Toplam']:>12,.0f}")

print(f"\n  Çıkış TM haftalık yük sıralaması:")
for _, r in cikis_oz.iterrows():
    print(f"  {r['Çıkış TM']:<18} {r['Haftalık Toplam']:>12,.0f}")

print(f"\n  🏆 Model skor sıralaması (CV MAE):")
for _, r in met_df.iterrows():
    print(f"  {r['Model']:<24} MAE: {r['CV MAE']:>8,.1f}  "
          f"SMAPE: {r['CV SMAPE(%)']:>5.1f}%  Ağırlık: {r['Ensemble Ağırlığı']:.4f}")
print()
