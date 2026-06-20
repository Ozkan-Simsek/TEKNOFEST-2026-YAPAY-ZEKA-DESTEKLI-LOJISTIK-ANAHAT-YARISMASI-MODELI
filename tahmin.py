#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
================================================================
 MODÜL: tahmin.py
 AMAÇ : TEKNOFEST 2026 Lojistik Projesi — Toplam Desi Talep Tahmini

 Bu modül, kargo firmasının geçmiş günlük toplam desi verilerini kullanarak
 11-17 Mayıs 2026 haftası için talep tahmini üretir. Tahminler, pipeline'ın
 sonraki aşamasında (tahmin_sehir.py → finalize.py → optimize_cost_v4.py)
 araç atama ve maliyet optimizasyonunun temelini oluşturur.

 PIPELINE'DAKİ YERİ:
   [tahmin.py] → tahmin_sehir.py → finalize.py → optimize_cost_v4.py → gorsellestir.py
================================================================
"""

import pandas as pd
import numpy as np
from datetime import timedelta
import warnings
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from matplotlib.patches import Patch

warnings.filterwarnings("ignore")

# ================================================================
# AYARLAR
# ================================================================
DATA_PATH     = r"data/Desi_talep.xlsx"
OUTPUT_EXCEL  = r"data/tahminler_v2.xlsx"
OUTPUT_GRAFIK = r"data/tahmin_grafik_v2.png"

TAHMIN_BASLANGIC = pd.Timestamp("2026-05-11")
TAHMIN_BITIS     = pd.Timestamp("2026-05-17")

USE_OPTUNA = True   # False → varsayılan parametreler (hızlı çalışır)
N_TRIALS   = 40    # Artırınca daha iyi sonuç ama daha yavaş

GUN_ADI  = {0:"Pazartesi",1:"Salı",2:"Çarşamba",3:"Perşembe",
             4:"Cuma",5:"Cumartesi",6:"Pazar"}
GUN_KISA = {0:"Pzt",1:"Sal",2:"Çar",3:"Per",4:"Cum",5:"Cmt",6:"Paz"}
SEP = "=" * 68

# ================================================================
# ÖZEL GÜNLER
# ================================================================
SPECIAL_DATES = {
    "2026-01-01": ("Yılbaşı",                    "tatil"),
    "2026-02-14": ("Sevgililer Günü",             "ozel"),
    "2026-02-19": ("Ramazan Başlangıcı",          "dini"),
    "2026-03-08": ("Kadınlar Günü",               "ozel"),
    "2026-03-16": ("Kadir Gecesi",                "dini"),
    "2026-03-19": ("Ramazan Bayramı Arifesi",     "tatil"),
    "2026-03-20": ("Ramazan Bayramı 1. Gün",      "tatil"),
    "2026-03-21": ("Ramazan Bayramı 2. Gün",      "tatil"),
    "2026-03-22": ("Ramazan Bayramı 3. Gün",      "tatil"),
    "2026-04-23": ("23 Nisan",                    "tatil"),
    "2026-05-01": ("İşçi Bayramı",               "tatil"),
}
TATIL_SETLER   = {k for k, v in SPECIAL_DATES.items() if v[1] == "tatil"}
HOLIDAY_DATES  = sorted([pd.Timestamp(k) for k in TATIL_SETLER])
RAMAZAN_BAS    = pd.Timestamp("2026-02-19")
RAMAZAN_BIT    = pd.Timestamp("2026-03-19")

# ================================================================
# METRİK VE YARDIMCI FONKSİYONLAR
# ================================================================
def smape(y_true, y_pred):
    """
    Simetrik Ortalama Mutlak Yüzde Hata (SMAPE) metriği.
    Düşük talep değerlerinde klasik MAPE'ye kıyasla daha dengeli sonuç verir.
    Değer 0'a yaklaştıkça payda sıfırlanmaz (1e-8 ile korunur).
    Döndürür: % cinsinden hata oranı (0–200 aralığı)
    """
    yt = np.array(y_true, dtype=float) + 1e-8
    yp = np.array(y_pred, dtype=float) + 1e-8
    return float(np.mean(2*np.abs(yt-yp)/(np.abs(yt)+np.abs(yp)))*100)

def mae_score(y_true, y_pred):
    """Ortalama Mutlak Hata (MAE) — basit hata metriği yardımcısı."""
    return float(np.mean(np.abs(np.array(y_true)-np.array(y_pred))))

def days_to_next_holiday(d):
    """
    Bir sonraki resmi tatile kaç gün kaldığını hesaplar.
    Tatil yakınlığı, talep tahmininde önemli bir özellik olarak kullanılır
    (örn. Ramazan Bayramı öncesi kargo hacminin artması).
    Maksimum değer 30 gün ile sınırlandırılmıştır (aşırı uzak tatiller anlamsız).
    """
    future = [h for h in HOLIDAY_DATES if h >= d]
    return min((future[0]-d).days, 30) if future else 30

def days_from_last_holiday(d):
    """
    Son geçmiş resmi tatilden bu yana kaç gün geçtiğini hesaplar.
    Tatil sonrası talep patikasını modellemek için kullanılır.
    Maksimum değer 30 gün ile sınırlandırılmıştır.
    """
    past = [h for h in HOLIDAY_DATES if h <= d]
    return min((d-past[-1]).days, 30) if past else 30

def build_calendar_features(d: pd.Timestamp) -> dict:
    """
    Verilen tarih için kapsamlı takvim özellik sözlüğü üretir.

    Üretilen özellik kategorileri:
      - Temel takvim: gün, ay, hafta, trend değerleri
      - Gün tipi: hafta içi/sonu, Pazartesi, Cuma vb. bayrakları
      - Tatil bilgisi: is_holiday, tatil öncesi/sonrası bayrakları
      - Ramazan dönemi bayrağı (dini gün etkisi)
      - Tatile yakınlık: days_to/from_holiday (sürekli değişken)

    Bu özellikler FEATURE_COLS listesine dahil edilip model eğitiminde kullanılır.
    """
    ds       = d.strftime("%Y-%m-%d")
    dow      = d.dayofweek
    spec     = SPECIAL_DATES.get(ds)
    pre1_key = (d+timedelta(days=1)).strftime("%Y-%m-%d")
    pre2_key = (d+timedelta(days=2)).strftime("%Y-%m-%d")
    post_key = (d-timedelta(days=1)).strftime("%Y-%m-%d")
    d2h      = days_to_next_holiday(d)
    dfh      = days_from_last_holiday(d)
    return {
        "ds":                 d,
        "day_of_week":        dow,
        "day_of_month":       d.day,
        "month":              d.month,
        "week_of_year":       int(d.isocalendar()[1]),
        "is_weekend":         int(dow >= 5),
        "is_monday":          int(dow == 0),
        "is_tuesday":         int(dow == 1),
        "is_wednesday":       int(dow == 2),
        "is_thursday":        int(dow == 3),
        "is_friday":          int(dow == 4),
        "is_saturday":        int(dow == 5),
        "is_sunday":          int(dow == 6),
        "is_holiday":         int(spec is not None and spec[1]=="tatil"),
        "is_special_day":     int(spec is not None and spec[1]=="ozel"),
        "is_dini_gun":        int(spec is not None and spec[1]=="dini"),
        "is_ramazan":         int(RAMAZAN_BAS <= d <= RAMAZAN_BIT),
        "is_pre_holiday":     int(pre1_key in TATIL_SETLER),
        "is_pre2_holiday":    int(pre2_key in TATIL_SETLER),
        "is_post_holiday":    int(post_key in TATIL_SETLER),
        "days_to_holiday":    d2h,
        "days_from_holiday":  dfh,
        "is_near_holiday":    int(d2h <= 3 or dfh <= 3),
    }

# ================================================================
# 1. VERİ YÜKLEME & AGREGASYON
# ================================================================
print(SEP)
print("  DESİ TALEP TAHMİN SİSTEMİ v2 — 11-17 MAYIS 2026")
print(SEP)
print("\n[1/7] Veri yükleniyor ve günlük toplama yapılıyor...")

df = pd.read_excel(DATA_PATH)
df["Tarih"] = pd.to_datetime(df["Tarih"])

daily = (df.groupby("Tarih")["Toplam Desi"]
           .sum()
           .reset_index()
           .rename(columns={"Tarih":"ds","Toplam Desi":"y"})
           .sort_values("ds")
           .reset_index(drop=True))

print(f"  ✓ Günlük kayıt    : {len(daily)} gün")
print(f"  ✓ Tarih aralığı   : {daily['ds'].min().date()} → {daily['ds'].max().date()}")
print(f"  ✓ Ort. / Min / Max: {daily['y'].mean():>12,.0f} / {daily['y'].min():,.0f} / {daily['y'].max():,.0f}")

# ================================================================
# 2. ÖZELLİK MÜHENDİSLİĞİ
# ================================================================
print("\n[2/7] Kapsamlı özellik mühendisliği yapılıyor...")

# Takvim & tatil özellikleri
cal_df = pd.DataFrame([build_calendar_features(d) for d in daily["ds"]])
daily  = daily.merge(cal_df, on="ds", how="left")

# Trend
daily["trend"]    = np.arange(len(daily), dtype=float)
daily["trend_sq"] = daily["trend"] ** 2

# Lag özellikleri (orijinal + log uzayı)
for lag in [7, 14, 21, 28]:
    raw = daily["y"].shift(lag)
    daily[f"lag_{lag}"]     = raw
    daily[f"lag_{lag}_log"] = np.log1p(raw.clip(lower=0))

# Rolling istatistikler
for w in [7, 14, 21]:
    base = daily["y"].shift(1)
    daily[f"roll_mean_{w}"]     = base.rolling(w, min_periods=3).mean()
    daily[f"roll_std_{w}"]      = base.rolling(w, min_periods=3).std()
    daily[f"roll_mean_{w}_log"] = np.log1p(daily[f"roll_mean_{w}"].clip(lower=0))

base7 = daily["y"].shift(1)
daily["roll_max_7"]    = base7.rolling(7, min_periods=3).max()
daily["roll_min_7"]    = base7.rolling(7, min_periods=3).min()
daily["roll_median_7"] = base7.rolling(7, min_periods=3).median()

# Exponential Weighted Mean
daily["ewm_7"]      = daily["y"].shift(1).ewm(span=7,  min_periods=3).mean()
daily["ewm_14"]     = daily["y"].shift(1).ewm(span=14, min_periods=5).mean()
daily["ewm_7_log"]  = np.log1p(daily["ewm_7"].clip(lower=0))
daily["ewm_14_log"] = np.log1p(daily["ewm_14"].clip(lower=0))

# DOW bazlı istatistikler
_dow = (daily.groupby("day_of_week")["y"]
        .agg(dow_mean="mean", dow_median="median",
             dow_std="std",
             dow_q25=lambda x: x.quantile(0.25),
             dow_q75=lambda x: x.quantile(0.75))
        .reset_index())
daily = daily.merge(_dow, on="day_of_week", how="left")
daily["dow_mean_log"] = np.log1p(daily["dow_mean"].clip(lower=0))

# Ay bazlı istatistikler
_month = (daily.groupby("month")["y"]
          .agg(month_mean="mean", month_median="median")
          .reset_index())
daily = daily.merge(_month, on="month", how="left")

# Etkileşim özellikleri
daily["dow_x_month"]       = daily["day_of_week"] * daily["month"]
daily["dow_x_ramazan"]     = daily["day_of_week"] * daily["is_ramazan"]
daily["dow_x_holiday"]     = daily["day_of_week"] * daily["is_holiday"]
daily["dow_x_pre_holiday"] = daily["day_of_week"] * daily["is_pre_holiday"]
daily["weekend_x_month"]   = daily["is_weekend"]  * daily["month"]
daily["holiday_x_month"]   = daily["is_holiday"]  * daily["month"]

# Log hedef (eğitim için)
daily["y_log"] = np.log1p(daily["y"])

# Özellik listesi (kesin tanım)
FEATURE_COLS = [
    # Takvim
    "day_of_week","day_of_month","month","week_of_year","trend","trend_sq",
    # Gün tipi
    "is_weekend","is_monday","is_tuesday","is_wednesday","is_thursday",
    "is_friday","is_saturday","is_sunday",
    # Tatil
    "is_holiday","is_special_day","is_dini_gun","is_ramazan",
    "is_pre_holiday","is_pre2_holiday","is_post_holiday",
    "is_near_holiday","days_to_holiday","days_from_holiday",
    # Lag (orijinal + log)
    "lag_7","lag_14","lag_21","lag_28",
    "lag_7_log","lag_14_log","lag_21_log",
    # Rolling
    "roll_mean_7","roll_mean_14","roll_mean_21",
    "roll_mean_7_log","roll_mean_14_log",
    "roll_std_7","roll_std_14",
    "roll_max_7","roll_min_7","roll_median_7",
    # EWM
    "ewm_7","ewm_14","ewm_7_log","ewm_14_log",
    # DOW istatistikleri
    "dow_mean","dow_median","dow_std","dow_q25","dow_q75","dow_mean_log",
    # Ay istatistikleri
    "month_mean","month_median",
    # Etkileşimler
    "dow_x_month","dow_x_ramazan","dow_x_holiday",
    "dow_x_pre_holiday","weekend_x_month","holiday_x_month",
]

# Eğitim veri setleri
train_all = daily.dropna(subset=FEATURE_COLS+["y_log"]).copy()
train_wd  = train_all[train_all["day_of_week"] < 5].copy()   # Pzt-Cum
train_we  = train_all[train_all["day_of_week"] >= 5].copy()  # Cmt-Paz

print(f"  ✓ Toplam özellik       : {len(FEATURE_COLS)}")
print(f"  ✓ Eğitim (hafta içi)   : {len(train_wd)} gün (Pzt-Cum)")
print(f"  ✓ Eğitim (hafta sonu)  : {len(train_we)} gün (Cmt-Paz)")

# ================================================================
# 3. KÜTÜPHANELERİ YÜKLEYİP KONTROL ET
# ================================================================
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error

try:
    import xgboost as xgb;   HAS_XGB = True
except ImportError:
    HAS_XGB = False; print("  ⚠ XGBoost yok — pip install xgboost")

try:
    import lightgbm as lgb;  HAS_LGB = True
except ImportError:
    HAS_LGB = False; print("  ⚠ LightGBM yok — pip install lightgbm")

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = USE_OPTUNA
except ImportError:
    HAS_OPTUNA = False; print("  ⚠ Optuna yok — pip install optuna")

# ================================================================
# 4. CV DEĞERLENDİRME
# ================================================================
def cv_evaluate_log(model_cls, params, X, y_log, n_splits=3, test_size=5):
    """
    Zaman Serisi Çapraz Doğrulama (TimeSeriesSplit) ile model performansını ölçer.

    Önemli teknik detay: Model log uzayında (log1p) eğitilir; tahminler
    expm1 ile orijinal ölçeğe geri dönüştürülerek MAE/SMAPE hesaplanır.
    Bu yaklaşım, yüksek talep günleriyle düşük talep günlerinin model
    üzerindeki etkisini dengeler (varyans stabilizasyonu).

    Döndürür:
        (ortalama_MAE: float, ortalama_SMAPE: float)
    """
    tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size)
    maes, smapes_ = [], []
    for tr, te in tscv.split(X):
        m = model_cls(**params)
        m.fit(X.iloc[tr], y_log.iloc[tr])
        p_log = m.predict(X.iloc[te])
        p     = np.expm1(np.maximum(p_log, 0))
        yt    = np.expm1(y_log.iloc[te].values)
        maes.append(mean_absolute_error(yt, p))
        smapes_.append(smape(yt, p))
    return float(np.mean(maes)), float(np.mean(smapes_))

# ================================================================
# 5. OPTUNA TUNING
# ================================================================
print(f"\n[3/7] Optuna hiperparametre optimizasyonu ({N_TRIALS} trial)...")

def tune_model(model_type, X, y_log, n_trials, n_splits=3, test_size=5, label=""):
    """
    Optuna çerçevesiyle otomatik hiperparametre optimizasyonu yapar.

    Desteklenen model türleri: 'xgb' (XGBoost) veya 'lgb' (LightGBM).
    Bayesian optimizasyon (TPESampler) ile n_trials kadar deneme yapılır;
    her denemede TimeSeriesSplit CV üzerinden MAE minimize edilir.

    Hafta içi modeli için N_TRIALS=40, hafta sonu modeli için N_TRIALS//2
    kullanılarak hafta sonu veri kıtlığına karşı aşırı öğrenme önlenir.

    Optuna kurulu değilse (ImportError), fonksiyon None döndürür;
    çağıran kod varsayılan parametrelere (DEFAULT_WD/DEFAULT_WE) düşer.
    """
    if not HAS_OPTUNA: return None
    if model_type == "xgb" and not HAS_XGB: return None
    if model_type == "lgb" and not HAS_LGB: return None

    def objective(trial):
        if model_type == "xgb":
            params = {
                "n_estimators":      trial.suggest_int("n_estimators",100,600),
                "max_depth":         trial.suggest_int("max_depth",3,8),
                "learning_rate":     trial.suggest_float("learning_rate",0.01,0.2,log=True),
                "subsample":         trial.suggest_float("subsample",0.5,1.0),
                "colsample_bytree":  trial.suggest_float("colsample_bytree",0.5,1.0),
                "min_child_weight":  trial.suggest_int("min_child_weight",1,10),
                "reg_alpha":         trial.suggest_float("reg_alpha",1e-4,10.0,log=True),
                "reg_lambda":        trial.suggest_float("reg_lambda",1e-4,10.0,log=True),
                "random_state":42,"verbosity":0,
            }
            mae_v, _ = cv_evaluate_log(xgb.XGBRegressor, params, X, y_log, n_splits, test_size)
        else:
            params = {
                "n_estimators":      trial.suggest_int("n_estimators",100,600),
                "max_depth":         trial.suggest_int("max_depth",3,8),
                "learning_rate":     trial.suggest_float("learning_rate",0.01,0.2,log=True),
                "subsample":         trial.suggest_float("subsample",0.5,1.0),
                "colsample_bytree":  trial.suggest_float("colsample_bytree",0.5,1.0),
                "min_child_samples": trial.suggest_int("min_child_samples",5,30),
                "reg_alpha":         trial.suggest_float("reg_alpha",1e-4,10.0,log=True),
                "reg_lambda":        trial.suggest_float("reg_lambda",1e-4,10.0,log=True),
                "random_state":42,"verbose":-1,
            }
            mae_v, _ = cv_evaluate_log(lgb.LGBMRegressor, params, X, y_log, n_splits, test_size)
        return mae_v

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    name = "XGBoost" if model_type=="xgb" else "LightGBM"
    print(f"  ✓ {name} {label:<15} → en iyi MAE: {study.best_value:>12,.0f}")
    return study.best_params

X_wd  = train_wd[FEATURE_COLS];  y_wd  = train_wd["y_log"]
X_we  = train_we[FEATURE_COLS];  y_we  = train_we["y_log"]

xgb_params_wd = tune_model("xgb", X_wd, y_wd, N_TRIALS,    n_splits=4, test_size=5, label="(hafta içi)")
lgb_params_wd = tune_model("lgb", X_wd, y_wd, N_TRIALS,    n_splits=4, test_size=5, label="(hafta içi)")
xgb_params_we = tune_model("xgb", X_we, y_we, N_TRIALS//2, n_splits=2, test_size=3, label="(hafta sonu)")
lgb_params_we = tune_model("lgb", X_we, y_we, N_TRIALS//2, n_splits=2, test_size=3, label="(hafta sonu)")

if not HAS_OPTUNA:
    print("  (Optuna yok → varsayılan parametreler kullanılacak)")

# ================================================================
# 6. MODEL TANIMI VE EĞİTİMİ
# ================================================================
print("\n[4/7] Modeller eğitiliyor ve doğrulanıyor...")

DEFAULT_WD = dict(n_estimators=300, max_depth=5, learning_rate=0.04,
                  subsample=0.8, colsample_bytree=0.8, random_state=42)
DEFAULT_WE = dict(n_estimators=150, max_depth=4, learning_rate=0.05,
                  subsample=0.8, colsample_bytree=0.8, random_state=42)

def make_model_set(xgb_params, lgb_params, is_weekend=False):
    """
    Hafta içi veya hafta sonu için model seti oluşturur.

    Hafta sonu (is_weekend=True) için daha sığ ağaçlar kullanılır;
    çünkü Cmt-Paz veri hacmi daha azdır ve derin ağaçlar aşırı öğrenir.
    Optuna parametreleri varsa kullanılır; yoksa varsayılan değerlere düşülür.
    """
    n_est = 150 if is_weekend else 300
    d_depth = 4 if is_weekend else 6
    models = {
        "RandomForest": RandomForestRegressor(
            n_estimators=n_est, max_depth=d_depth, min_samples_leaf=3 if is_weekend else 2,
            max_features="sqrt", random_state=42, n_jobs=-1),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=n_est, max_depth=3 if is_weekend else 4,
            learning_rate=0.05, subsample=0.8, random_state=42),
    }
    if HAS_XGB:
        p = {**(xgb_params if xgb_params else (DEFAULT_WE if is_weekend else DEFAULT_WD)),
             "random_state":42,"verbosity":0}
        models["XGBoost"] = xgb.XGBRegressor(**p)
    if HAS_LGB:
        p = {**(lgb_params if lgb_params else (DEFAULT_WE if is_weekend else DEFAULT_WD)),
             "random_state":42,"verbose":-1}
        models["LightGBM"] = lgb.LGBMRegressor(**p)
    return models

def train_and_validate(label, train_df, xgb_params, lgb_params,
                       is_weekend, n_splits, test_size):
    """
    Tüm modelleri eğitir, CV ile doğrular ve terminal çıktısı üretir.

    Süreç:
      1. Her model için TimeSeriesSplit CV skoru hesaplanır (MAE + SMAPE).
      2. Model tüm eğitim verisiyle yeniden eğitilir (CV sadece değerlendirme içindir).
      3. Naive (DOW ortalaması) baseline de ensemblea dahil edilir ve skoru hesaplanır.

    Bu fonksiyon hem hafta içi hem hafta sonu set için ayrı ayrı çağrılır.
    Döndürür: (eğitilmiş_modeller: dict, cv_skorlar: dict)
    """
    X      = train_df[FEATURE_COLS]
    y_log  = train_df["y_log"]
    models = make_model_set(xgb_params, lgb_params, is_weekend)
    scores = {}
    trained = {}

    print(f"\n  ── {label} ({len(train_df)} gün) ──")
    print(f"  {'Model':<20} {'MAE':>12} {'SMAPE':>8}")
    print(f"  {'-'*44}")

    for name, model in models.items():
        mae_v, smape_v = cv_evaluate_log(
            model.__class__, model.get_params(), X, y_log, n_splits, test_size)
        scores[name] = {"MAE":mae_v, "SMAPE":smape_v}
        model.fit(X, y_log)  # tüm veriyle yeniden eğit
        trained[name] = model
        print(f"  {name:<20} {mae_v:>12,.0f} {smape_v:>7.1f}%")

    # Naive baseline — CV skoru hesapla (tscv ile sızıntısız)
    tscv_naive = TimeSeriesSplit(n_splits=n_splits, test_size=test_size)
    naive_maes, naive_smapes_ = [], []
    for tr_n, te_n in tscv_naive.split(train_df):
        dow_means_fold = train_df.iloc[tr_n].groupby("day_of_week")["y"].mean().to_dict()
        naive_te = train_df.iloc[te_n]["day_of_week"].map(dow_means_fold).fillna(train_df["y"].mean()).values
        yt_n     = train_df.iloc[te_n]["y"].values
        naive_maes.append(mean_absolute_error(yt_n, naive_te))
        naive_smapes_.append(smape(yt_n, naive_te))
    nb_mae   = float(np.mean(naive_maes))
    nb_smape = float(np.mean(naive_smapes_))
    scores["Naive (DOW Ort.)"] = {"MAE": nb_mae, "SMAPE": nb_smape}
    print(f"  {'Naive (DOW Ort.)':<20} {nb_mae:>12,.0f} {nb_smape:>7.1f}%  ← ensemblea dahil!")
    return trained, scores

models_wd, scores_wd = train_and_validate(
    "HAFTA İÇİ  (Pzt-Cum)", train_wd, xgb_params_wd, lgb_params_wd,
    is_weekend=False, n_splits=4, test_size=5)

models_we, scores_we = train_and_validate(
    "HAFTA SONU (Cmt-Paz)", train_we, xgb_params_we, lgb_params_we,
    is_weekend=True, n_splits=2, test_size=3)

def ensemble_weights(scores):
    """
    Her modelin CV MAE'sinin tersini ağırlık olarak kullanır (Inverse-MAE Ensemble).
    Daha düşük MAE → daha yüksek ağırlık. Ağırlıklar toplamı 1.0'a normalize edilir.
    Bu strateji, tekil modellere kıyasla daha kararlı ve güvenilir tahmin üretir.
    """
    raw   = {k: 1.0/(v["MAE"]+1e-6) for k,v in scores.items()}
    total = sum(raw.values())
    return {k: v/total for k,v in raw.items()}

w_wd = ensemble_weights(scores_wd)
w_we = ensemble_weights(scores_we)

print(f"\n  Hafta içi  ensemble ağırlıkları: " +
      " | ".join(f"{k}: {v:.3f}" for k,v in w_wd.items()))
print(f"  Hafta sonu ensemble ağırlıkları: " +
      " | ".join(f"{k}: {v:.3f}" for k,v in w_we.items()))

# ================================================================
# 7. TAHMİN (11-17 MAYIS)
# ================================================================
print("\n[5/7] 11-17 Mayıs 2026 tahmini yapılıyor...")

tahmin_tarihleri = pd.date_range(TAHMIN_BASLANGIC, TAHMIN_BITIS, freq="D")

def build_prediction_row(d: pd.Timestamp) -> dict:
    """
    Tahmin edilecek gün (d) için tüm ML özelliklerini hesaplar.

    Özellikler veri sızıntısı olmadan hesaplanır:
      - Lag değerleri: yalnızca geçmiş veriden (d - lag_gün)
      - Rolling istatistikler: yalnızca d tarihinden önceki pencere
      - EWM: son 21 günün exponential ağırlıklı ortalaması
      - Trend: eğitim verisinin doğal devamı (eğitim_uzunluğu + delta_gün)

    Döndürür: FEATURE_COLS ile birebir eşleşen özellik sözlüğü
    """
    row = build_calendar_features(d)

    # Trend (eğitim verisinin devamı)
    row["trend"]    = float(len(daily) + (d - daily["ds"].iloc[-1]).days - 1)
    row["trend_sq"] = row["trend"] ** 2

    # Lag (geçmiş veriden al)
    for lag in [7, 14, 21, 28]:
        ref   = d - timedelta(days=lag)
        match = daily[daily["ds"] == ref]["y"]
        raw   = float(match.iloc[0]) if len(match)>0 else float(daily["y"].mean())
        row[f"lag_{lag}"]     = raw
        row[f"lag_{lag}_log"] = float(np.log1p(raw))

    # Rolling istatistikler
    for w in [7, 14, 21]:
        start = d - timedelta(days=w)
        wd_   = daily[(daily["ds"]>=start) & (daily["ds"]<d)]["y"]
        val   = float(wd_.mean()) if len(wd_)>=3 else float(daily["y"].mean())
        std   = float(wd_.std())  if len(wd_)>=3 else float(daily["y"].std())
        row[f"roll_mean_{w}"]     = val
        row[f"roll_std_{w}"]      = std
        row[f"roll_mean_{w}_log"] = float(np.log1p(val))

    w7 = daily[(daily["ds"]>=(d-timedelta(7))) & (daily["ds"]<d)]["y"]
    row["roll_max_7"]    = float(w7.max())    if len(w7)>=3 else float(daily["y"].max())
    row["roll_min_7"]    = float(w7.min())    if len(w7)>=3 else float(daily["y"].min())
    row["roll_median_7"] = float(w7.median()) if len(w7)>=3 else float(daily["y"].median())

    # EWM
    recent = daily[daily["ds"]<d].tail(21)["y"]
    ewm7  = float(recent.ewm(span=7).mean().iloc[-1])  if len(recent)>=3 else float(daily["y"].mean())
    ewm14 = float(recent.ewm(span=14).mean().iloc[-1]) if len(recent)>=5 else float(daily["y"].mean())
    row["ewm_7"]     = ewm7;  row["ewm_7_log"]  = float(np.log1p(ewm7))
    row["ewm_14"]    = ewm14; row["ewm_14_log"] = float(np.log1p(ewm14))

    # DOW & Month istatistikleri
    dow = d.dayofweek; mon = d.month
    dr  = _dow[_dow["day_of_week"]==dow]
    mr  = _month[_month["month"]==mon]
    row["dow_mean"]    = float(dr["dow_mean"].iloc[0])    if len(dr)>0 else float(daily["y"].mean())
    row["dow_median"]  = float(dr["dow_median"].iloc[0])  if len(dr)>0 else float(daily["y"].median())
    row["dow_std"]     = float(dr["dow_std"].iloc[0])     if len(dr)>0 else float(daily["y"].std())
    row["dow_q25"]     = float(dr["dow_q25"].iloc[0])     if len(dr)>0 else float(daily["y"].quantile(0.25))
    row["dow_q75"]     = float(dr["dow_q75"].iloc[0])     if len(dr)>0 else float(daily["y"].quantile(0.75))
    row["dow_mean_log"]= float(np.log1p(row["dow_mean"]))
    row["month_mean"]  = float(mr["month_mean"].iloc[0])  if len(mr)>0 else float(daily["y"].mean())
    row["month_median"]= float(mr["month_median"].iloc[0])if len(mr)>0 else float(daily["y"].median())

    # Etkileşimler
    row["dow_x_month"]       = dow * mon
    row["dow_x_ramazan"]     = dow * row["is_ramazan"]
    row["dow_x_holiday"]     = dow * row["is_holiday"]
    row["dow_x_pre_holiday"] = dow * row["is_pre_holiday"]
    row["weekend_x_month"]   = row["is_weekend"] * mon
    row["holiday_x_month"]   = row["is_holiday"] * mon

    return row

pred_rows_list = [build_prediction_row(d) for d in tahmin_tarihleri]
pred_df = pd.DataFrame(pred_rows_list)
X_pred  = pred_df[FEATURE_COLS]

# DOW ortalamaları (Naive tahmin için)
dow_means_wd = train_wd.groupby("day_of_week")["y"].mean().to_dict()  # Pzt-Cum
dow_means_we = train_we.groupby("day_of_week")["y"].mean().to_dict()  # Cmt-Paz

# Her gün için ayrı modelle tahmin
final_preds     = np.zeros(len(tahmin_tarihleri))
all_model_names = list(models_wd.keys()) + ["Naive (DOW Ort.)"]
model_preds_all = {name: np.zeros(len(tahmin_tarihleri)) for name in all_model_names}

for i, d in enumerate(tahmin_tarihleri):
    is_we_day   = d.dayofweek >= 5
    models_use  = models_we  if is_we_day else models_wd
    weights_use = w_we       if is_we_day else w_wd
    dow_means   = dow_means_we if is_we_day else dow_means_wd
    X_row = X_pred.iloc[[i]]

    # ML modelleri
    for name, model in models_use.items():
        p_log = float(model.predict(X_row)[0])
        p     = float(np.expm1(max(p_log, 0)))
        model_preds_all[name][i] = p
        final_preds[i] += weights_use.get(name, 0) * p

    # Naive (DOW ortalaması)
    naive_p = float(dow_means.get(d.dayofweek, daily["y"].mean()))
    model_preds_all["Naive (DOW Ort.)"][i] = naive_p
    final_preds[i] += weights_use.get("Naive (DOW Ort.)", 0) * naive_p

    final_preds[i] = max(final_preds[i], 0)

# Güven aralığı (DOW geçmiş Q25-Q75 bazlı)
pred_lower = np.array([
    max(float(_dow[_dow["day_of_week"]==d.dayofweek]["dow_q25"].iloc[0]), 0)
    if len(_dow[_dow["day_of_week"]==d.dayofweek])>0 else final_preds[i]*0.8
    for i,d in enumerate(tahmin_tarihleri)
])
pred_upper = np.array([
    float(_dow[_dow["day_of_week"]==d.dayofweek]["dow_q75"].iloc[0])
    if len(_dow[_dow["day_of_week"]==d.dayofweek])>0 else final_preds[i]*1.2
    for i,d in enumerate(tahmin_tarihleri)
])
# Güven aralığını ensemble tahminiyle hizala
pred_lower = np.minimum(pred_lower, final_preds * 0.88)
pred_upper = np.maximum(pred_upper, final_preds * 1.12)

# ================================================================
# 8. SONUÇ TABLOSU
# ================================================================
print(f"\n[6/7] Sonuçlar...")
print()
print(f"  {'─'*74}")
print(f"  {'Tarih':<12} {'Gün':<12} {'Tahmin':>14} {'Q25 Alt':>12} {'Q75 Üst':>12} {'Model'}")
print(f"  {'─'*74}")

results = []
for i, d in enumerate(tahmin_tarihleri):
    gun    = GUN_ADI[d.dayofweek]
    t      = final_preds[i]
    lo     = pred_lower[i]
    hi     = pred_upper[i]
    tip    = "Hafta Sonu" if d.dayofweek>=5 else "Hafta İçi"
    row    = {
        "Tarih":           d.date(),
        "Gün":             gun,
        "Model Tipi":      tip,
        "Ensemble Tahmin": round(t, 0),
        "Alt (Q25)":       round(lo, 0),
        "Üst (Q75)":       round(hi, 0),
    }
    for name in model_preds_all:
        row[f"{name} Tahmini"] = round(model_preds_all[name][i], 0)
    results.append(row)
    print(f"  {str(d.date()):<12} {gun:<12} {t:>14,.0f} {lo:>12,.0f} {hi:>12,.0f}  {tip}")

print(f"  {'─'*74}")
print(f"  {'HAFTALIK TOPLAM':<24} {final_preds.sum():>14,.0f} {pred_lower.sum():>12,.0f} {pred_upper.sum():>12,.0f}")
print(f"  {'HAFTALIK ORTALAMA':<24} {final_preds.mean():>14,.0f}")
print(f"  {'─'*74}")

# ================================================================
# 9. EXCEL ÇIKTISI
# ================================================================
results_df  = pd.DataFrame(results)

all_sc_flat = {}
for n,s in scores_wd.items(): all_sc_flat[f"{n} (Hafta İçi)"]  = s
for n,s in scores_we.items(): all_sc_flat[f"{n} (Hafta Sonu)"] = s

metrics_df = pd.DataFrame([
    {"Model": k, "MAE":round(v["MAE"],0), "SMAPE(%)":round(v["SMAPE"],2)}
    for k,v in all_sc_flat.items()
])

dow_sum = (_dow.rename(columns={
    "day_of_week":"Gün Kodu","dow_mean":"Ortalama","dow_median":"Medyan",
    "dow_std":"Std","dow_q25":"Q25","dow_q75":"Q75"}).copy())
dow_sum.insert(0,"Gün",[GUN_ADI[i] for i in dow_sum["Gün Kodu"]])
dow_sum = dow_sum.drop(columns=["Gün Kodu"])

with pd.ExcelWriter(OUTPUT_EXCEL, engine="openpyxl") as writer:
    results_df.to_excel(writer, sheet_name="11-17 Mayıs Tahmin", index=False)
    daily[["ds","y","y_log"]].rename(columns={
        "ds":"Tarih","y":"Toplam Desi","y_log":"Log(Desi)"
    }).to_excel(writer, sheet_name="Geçmiş Günlük", index=False)
    metrics_df.to_excel(writer, sheet_name="Model Metrikleri", index=False)
    dow_sum.to_excel(writer, sheet_name="Günlük İstatistikler", index=False)

print(f"\n  ✓ Excel → {OUTPUT_EXCEL}")

# ================================================================
# 10. GÖRSELLEŞTİRME (5 Panel)
# ================================================================
print("\n[7/7] Grafikler oluşturuluyor...")

try:
    plt.style.use("seaborn-v0_8-whitegrid")
except:
    plt.style.use("ggplot")

C = {"blue":"#3B82F6","red":"#EF4444","green":"#10B981",
     "orange":"#F59E0B","purple":"#8B5CF6","gray":"#6B7280"}

fig = plt.figure(figsize=(22, 16))
fig.patch.set_facecolor("#F0F4F8")

gs = fig.add_gridspec(3, 3, hspace=0.45, wspace=0.32,
                      left=0.06, right=0.97, top=0.93, bottom=0.06)
ax1 = fig.add_subplot(gs[0, :2])
ax2 = fig.add_subplot(gs[0, 2])
ax3 = fig.add_subplot(gs[1, :2])
ax4 = fig.add_subplot(gs[1, 2])
ax5 = fig.add_subplot(gs[2, :])

for ax in [ax1,ax2,ax3,ax4,ax5]:
    ax.set_facecolor("#FFFFFF")

fig.suptitle(
    "Desi Talep Tahmini v2 — 11-17 Mayıs 2026\n"
    "Log Dönüşüm | Ayrı Hafta İçi/Sonu Modeli | Optuna Tuning | 50+ Özellik",
    fontsize=13, fontweight="bold", color="#1E293B")

# ── Panel 1: Tüm dönem ──────────────────────────────────────────
ax1.plot(daily["ds"], daily["y"], color=C["blue"], lw=1.2, alpha=0.75, label="Geçmiş Veri")
ax1.fill_between(tahmin_tarihleri, pred_lower, pred_upper,
                 alpha=0.3, color=C["orange"], label="Q25-Q75 Güven Aralığı")
ax1.plot(tahmin_tarihleri, final_preds, "o-", color=C["red"], lw=2.5, ms=9,
         label="11-17 Mayıs Tahmini", zorder=6)
for k,(nm,tp) in SPECIAL_DATES.items():
    ts = pd.Timestamp(k)
    if tp=="tatil" and daily["ds"].min()<=ts<=daily["ds"].max():
        ax1.axvline(ts, color="#DC2626", alpha=0.2, lw=1.2, ls="--")
ax1.set_title("Tüm Dönem: Geçmiş + 11-17 Mayıs Tahmini (kırmızı çizgiler=tatil)", fontweight="bold")
ax1.legend(fontsize=8.5, loc="upper right")
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
ax1.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=38, ha="right", fontsize=8)
ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"{x/1e6:.1f}M"))
ax1.set_ylabel("Toplam Desi"); ax1.grid(alpha=0.3)

# ── Panel 2: Model MAE karşılaştırma ────────────────────────────
names_flat  = list(all_sc_flat.keys())
maes_flat   = [all_sc_flat[n]["MAE"] for n in names_flat]
smaps_flat  = [all_sc_flat[n]["SMAPE"] for n in names_flat]
bar_clrs    = [C["blue"] if "İçi" in n else C["purple"] for n in names_flat]
short_nms   = [n.replace(" (Hafta İçi)","").replace(" (Hafta Sonu)"," (W/E)") for n in names_flat]
brs2 = ax2.barh(short_nms, maes_flat, color=bar_clrs, alpha=0.82)
for bar, mae_v, sm_v in zip(brs2, maes_flat, smaps_flat):
    ax2.text(mae_v + max(maes_flat)*0.01, bar.get_y()+bar.get_height()/2,
             f"{mae_v/1e3:.0f}K  ({sm_v:.0f}%)", va="center", fontsize=7.5)
ax2.set_title("Doğrulama: MAE (SMAPE%)", fontweight="bold")
ax2.set_xlabel("MAE (Desi)")
ax2.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"{x/1e3:.0f}K"))
ax2.legend(handles=[Patch(color=C["blue"],label="Hafta İçi"),
                    Patch(color=C["purple"],label="Hafta Sonu")], fontsize=8)
ax2.grid(alpha=0.3, axis="x")

# ── Panel 3: Son 6 hafta zoom ────────────────────────────────────
zoom_start = pd.Timestamp("2026-03-30")
zd = daily[daily["ds"]>=zoom_start]
ax3.plot(zd["ds"], zd["y"], "o-", color=C["blue"], lw=1.5, ms=4.5,
         alpha=0.85, label="Gerçek")
ax3.fill_between(tahmin_tarihleri, pred_lower, pred_upper,
                 alpha=0.25, color=C["orange"])
ax3.plot(tahmin_tarihleri, final_preds, "o-", color=C["red"], lw=2.5, ms=9,
         label="Ensemble", zorder=6)
ls_cycle = ["--","-.",":","--"]
for idx,(name,preds) in enumerate(model_preds_all.items()):
    ax3.plot(tahmin_tarihleri, preds, ls_cycle[idx%4], lw=1.1, alpha=0.5, label=name)
ax3.axvspan(TAHMIN_BASLANGIC-timedelta(hours=12),
            TAHMIN_BITIS+timedelta(hours=12), alpha=0.07, color=C["orange"])
ax3.set_title("Son 6 Hafta Zoom — Tüm Model Tahminleri", fontweight="bold")
ax3.legend(fontsize=7.5, loc="upper left", ncol=2)
ax3.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
ax3.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
plt.setp(ax3.xaxis.get_majorticklabels(), rotation=38, ha="right", fontsize=8)
ax3.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"{x/1e6:.1f}M"))
ax3.set_ylabel("Toplam Desi"); ax3.grid(alpha=0.3)

# ── Panel 4: Violin + tahmin yıldızı ────────────────────────────
viol_data = [daily[daily["day_of_week"]==i]["y"].values for i in range(7)]
vp = ax4.violinplot(viol_data, positions=range(7), showmedians=True, showextrema=True)
for i,body in enumerate(vp["bodies"]):
    body.set_facecolor(C["red"] if i>=5 else C["blue"])
    body.set_alpha(0.45)
vp["cmedians"].set_color("#1F2937"); vp["cmedians"].set_lw(2)
for d,pred in zip(tahmin_tarihleri, final_preds):
    ax4.scatter(d.dayofweek, pred, color=C["red"], s=130, zorder=7, marker="*",
                label="11-17 Mayıs" if d.dayofweek==0 else "")
ax4.set_xticks(range(7))
ax4.set_xticklabels([GUN_KISA[i] for i in range(7)], fontsize=9)
ax4.set_title("Günlük Dağılım (Violin) + Tahmin ★", fontweight="bold")
ax4.set_ylabel("Toplam Desi")
ax4.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"{x/1e6:.1f}M"))
ax4.legend(fontsize=8); ax4.grid(alpha=0.3, axis="y")

# ── Panel 5: 11-17 Mayıs bar (tam ekran genişlik) ───────────────
x5 = np.arange(len(tahmin_tarihleri))
bar_clrs5 = [C["red"] if d.dayofweek>=5 else C["blue"] for d in tahmin_tarihleri]
brs5 = ax5.bar(x5, final_preds, width=0.5, color=bar_clrs5, alpha=0.85, zorder=3)
ax5.fill_between(x5, pred_lower, pred_upper, alpha=0.2, color=C["orange"],
                 label="Q25-Q75 Güven Aralığı", step="mid")
for idx,(name,preds) in enumerate(model_preds_all.items()):
    ax5.plot(x5, preds, "o"+ls_cycle[idx%4], lw=1.3, ms=5.5, alpha=0.6, label=name)
for bar,val,lo,hi in zip(brs5, final_preds, pred_lower, pred_upper):
    ax5.text(bar.get_x()+bar.get_width()/2, hi+final_preds.max()*0.012,
             f"{val/1e6:.3f}M", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax5.annotate("", xy=(bar.get_x()+bar.get_width()/2, hi),
                 xytext=(bar.get_x()+bar.get_width()/2, lo),
                 arrowprops=dict(arrowstyle="|-|", color=C["gray"], lw=1.5))
ax5.set_xticks(x5)
ax5.set_xticklabels([f"{GUN_ADI[d.dayofweek]}\n{d.strftime('%d %b')}"
                     for d in tahmin_tarihleri], fontsize=9.5)
ax5.set_title("11-17 Mayıs 2026 — Günlük Tahmin Detayı  (Mavi=İş Günü · Kırmızı=Hafta Sonu)",
              fontweight="bold")
ax5.set_ylabel("Toplam Desi")
ax5.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"{x/1e6:.1f}M"))
ax5.legend(fontsize=8, loc="upper right", ncol=3)
ax5.grid(alpha=0.3, axis="y", zorder=0)
ax5.text(0.995, 0.97,
         f"Haftalık Toplam: {final_preds.sum()/1e6:.3f}M desi\n"
         f"Günlük Ort.:     {final_preds.mean()/1e6:.3f}M desi",
         transform=ax5.transAxes, ha="right", va="top",
         fontsize=10.5, fontweight="bold",
         bbox=dict(boxstyle="round,pad=0.5", facecolor="#FEF3C7",
                   edgecolor=C["orange"], alpha=0.95))

plt.savefig(OUTPUT_GRAFIK, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"  ✓ Grafik → {OUTPUT_GRAFIK}")
plt.show()

# ================================================================
# ÖZET
# ================================================================
print()
print(SEP)
print("  TAHMİN v2 TAMAMLANDI!")
print(SEP)
print(f"\n  📊 11-17 Mayıs 2026 — Günlük Tahminler:")
for i,d in enumerate(tahmin_tarihleri):
    tip = "🔴 Hafta Sonu" if d.dayofweek>=5 else "🔵 Hafta İçi"
    lo  = pred_lower[i]; hi = pred_upper[i]
    print(f"     {GUN_ADI[d.dayofweek]:<12} {str(d.date())} │ "
          f"{final_preds[i]:>12,.0f}  [{lo:>12,.0f} – {hi:>12,.0f}]  {tip}")
print(f"     {'─'*70}")
print(f"     {'Haftalık Toplam':<28} │ {final_preds.sum():>12,.0f}  "
      f"[{pred_lower.sum():>12,.0f} – {pred_upper.sum():>12,.0f}]")
print(f"     {'Haftalık Ortalama':<28} │ {final_preds.mean():>12,.0f}")
print(f"\n  📁 Çıktı dosyaları:")
print(f"     - {OUTPUT_EXCEL}")
print(f"     - {OUTPUT_GRAFIK}")
best_wd = min(scores_wd, key=lambda k: scores_wd[k]["MAE"])
best_we = min(scores_we, key=lambda k: scores_we[k]["MAE"])
print(f"\n  🏆 En iyi model (doğrulama MAE):")
print(f"     Hafta içi  → {best_wd:<20} MAE: {scores_wd[best_wd]['MAE']:>12,.0f}  "
      f"SMAPE: {scores_wd[best_wd]['SMAPE']:.1f}%")
print(f"     Hafta sonu → {best_we:<20} MAE: {scores_we[best_we]['MAE']:>12,.0f}  "
      f"SMAPE: {scores_we[best_we]['SMAPE']:.1f}%")
print()
