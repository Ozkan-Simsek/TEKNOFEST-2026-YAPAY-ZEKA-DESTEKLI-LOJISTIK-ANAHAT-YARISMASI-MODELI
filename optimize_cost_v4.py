#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
================================================================
 MODÜL: optimize_cost_v4.py
 AMAÇ : TEKNOFEST 2026 Lojistik Projesi — Kısıtlı Araç Maliyet Optimizasyonu

 Bu modül, finalize.py çıktısı olan tahminleri alarak her gün
 için en düşük maliyetli, TEKNOFEST kurallarına tam uyumlu araç planı üretir.

 PIPELINE'DAKİ YERİ:
   tahmin.py → tahmin_sehir.py → finalize.py → [optimize_cost_v4.py] → gorsellestir.py

 ALGORİTMA:
   1. Kiralık araçlar (zorunlu sabit katman) her gün önce atanır.
   2. Kalan talep Kısıtlı VRP (Global Optimum Set-Partitioning) ile spot araca verilir.
   3. Her spot ataması için %10 doluluk hard constraint kontrol edilir.
   4. Kısıtı geçemeyen kargolar Zorunlu Kamyonet ile taşınır (kargo asla silinemez).

 TEKNOFEST ÇIKTI FORMATI:
   Tarih | Çıkış TM | Varış TM | Tahmin Edilen Desi | Araç Tipi
================================================================
"""

import os
import pandas as pd
import math
import itertools
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")

# Windows cp1252 terminalinde Türkçe karakterlerin UnicodeEncodeError vermesini önle.
import sys, io
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


print("=" * 65)
print("  ARAÇ MALİYET OPTİMİZASYONU — TEKNOFEST 2026 YARIŞMA MODU (V6)")
print("  Kısıtlı VRP + Global Optimum Uğrama Algoritması")
print("=" * 65)


# ─────────────────────────────────────────────────────────────────────
# Yardımcı Fonksiyonlar
# ─────────────────────────────────────────────────────────────────────
def haversine_pure(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Haversine formülüyle iki coğrafi nokta arasındaki kuş uçuşu mesafesini hesaplar (km).
    Optimizeçi, araç rota maliyetini (KM × BirimFiyat) hesaplamak için bu fonksiyonu kullanır.
    Koordinatlar data/Koordinatlar.xlsx dosyasından okunur.
    """
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return R * 2 * math.asin(math.sqrt(a))


def fix_date(d) -> str:
    """
    Excel'den okunan tarih değerini standart YYYY-MM-DD string formatına döndürür.

    TEKNOFEST jüri graderı, tarih kolonunu tam string eşleme (exact match) ile kontrol eder.
    Excel'de serial number (int/float) veya datetime olarak gelebilecek değerleri
    güvenli şekilde normalize eder. Hatalı tarih durumunda str(d) fallbackı devreye girer.
    """
    try:
        if isinstance(d, (int, float)):
            return (pd.to_datetime("1899-12-30") + pd.to_timedelta(d, "D")).strftime("%Y-%m-%d")
        return pd.to_datetime(d).strftime("%Y-%m-%d")
    except Exception:
        return str(d)


# ─────────────────────────────────────────────────────────────────────
# Set-Partitioning — Global Optimum VRP İçin
# ─────────────────────────────────────────────────────────────────────
def partition(collection: list):
    """
    Verilen listeyi tüm olası küme bölümlemelerine (set partitioning) ayırır.

    Global Optimum Kısıtlı VRP için temel kombinatorik motor.
    Her alt-küme bağımsız bir araç seferi (varış grubu) olarak değerlendirilir.

    Örnek: [A, B, C] girdisi için üretilen bölümlemeler:
      [{A,B,C}], [{A,B},{C}], [{A,C},{B}], [{B,C},{A}], [{A},{B},{C}]

    Her bölümleme için toplam maliyet hesaplanır; en düşük maliyetli
    %10 hard constraintı geçen bölümleme seçilir (Global Optimum).

    KURAL 2: Farklı çıkış TM'lerinden gelen yükler asla aynı listeye girmez.
    """
    if len(collection) == 1:
        yield [collection]
        return
    first = collection[0]
    for smaller in partition(collection[1:]):
        for n, subset in enumerate(smaller):
            yield smaller[:n] + [[first] + subset] + smaller[n + 1:]
        yield [[first]] + smaller


# ─────────────────────────────────────────────────────────────────────
# 1. Veri Yükleme
# ─────────────────────────────────────────────────────────────────────
print("\n[1/4] Veriler yükleniyor...")

f_opt_input = "data/Optimizasyon_Kiralik_11_17_mayis.xlsx"
f_kapasite  = [f for f in os.listdir("data") if "Kapasite" in f and f.endswith(".xlsx")][0]
f_kiralik   = [f for f in os.listdir("data") if "Kiral"    in f and f.endswith(".xlsx")][0]
f_koor      = [f for f in os.listdir("data") if "Koor"     in f and f.endswith(".xlsx")][0]

df_opt  = pd.read_excel(f_opt_input, sheet_name="Günlük Optimizasyon Verisi")
df_kap  = pd.read_excel(os.path.join("data", f_kapasite))
df_kir  = pd.read_excel(os.path.join("data", f_kiralik))
df_koor = pd.read_excel(os.path.join("data", f_koor))

# GÜNCELLEME 4: Hub-and-Spoke iptal— spot-only sayfa artık anlamsız, pass.
try:
    df_spot_input = pd.read_excel(f_opt_input, sheet_name="Spot Sadece Rotalar")
    df_spot_input["Tarih"] = df_spot_input["Tarih"].apply(fix_date)
    has_spot_only_sheet = True
    print(f"  ✓ Spot Sadece rotalar yüklendi: {len(df_spot_input)} satır")
except Exception:
    df_spot_input = pd.DataFrame()
    has_spot_only_sheet = False
    print("  ℹ  'Spot Sadece Rotalar' sayfası bulunamadı; tüm rotalar doğrudan (P2P) işlenecek.")

df_opt["Tarih"] = df_opt["Tarih"].apply(fix_date)

# Optimizasyon hedefi: Tahmin Edilen Desi (net ortalama)
toplam_net_talep = df_opt["Tahmin Edilen Desi"].sum()
# GÜNCELLEME 1/2: Q75 kolon artık mevcut olmayabilir — opsiyonel ref.
if "Üst Sınır (Q75)" in df_opt.columns:
    toplam_q75_ref = df_opt["Üst Sınır (Q75)"].sum()
    print(f"  Q75 Güvenlik Marjı (yalnızca referans)  : {toplam_q75_ref:>12,.0f} Desi")
print(f"\n  Net Talep Toplamı (optimizasyon hedefi) : {toplam_net_talep:>12,.0f} Desi")

# Koordinatlar sözlüğü
coords: dict = {}
for _, row in df_koor.iterrows():
    coords[str(row[df_koor.columns[0]]).strip()] = (
        float(row[df_koor.columns[1]]),
        float(row[df_koor.columns[2]])
    )


def get_dist(c1: str, c2: str) -> float:
    if c1 not in coords or c2 not in coords:
        raise ValueError(f"Koordinat eksik: '{c1}' veya '{c2}'")
    return haversine_pure(coords[c1][0], coords[c1][1], coords[c2][0], coords[c2][1])


# Araç kapasite ve maliyet tablosu
arac_info: dict = {}
for _, row in df_kap.iterrows():
    ad = str(row["Araç Adı"]).strip()
    arac_info[ad] = {
        "cap":      float(row["Kapasite (desi)"]),
        # KURAL 1: kir_sab artık batık maliyet değil — her seferde tam maliyete dahil edilir.
        "kir_sab":  float(row["Kiralık Araç Günlük Kira (TL)"]),
        "kir_km":   float(row["Kiralık Araç Kilometre Başına Maliyet (TL)"]),
        "spot_sab": float(row["Spot Araç Sabit Günlük Maliyet (TL)"]),
        "spot_km":  float(row["Spot Kilometre Başına Maliyet (TL)"])
    }

tir_cap = arac_info["Tır"]["cap"]

# Kiralık araçlar: rota → araç bilgileri
# KURAL 3: Bu sözlük her zaman planın sabit ilk katmanını oluşturur.
kiralik_durum: dict = {}
for _, row in df_kir.iterrows():
    c, v = row["Çıkış Transfer Merkezi"], row["Varış Transfer Merkezi"]
    kiralik_durum.setdefault((c, v), []).append({
        "tur":  str(row["Araç Türü"]).strip(),
        "sayi": int(row["Araç sayısı"])
    })

# Kapasite ve birim fiyat bilgilerini ekrana bas
print("\n  Araç Kapasite ve Maliyet Bilgileri:")
print(f"  {'Araç':<15} {'Kapasite (desi)':>16} {'Kira (TL/gün)':>14} {'KM Maliyeti':>12}")
print(f"  {'-'*62}")
for ad, bilgi in arac_info.items():
    print(f"  {ad:<15} {bilgi['cap']:>16,.0f} {bilgi['kir_sab']:>14,.0f} {bilgi['kir_km']:>12.2f}")


# ─────────────────────────────────────────────────────────────────────
# Kısıtlı Spot Araç Çözücüsü — %10 Hard Constraint İle
#
# KURAL 4a: Spot araçlar için en az %10 doluluk oranı zorunludur.
# Bu kısıtı sağlayan en ucuz kombinasyon seçilir.
# Kısıtı geçen hiçbir çözüm yoksa → None döner (fallback devreye girer).
# ─────────────────────────────────────────────────────────────────────
MIN_DOLULUK_ORANI = 0.10   # %10 Hard Constraint (TEKNOFEST Kuralı)


def solve_spot_vehicles_constrained(demand: float, distance: float):
    """
    Kısıtlı Spot VRP Çözücüsü — KURAL 4a: %10 Doluluk Hard Constraint ile.

    Verilen demand ve distance için şu koşulları sağlayan
    en ucuz spot araç kombinasyonunu tam tarama (brute-force) ile bulur:
      - Koşul 1 (KURAL 3): Toplam kapasite >= demand (tüm kargo taşınmalı)
      - Koşul 2 (KURAL 4a): Doluluk oranı = demand / toplam_cap >= %10

    Tır, Kamyon, Hafif Kamyon ve Kamyonet kombinasyonları denenir.
    Her kombinasyon için maliyet = sabit_kır + mesafe * km_fiyatı.

    Döndürür:
        (combo: dict | None, maliyet: float, kapasite: float)
        combo=None: %10 eşiğini geçen hiçbir çözüm yok → fallback gerekli
    """
    if demand <= 0:
        return None, 0.0, 0.0

    best_cost  = float("inf")
    best_combo = None
    best_cap   = 0.0

    # Araç başına üst sınır
    max_t  = int(demand // arac_info["Tır"]["cap"]) + 1
    max_k  = int(demand // arac_info["Kamyon"]["cap"]) + 1
    max_hk = int(demand // arac_info["Hafif Kamyon"]["cap"]) + 1
    max_km = int(demand // arac_info["Kamyonet"]["cap"]) + 1

    for t in range(max_t + 1):
        for k in range(max_k + 1):
            for hk in range(max_hk + 1):
                for km in range(max_km + 1):
                    if t == 0 and k == 0 and hk == 0 and km == 0:
                        continue

                    toplam_cap = (
                        t  * arac_info["Tır"]["cap"]
                        + k  * arac_info["Kamyon"]["cap"]
                        + hk * arac_info["Hafif Kamyon"]["cap"]
                        + km * arac_info["Kamyonet"]["cap"]
                    )

                    # Kısıt 1: Yeterli kapasite
                    if toplam_cap < demand:
                        continue

                    # KURAL 4a — %10 Hard Constraint (Spot Araçlar):
                    # demand / toplam_cap < 0.10 ise bu kombinasyon REDDEDİLİR.
                    doluluk = demand / toplam_cap
                    if doluluk < MIN_DOLULUK_ORANI:
                        continue

                    # Toplam spot maliyet = Sabit + KM × BirimFiyat
                    maliyet = (
                        t  * (arac_info["Tır"]["spot_sab"]          + distance * arac_info["Tır"]["spot_km"])
                        + k  * (arac_info["Kamyon"]["spot_sab"]       + distance * arac_info["Kamyon"]["spot_km"])
                        + hk * (arac_info["Hafif Kamyon"]["spot_sab"] + distance * arac_info["Hafif Kamyon"]["spot_km"])
                        + km * (arac_info["Kamyonet"]["spot_sab"]     + distance * arac_info["Kamyonet"]["spot_km"])
                    )

                    if maliyet < best_cost:
                        best_cost  = maliyet
                        best_combo = {"Tır": t, "Kamyon": k, "Hafif Kamyon": hk, "Kamyonet": km}
                        best_cap   = toplam_cap

    return best_combo, best_cost, best_cap


def zorunlu_kamyonet_fallback(demand: float, distance: float):
    """
    KURAL 4a Fallback: %10 doluluk eşiğini geçemeyen kargolar için zorunlu taşıma.

    TEKNOFEST kuralı gereği kargo asla silinemez. Global VRP'ın hiçbir bölümlemesi
    %10 eşiğini geçemediğinde bu fonksiyon devreye girer ve en küçük araç
    (Kamyonet) ile zorunlu taşıma yapılır. Raporlamada "Zorunlu Taşıma" olarak
    işaretlenir; jüri değerlendirmesinde şeffaflık için ayrı sayfada listelenir.
    """
    km_cost = (arac_info["Kamyonet"]["spot_sab"] + distance * arac_info["Kamyonet"]["spot_km"])
    combo   = {"Tır": 0, "Kamyon": 0, "Hafif Kamyon": 0, "Kamyonet": 1}
    return combo, km_cost, arac_info["Kamyonet"]["cap"]


# ─────────────────────────────────────────────────────────────────────
# 2. Günlük Plan — Kiralık (Zorunlu) + Kısıtlı VRP (Spot) Döngüsü
# ─────────────────────────────────────────────────────────────────────
print("\n[2/4] Kiralık araçlar sabitleniyor; Kısıtlı VRP çözülüyor...")

gunluk_plan = []

# Toplam maliyet akümülatörleri
toplam_kiralik_maliyet = 0.0   # KURAL 1: Kira + KM (birleşik tam maliyet)
toplam_spot_maliyet    = 0.0
toplam_zorunlu_taşıma  = 0    # %10 kısıtını geçemeyen fallback sayısı

gunler = sorted(df_opt["Tarih"].unique())

for tarih in gunler:
    df_gun = df_opt[df_opt["Tarih"] == tarih]

    # GÜNCELLEME 3: Kiralık hat+gün duplikasyonunu önlemek için takip seti.
    # Her gün sıfırlanır; aynı (tarih, cikis, varis) çifti yalnızca bir kez yazılır.
    yazilan_kiralik_hatlar: set = set()

    # ── KATMAN 1: KİRALIK ARAÇLAR (ZORUNLU SABİT UNSUR) ─────────────
    # KURAL 3: Tanımlı kiralık araç varsa, doluluk oranından bağımsız
    # olarak plan bu araçlarla başlar. Açıkta kalan talep aşağıya taşınır.
    # KURAL 2: Her rota kendi çıkış TM'sine göre tamamen izoledir.
    #          Farklı rotaların artıkları asla birleştirilmez.
    # ─────────────────────────────────────────────────────────────────

    # Rota bazında kalan talepleri izole sözlüklerde tut
    # {(cikis, varis): kalan_desi} — konsolidasyon yasağı gereği ayrı tutulur
    kalan_rotalar: dict = {}

    for _, row in df_gun.iterrows():
        # GÜNCELLEME 4: Kolon adı uyumu — 'Kiralık Çıkış TM' veya 'Çıkış TM'
        cikis = row.get("Kiralık Çıkış TM", row.get("Çıkış TM", ""))
        varis = row.get("Kiralık Varış TM",  row.get("Varış TM", ""))
        talep = float(row["Tahmin Edilen Desi"])
        dist  = get_dist(cikis, varis)

        # ── GÜNCELLEME 3: Duplikasyon Bug Çözümü ─────────────────────────────────
        # Aynı gün+hat için kiralık araç tek bir kez yazılır.
        # Bu birleştime anahtarı ile oluşturulmuş kiralık satırları takip et.
        kiralik_key = (tarih, cikis, varis)

        kiralik_cap = 0.0
        kiralik_str = []
        kiralik_tam_maliyet = 0.0

        if (cikis, varis) in kiralik_durum and kiralik_key not in yazilan_kiralik_hatlar:
            # İlk kez yazılıyor — kiralık araçları hesapla ve kaydet
            for ar in kiralik_durum[(cikis, varis)]:
                tur, sayi = ar["tur"], ar["sayi"]
                kiralik_cap += sayi * arac_info[tur]["cap"]
                kiralik_str.append(f"{sayi} {tur}")
                kiralik_tam_maliyet += sayi * (
                    arac_info[tur]["kir_sab"] + dist * arac_info[tur]["kir_km"]
                )

            toplam_kiralik_maliyet += kiralik_tam_maliyet
            yazilan_kiralik_hatlar.add(kiralik_key)

            kiralik_str_final = ", ".join(kiralik_str)
            hizmet_edilen     = min(talep, kiralik_cap)
            doluluk           = round(hizmet_edilen / kiralik_cap * 100, 1) if kiralik_cap > 0 else 0.0

            gunluk_plan.append({
                "Tarih":               tarih,
                "Çıkış TM":           cikis,
                "Varış TM":           varis,
                "Güzergah":           f"{cikis} -> {varis}",
                "Mesafe (km)":        round(dist, 1),
                "Tahmin Edilen Desi":  round(hizmet_edilen, 1),
                "Araç Tipi":          f"Kiralık: {kiralik_str_final}",
                "Sefer Türü":        "Ana Hat (Kiralık — Zorunlu)",
                "Maliyet (TL)":       round(kiralik_tam_maliyet, 0),
                "Doluluk Oranı (%)": doluluk,
            })

            # Kalan talep: kiralık kapasiteyi aşan kısım
            kalan = max(0.0, talep - kiralik_cap)
        elif (cikis, varis) in kiralik_durum and kiralik_key in yazilan_kiralik_hatlar:
            # GÜNCELLEME 3: Bu hat+gün için kiralık zaten yazıldı.
            # Kapasite tamamsa talebın tamamı spot'a aktarılır.
            for ar in kiralik_durum[(cikis, varis)]:
                tur, sayi = ar["tur"], ar["sayi"]
                kiralik_cap += sayi * arac_info[tur]["cap"]
            kalan = max(0.0, talep - kiralik_cap)
        else:
            # Kiralık tanımsız — tüm talep spot'a gider
            kalan = talep

        # Kalan varsa spot çözüme aktar
        if kalan > 0:
            kalan_rotalar.setdefault(cikis, []).append({
                "Varis": varis,
                "Talep": kalan
            })

    # ── POİNT TO POİNT ───────────────────────────
    #89 rota bağımsız P2P (Point-to-Point) olarak işleniyor.Her rota kendi kalan_rotalar'ına
    #düşüyor.Dolulukları artırmak için yalnızca Multi-stop VRP (uğrama) kullanılır.

    # ── KATMAN 2: SPOT ARAÇ OPTİMİZASYONU (KISITLI VRP) ──────────────
    # KURAL 3: Spot araçlar yalnızca kiralık kapasitesi aşıldığında devreye girer.
    # KURAL 4a: Her spot atamasında %10 doluluk hard constraint uygulanır.
    # KURAL 2: Farklı çıkış TM'leri asla tek seferde birleştirilmez.
    # ─────────────────────────────────────────────────────────────────

    for cikis, varislar in kalan_rotalar.items():

        # ── FTL Tır: Tam yük → her zaman %10'u geçer, matematiksel optimum ──
        for v in varislar:
            tam_tir = int(v["Talep"] // tir_cap)
            if tam_tir > 0:
                dist     = get_dist(cikis, v["Varis"])
                spot_mal = tam_tir * (
                    arac_info["Tır"]["spot_sab"] + dist * arac_info["Tır"]["spot_km"]
                )
                toplam_spot_maliyet += spot_mal
                gunluk_plan.append({
                    "Tarih":              tarih,
                    "Çıkış TM":          cikis,
                    "Varış TM":          v["Varis"],
                    "Güzergah":          f"{cikis} -> {v['Varis']}",
                    "Mesafe (km)":       round(dist, 1),
                    "Tahmin Edilen Desi": tam_tir * tir_cap,
                    "Araç Tipi":         f"Spot: {tam_tir} Tır",
                    "Sefer Türü":        "Spot (FTL Tam Yük)",
                    "Maliyet (TL)":      round(spot_mal, 0),
                    "Doluluk Oranı (%)": 100.0,
                })
                v["Talep"] -= tam_tir * tir_cap

        # ── Kısıtlı VRP — Global Optimum Uğrama (LTL Multi-Stop) ────────
        # Tüm set bölümleme kombinasyonları denenir (Global Optimum).
        # Her bölümleme için: en kısa permütasyon rotası + %10 kısıtlı araç seçimi.
        # KURAL 2: Uğrama rotası kendi çıkışından (cikis) başlar; başka TM yükü almaz.
        ltl_list = [v for v in varislar if v["Talep"] > 0]
        if not ltl_list:
            continue

        best_overall_cost   = float("inf")
        best_partition_info = None

        for part in partition(ltl_list):
            part_cost = 0.0
            part_info = []
            valid     = True

            for subset in part:
                subset_load = sum(x["Talep"] for x in subset)
                nodes       = [x["Varis"] for x in subset]

                # En kısa uğrama rotası (permütasyon — Nearest Neighbor değil, Global)
                best_dist = float("inf")
                best_perm = None
                for p in itertools.permutations(nodes):
                    d = get_dist(cikis, p[0])
                    for i in range(len(p) - 1):
                        d += get_dist(p[i], p[i + 1])
                    if d < best_dist:
                        best_dist = d
                        best_perm = p

                # %10 Hard Constraint ile kısıtlı spot çözücü
                combo, cost, cap = solve_spot_vehicles_constrained(subset_load, best_dist)

                if combo is None:
                    # %10 kısıtını geçen kombinasyon yok → bu bölümleme geçersiz
                    valid = False
                    break

                part_cost += cost
                part_info.append({
                    "rota":  " -> ".join([cikis] + list(best_perm)),
                    "cikis_tm": cikis,
                    "varis_tm": best_perm[-1] if len(best_perm) == 1 else " & ".join(best_perm),
                    "dist":  best_dist,
                    "load":  subset_load,
                    "combo": combo,
                    "cap":   cap,
                    "cost":  cost
                })

            if valid and part_cost < best_overall_cost:
                best_overall_cost   = part_cost
                best_partition_info = part_info

        # En iyi global rotayı kaydet
        if best_partition_info is not None:
            for info in best_partition_info:
                combo_str = ", ".join(f"{v} {k}" for k, v in info["combo"].items() if v > 0)
                doluluk   = round(info["load"] / info["cap"] * 100, 1)
                toplam_spot_maliyet += info["cost"]
                gunluk_plan.append({
                    "Tarih":              tarih,
                    "Çıkış TM":          info["cikis_tm"],
                    "Varış TM":          info["varis_tm"],
                    "Güzergah":          info["rota"],
                    "Mesafe (km)":       round(info["dist"], 1),
                    "Tahmin Edilen Desi": round(info["load"], 1),
                    "Araç Tipi":         f"Spot: {combo_str}",
                    "Sefer Türü":        "Spot (Global VRP Uğrama)",
                    "Maliyet (TL)":      round(info["cost"], 0),
                    "Doluluk Oranı (%)": doluluk,
                })
        else:
            # ── %10 Kısıtı Fallback: Zorunlu Kamyonet ─────────────────────
            # Global VRP'nin hiçbir bölümlemesi %10 eşiğini geçemedi.
            # KURAL 4a: Kargo asla silinemez → Zorunlu Kamyonet ile taşı.
            for v in ltl_list:
                dist = get_dist(cikis, v["Varis"])
                combo, cost, cap = zorunlu_kamyonet_fallback(v["Talep"], dist)
                doluluk = round(v["Talep"] / cap * 100, 1)
                toplam_spot_maliyet += cost
                toplam_zorunlu_taşıma += 1
                gunluk_plan.append({
                    "Tarih":              tarih,
                    "Çıkış TM":          cikis,
                    "Varış TM":          v["Varis"],
                    "Güzergah":          f"{cikis} -> {v['Varis']}",
                    "Mesafe (km)":       round(dist, 1),
                    "Tahmin Edilen Desi": round(v["Talep"], 1),
                    "Araç Tipi":         "Spot: 1 Kamyonet (Zorunlu)",
                    "Sefer Türü":        "Spot (Zorunlu Taşıma — %10 Kısıtı Aşılamadı)",
                    "Maliyet (TL)":      round(cost, 0),
                    "Doluluk Oranı (%)": doluluk,
                })


# ─────────────────────────────────────────────────────────────────────
# 3. Raporlar ve Bütçe Özeti
# ─────────────────────────────────────────────────────────────────────
print("\n[3/4] TEKNOFEST raporları hazırlanıyor...")

df_plan = pd.DataFrame(gunluk_plan)

hizmet_edilen_toplam = df_plan["Tahmin Edilen Desi"].sum()
genel_toplam         = toplam_kiralik_maliyet + toplam_spot_maliyet

# TEKNOFEST Bütçe Özeti:
# KURAL 1: Kiralık maliyet = Kira + KM (ayrıştırılmaz tam maliyet)
# Spot maliyet ayrıca gösterilir
toplam_butce = pd.DataFrame({
    "Maliyet Kalemi": [
        "Kiralık Araç Toplam (Günlük Kira + KM Maliyeti)",
        "Spot Araç Toplam (FTL + Global VRP Uğrama)",
        "GENEL TOPLAM"
    ],
    "Toplam Tutar (TL)": [
        round(toplam_kiralik_maliyet, 0),
        round(toplam_spot_maliyet, 0),
        round(genel_toplam, 0),
    ],
    "Açıklama": [
        f"Tam Maliyet: Kira (5.000 TL Kamyon / 7.000 TL Tır) + KM × BirimFiyat",
        f"Kısıtlı VRP Global Optimum (%%10 Hard Constraint uygulandı)",
        "Toplam Operasyonel Bütçe — TEKNOFEST V6"
    ]
})


# ─────────────────────────────────────────────────────────────────────
# 4. Excel Çıktısı — TEKNOFEST Formatı
# ─────────────────────────────────────────────────────────────────────
print("\n[4/4] TEKNOFEST V6 Excel dosyası kaydediliyor...")

# KURAL 1: Tarih → strict YYYY-MM-DD string (timestamp '00:00:00' temizlenir)
df_plan["Tarih"] = pd.to_datetime(df_plan["Tarih"]).dt.strftime("%Y-%m-%d")

# KURAL 2: Çıkış TM / Varış TM güvenli suffix ( mükerrer " TM TM" önleme)
for _c in ["Çıkış TM", "Varış TM"]:
    if _c in df_plan.columns:
        df_plan[_c] = (df_plan[_c]
                       .astype(str).str.strip()
                       .str.replace(r'\s*TM$', '', regex=True).str.strip()
                       + " TM")

# KURAL 4b — Standart TEKNOFEST kolon sırası (jüri grader tam string eşleme yapar):
# Tarih | Çıkış TM | Varış TM | Tahmin Edilen Desi | Araç Tipi
teknofest_kolonlar = [
    "Tarih",
    "Çıkış TM",
    "Varış TM",
    "Tahmin Edilen Desi",
    "Araç Tipi",
    "Güzergah",
    "Mesafe (km)",
    "Sefer Türü",
    "Maliyet (TL)",
    "Doluluk Oranı (%)",
]

# Yalnızca mevcut kolonları seç (güvenli filtreleme)
mevcut_kolonlar = [k for k in teknofest_kolonlar if k in df_plan.columns]
df_teknofest    = df_plan[mevcut_kolonlar].copy()


_jury_nihai = df_plan[["Tarih","Araç Tipi","Çıkış TM","Varış TM",
                        "Tahmin Edilen Desi","Maliyet (TL)"]].copy()

# Timestamp ekle → dosya Excel'de açık olsa bile PermissionError almaz.
_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
cikis_yolu = f"data/Lojistik_Maliyet_Optimizasyonu_V6_TEKNOFEST_{_ts}.xlsx"
with pd.ExcelWriter(cikis_yolu, engine="openpyxl") as writer:
    df_teknofest.to_excel(writer, sheet_name="Sefer Planı (TEKNOFEST)", index=False)
    # KURAL 3b: Jüri nihai sayfası — resmi şablondaki yazım hatalarıyla birebir
    _jury_nihai.to_excel(writer, sheet_name="Jüri Nihai (Resmi Format)", index=False)
    toplam_butce.to_excel(writer, sheet_name="Bütçe Özeti", index=False)

    # Doğrulama sayfası: Sert kısıt ihlali kontrolü
    spot_satırlar = df_teknofest[df_teknofest["Araç Tipi"].str.startswith("Spot:")].copy()
    if not spot_satırlar.empty and "Doluluk Oranı (%)" in spot_satırlar.columns:
        zorunlu_taşıma_df = spot_satırlar[
            spot_satırlar["Araç Tipi"].str.contains("Zorunlu")
        ].copy()
        if not zorunlu_taşıma_df.empty:
            zorunlu_taşıma_df.to_excel(
                writer, sheet_name="Zorunlu Taşıma (Fallback)", index=False
            )

# ─────────────────────────────────────────────────────────────────────
# 5. Terminal Özeti — TEKNOFEST V6
# ─────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  OPTİMİZASYON TAMAMLANDI — TEKNOFEST 2026 YARIŞMA MODU V6")
print("=" * 65)

print(f"\n  📦 TALEPLERİ KARŞILAMA:")
print(f"  Net Talep Toplamı (Tahmin Edilen Desi) : {toplam_net_talep:>12,.1f}")
print(f"  Hizmet Edilen Toplam                   : {hizmet_edilen_toplam:>12,.1f}")
acikta = toplam_net_talep - hizmet_edilen_toplam
if abs(acikta) < 1.0:
    print("  => MÜKEMMEL! Açıkta kargo yok.")
elif acikta > 0:
    print(f"  => DİKKAT! {acikta:,.1f} Desi taşınamadı.")
else:
    print(f"  => Fazla kapasite: {-acikta:,.1f} Desi.")

print(f"\n  💰 BÜTÇE ÖZETİ (TEKNOFEST V6 — Tam Maliyet):")
print(f"  {'Kalem':<50} {'Tutar (TL)':>12}")
print(f"  {'-'*64}")
print(f"  {'Kiralık Toplam (Kira + KM)':<50} {toplam_kiralik_maliyet:>12,.0f}")
print(f"  {'Spot Araç Toplam (FTL + Global VRP)':<50} {toplam_spot_maliyet:>12,.0f}")
print(f"  {'-'*64}")
print(f"  {'GENEL TOPLAM':<50} {genel_toplam:>12,.0f}")

print(f"\n  📋 TEKNOFEST V6 KURAL UYUM RAPORU:")
print(f"  [K1] Tam Maliyet      : Kira + KM birleşik hesaplandı (sunk cost YOK)")
print(f"  [K2] Konsolidasyon    : Yasak — her çıkış TM izole, hub mantığı YOK")
print(f"  [K3] Zorunlu Kiralık  : Her hatta tanımlı araç planın sabit ilk katmanı")
print(f"  [K4] %%10 Hard Const.  : Spot araçlarda zorunlu — {toplam_zorunlu_taşıma} fallback atama yapıldı")
print(f"\n  [FORMAT] Zorunlu TEKNOFEST Kolonları:")
print(f"  Tarih | Çıkış TM | Varış TM | Tahmin Edilen Desi | Araç Tipi")

print(f"\n  ✓ Çıktı Dosyası: {cikis_yolu}\n")
