#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import subprocess
import os

"""
================================================================
 MODÜL: finalize.py
 AMAÇ : TEKNOFEST 2026 Lojistik Projesi — Güzel Tahmini Global Kalibrasyonu

 Bu modül, tahmin_sehir.py tarafından üretilen güzel bazı (TM) tahminlerini,
 tahmin.py'deki global ensemble toplamı (6.508.466 desi) ile orantılı olarak
 kalibre eder. Kalibrasyon olmadan güzel modeli sistematik bir sapma
 (bias) gösterebilir.

 PIPELINE'DAKİ YERİ:
   tahmin.py → tahmin_sehir.py → [finalize.py] → optimize_cost_v4.py → gorsellestir.py

 TEKNOFEST KURAL UYUMU:
   - KURAL 1: Tarih kolonu strict YYYY-MM-DD string formatında yazılır.
   - KURAL 2: Çıkış TM / Varış TM kolonlarına güvenli " TM" suffix eklenir;
              mükerrer " TM TM" yazımı regex ile önlenir.
   - KURAL 3: Çıktı kolon isimleri jüri graderının beklediği sabit sırayla yazılır.
================================================================
"""

import pandas as pd
import numpy as np
from datetime import datetime
import sys

# Windows cp1252 terminalinde Türkçe UnicodeEncodeError önleme
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

INPUT_V2     = r"data/tahmin_tm_v2_11_17_mayis.xlsx"
# Timestamp → dosya Excel'de açık olsa bile PermissionError almaz.
_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_FINAL = rf"data/tahmin_FINAL_11_17_mayis_{_ts}.xlsx"

# Global hedef: tahmin.py (ensemble) sonucu — bu değer sabit, değiştirilmez.
GLOBAL_TOPLAM = 6_508_466

GUN_ADI = {0: "Pazartesi", 1: "Salı", 2: "Çarşamba", 3: "Perşembe",
           4: "Cuma", 5: "Cumartesi", 6: "Pazar"}

print("=" * 60)
print("  FINAL KALİBRASYON — Araç Optimizasyonu Teslim Dosyası")
print("=" * 60)

# ── Veri yükle ────────────────────────────────────────────
df = pd.read_excel(INPUT_V2, sheet_name="TM Tahminler (Ana)")
df["Tarih"] = pd.to_datetime(df["Tarih"])

# ── KURAL 1: Tarih → strict YYYY-MM-DD string (timestamp '00:00:00' temizlenir) ──
df["Tarih"] = df["Tarih"].dt.strftime("%Y-%m-%d")

# ── KURAL 2: Çıkış TM / Varış TM — güvenli " TM" suffix (mükerrer önleme) ──
for _col in ["Çıkış TM", "Varış TM"]:
    if _col in df.columns:
        df[_col] = (df[_col]
                    .astype(str)
                    .str.strip()
                    .str.replace(r'\s*TM$', '', regex=True)  # var olan TM'yi sil
                    .str.strip()
                    + " TM")                                   # temiz TM ekle

# Ölçek faktörünü yalnızca "Tahmin Edilen Desi" (net ortalama) üzerinden hesapla.
# Q25 / Q75 de aynı katsayıyla ölçeklenir; bunlar referans bilgidir, talep değil.
route_toplam = df["Tahmin Edilen Desi"].sum()
olcek = GLOBAL_TOPLAM / route_toplam

print(f"\n  Route model toplamı    : {route_toplam:>12,.0f}")
print(f"  Global hedef (ensemble): {GLOBAL_TOPLAM:>12,.0f}")
print(f"  Ölçekleme faktörü      : {olcek:.4f}")

# ── Ölçekleme uygula ───────────────────────────────────────────
df["Tahmin Edilen Desi"] = (df["Tahmin Edilen Desi"] * olcek).round(4)
# Güncelleme 1/2: Q25/Q75 artık tahmin sütunlarında olmayabilir—opsiyonel uygulanır.
if "Alt Sınır (Q25)" in df.columns:
    df["Alt Sınır (Q25)"] = (df["Alt Sınır (Q25)"] * olcek).round(1)
if "Üst Sınır (Q75)" in df.columns:
    df["Üst Sınır (Q75)"] = (df["Üst Sınır (Q75)"] * olcek).round(1)

# Güvenlik: negatif çıkmasın
df["Tahmin Edilen Desi"] = df["Tahmin Edilen Desi"].clip(lower=0)
if "Alt Sınır (Q25)" in df.columns:
    df["Alt Sınır (Q25)"] = df["Alt Sınır (Q25)"].clip(lower=0)
if "Üst Sınır (Q75)" in df.columns:
    df["Üst Sınır (Q75)"] = df["Üst Sınır (Q75)"].clip(lower=0)

# ── Tutarlılık kontrolü ──────────────────────────────────────────
kalibre_toplam = df["Tahmin Edilen Desi"].sum()
print(f"\n  Kalibre Tahmin Toplamı : {kalibre_toplam:>12,.0f}  ← OPTİMİZASYON BUNU KULLANIR")
if "Üst Sınır (Q75)" in df.columns:
    q75_toplam = df["Üst Sınır (Q75)"].sum()
    print(f"  Q75 Güvenlik Üst Sınırı: {q75_toplam:>12,.0f}  ← Yalnızca güvenlik marjı referansı")

# ── Özet tablolar ─────────────────────────────────────────────
# 1. Günlük özet
gunluk = (df.groupby(["Tarih", "Gün"])["Tahmin Edilen Desi"]
          .sum().reset_index().rename(columns={"Tahmin Edilen Desi": "Günlük Toplam"}))
gunluk["Günlük Toplam"] = gunluk["Günlük Toplam"].round(0)

if "Alt Sınır (Q25)" in df.columns and "Üst Sınır (Q75)" in df.columns:
    alt_gun = (df.groupby(["Tarih", "Gün"])["Alt Sınır (Q25)"]
               .sum().reset_index().rename(columns={"Alt Sınır (Q25)": "Alt Sınır (Q25)"}))
    ust_gun = (df.groupby(["Tarih", "Gün"])["Üst Sınır (Q75)"]
               .sum().reset_index().rename(columns={"Üst Sınır (Q75)": "Üst Sınır (Q75)"}))
    gunluk = gunluk.merge(alt_gun, on=["Tarih", "Gün"]).merge(ust_gun, on=["Tarih", "Gün"])
    gunluk[["Alt Sınır (Q25)", "Üst Sınır (Q75)"]] = gunluk[["Alt Sınır (Q25)", "Üst Sınır (Q75)"]].round(0)

# 2. Güzergah haftalık özet
guz = (df.groupby(["Çıkış TM", "Varış TM"])
       .agg({"Tahmin Edilen Desi": "sum"})
       .reset_index()
       .sort_values("Tahmin Edilen Desi", ascending=False)
       .reset_index(drop=True))
guz["Tahmin Edilen Desi"] = guz["Tahmin Edilen Desi"].round(0)
guz.columns = ["Çıkış TM", "Varış TM", "Haftalık Tahmin"]
guz["Pay (%)"] = (guz["Haftalık Tahmin"] / guz["Haftalık Tahmin"].sum() * 100).round(2)

# 3. Çıkış TM özeti
cikis = (df.groupby("Çıkış TM")
         .agg({"Tahmin Edilen Desi": "sum"})
         .reset_index()
         .sort_values("Tahmin Edilen Desi", ascending=False)
         .reset_index(drop=True))
cikis["Tahmin Edilen Desi"] = cikis["Tahmin Edilen Desi"].round(0)
cikis.columns = ["Çıkış TM", "Haftalık Tahmin"]
cikis["Pay (%)"] = (cikis["Haftalık Tahmin"] / cikis["Haftalık Tahmin"].sum() * 100).round(2)

# 4. Varış TM özeti
varis = (df.groupby("Varış TM")
         .agg({"Tahmin Edilen Desi": "sum"})
         .reset_index()
         .sort_values("Tahmin Edilen Desi", ascending=False)
         .reset_index(drop=True))
varis["Tahmin Edilen Desi"] = varis["Tahmin Edilen Desi"].round(0)
varis.columns = ["Varış TM", "Haftalık Tahmin"]
varis["Pay (%)"] = (varis["Haftalık Tahmin"] / varis["Haftalık Tahmin"].sum() * 100).round(2)

# 5. Güzergah × Gün matrisi (pivot)
pivot = df.pivot_table(
    index=["Çıkış TM", "Varış TM"],
    columns="Gün",
    values="Tahmin Edilen Desi",
    aggfunc="sum"
).round(0).reset_index()
gun_sira = ["Pazartesi", "Salı", "Çarşamba", "Perşembe", "Cuma", "Cumartesi", "Pazar"]
pivot_cols = ["Çıkış TM", "Varış TM"] + [g for g in gun_sira if g in pivot.columns]
pivot = pivot[pivot_cols].fillna(0)
pivot["Haftalık Toplam"] = pivot[gun_sira].sum(axis=1).round(0)
pivot = pivot.sort_values("Haftalık Toplam", ascending=False).reset_index(drop=True)

# 6. Model not kartı
# DÜZELTME: Eski notta "Araç planlamasında ÜST SINIR kullanın" yazıyordu.
# Bu yanlış yönlendirme kaldırıldı. Optimize edici Tahmin Edilen Desi kullanır.
not_data = {
    "Parametre": [
        "Tahmin Dönemi", "Güzergah Sayısı",
        "Haftalık Toplam Desi (Net Tahmin)",
        "Haftalık Alt Sınır (Q25)",
        "Haftalık Üst Sınır (Q75) — Güvenlik Marjı",
        "Global Model SMAPE (hafta içi)", "Global Model SMAPE (hafta sonu)",
        "Route Model MAE (ortalama)", "Ölçekleme Faktörü",
        "Eğitim Verisi", "Model Türü",
        "OPTİMİZASYON NOTU",
        "BÜTÇE NOTU"
    ],
    "Değer": [
        "11-17 Mayıs 2026", "89 güzergah",
        f"{df['Tahmin Edilen Desi'].sum():,.0f}  ← Araç ataması bu değer üzerinden yapılır",
        f"{df['Alt Sınır (Q25)'].sum():,.0f}",
        f"{df['Üst Sınır (Q75)'].sum():,.0f}  ← Sadece güvenlik marjı raporu; araç atamasında KULLANILMAZ",
        "%32.6 (±%22 hata payı)", "%52.8 (±%53 hata payı)",
        "2,615 desi/güzergah/gün",
        f"{olcek:.4f} (global ensemble ile uyumlu)",
        "1 Ocak – 10 Mayıs 2026 (130 gün)",
        "LightGBM + XGBoost + RandomForest + Naive (Route×DOW) Ensemble",
        "Kapasite ataması ve araç sayısı 'Tahmin Edilen Desi' (net ortalama) üzerinden yapılır. "
        "Q75 sadece 'bu kadar kargo gelebilir' güvenlik kontrolü için raporlanır.",
        "Kiralık araçların günlük sabit kirası Batık Maliyet'tir; optimizasyon döngüsüne dahil edilmez. "
        "Yalnızca nihai Genel Bütçe özeti sırasında tek seferlik eklenir."
    ]
}
not_df = pd.DataFrame(not_data)

# ── Excel yazma ───────────────────────────────────────────────
print("\n  Excel sayfaları yazılıyor...")
# GÜNCELLEME 2: Ana sayfa kolonu Tarih | Çıkış TM | Varış TM | Tahmin Edilen Desi
ana_kolon = ["Tarih", "Çıkış TM", "Varış TM", "Tahmin Edilen Desi"]
if "Gün" in df.columns:
    ana_kolon = ["Tarih", "Gün"] + [c for c in ana_kolon if c != "Tarih"]

with pd.ExcelWriter(OUTPUT_FINAL, engine="openpyxl") as writer:
    # KURAL 3: Kesin kolon sırası ve isimler — jüri grader tam string eşleme yapar
    _jury_cols = ["Tarih", "Çıkış TM", "Varış TM", "Tahmin Edilen Desi"]
    _df_jury = df[[c for c in _jury_cols if c in df.columns]].copy()
    _df_jury.columns = _jury_cols[:len(_df_jury.columns)]   # tam isim override
    _df_jury.to_excel(writer, sheet_name="📦 TM Tahminler", index=False)

    # Sayfa 2: Güzergah × Gün matrisi (en kullanışlı!)
    pivot.to_excel(writer, sheet_name="📅 Güzergah×Gün Matrisi", index=False)

    # Sayfa 3: Günlük özet
    gunluk.to_excel(writer, sheet_name="📊 Günlük Özet", index=False)

    # Sayfa 4: Güzergah özeti
    guz.to_excel(writer, sheet_name="🛣 Güzergah Özeti", index=False)

    # Sayfa 5: Çıkış TM (araç kalkış planlaması)
    cikis.to_excel(writer, sheet_name="🚚 Çıkış TM (Kalkış)", index=False)

    # Sayfa 6: Varış TM (kapasite planlaması)
    varis.to_excel(writer, sheet_name="📍 Varış TM (Varış)", index=False)

    # Sayfa 7: Model notu
    not_df.to_excel(writer, sheet_name="ℹ️ Model Notu", index=False)

print(f"  ✓ Excel → {OUTPUT_FINAL}")

# ── Terminal özet ─────────────────────────────────────────────
print()
print("=" * 60)
print("  FINAL ÖZET — Araç Optimizasyonu İçin")
print("=" * 60)
print(f"\n  Günlük tahminler (kalibre edilmiş):")
_has_q = "Alt Sınır (Q25)" in gunluk.columns and "Üst Sınır (Q75)" in gunluk.columns
_hdr = f"  {'Tarih':<12} {'Gün':<12} {'Tahmin':<14}" + (" {'Alt(Q25)':<14} {'Üst(Q75)'}" if _has_q else "")
print(f"  {'Tarih':<12} {'Gün':<12} {'Tahmin':<14}" + (" {'Alt(Q25)':<14} {'Üst(Q75)'}" if _has_q else ""))
print(f"  {'-'*65}")
for _, r in gunluk.iterrows():
    line = f"  {str(r['Tarih']):<12} {r['Gün']:<12} {r['Günlük Toplam']:>12,.0f}"
    if _has_q:
        line += f"   {r['Alt Sınır (Q25)']:>12,.0f}   {r['Üst Sınır (Q75)']:>12,.0f}"
    print(line)
print(f"  {'-'*65}")
print(f"  {'TOPLAM':<24} {gunluk['Günlük Toplam'].sum():>12,.0f}")

print(f"\n  Çıkış TM haftalık yük (kalibre edilmiş):")
print(f"  {'Çıkış TM':<18} {'Tahmin':<14} {'Pay (%)'}")
print(f"  {'-'*42}")
for _, r in cikis.iterrows():
    print(f"  {r['Çıkış TM']:<18} {r['Haftalık Tahmin']:>12,.0f} {r['Pay (%)']:>7.1f}%")

print(f"\n  En yoğun 10 güzergah:")
print(f"  {'Çıkış':<15} {'Varış':<15} {'Haftalık':<14} {'Pay'}")
print(f"  {'-'*52}")
for _, r in guz.head(10).iterrows():
    print(f"  {r['Çıkış TM']:<15} {r['Varış TM']:<15} {r['Haftalık Tahmin']:>12,.0f} {r['Pay (%)']:>5.1f}%")

print(f"\n  ⚠  OPTİMİZASYON KULLANIM NOTU:")
print(f"  Araç KAPASITE ATAMASI: 'Tahmin Edilen Desi' (net ortalama) kullanılır.")
print(f"  Q75 Üst Sınırı       : Yalnızca güvenlik marjı kontrolü içindir.")
print(f"  Kiralık Sabit Maliyet: Optimizasyon döngüsünde YER ALMAZ (Batık Maliyet).")
print(f"\n  📁 Dosya: {OUTPUT_FINAL}")
print()
