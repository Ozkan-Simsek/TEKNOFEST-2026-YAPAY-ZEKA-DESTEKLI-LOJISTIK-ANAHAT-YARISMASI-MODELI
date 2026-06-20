#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
================================================================
 MODÜL: map_kiralik.py
 AMAÇ : TEKNOFEST 2026 Lojistik Projesi — Kiralık Araç Rota Haritası

 Bu modül, tahmin edilen her güzel çifti (orijinal rota) için kiralık
 hat olup olmadığına karar verir ve optimize_cost_v4.py için hazırlık
 Excel dosyasını (Optimizasyon_Kiralik_11_17_mayis.xlsx) üretir.

 PIPELINE'DAKİ YERİ:
   finalize.py → [map_kiralik.py] → optimize_cost_v4.py

 TEKNOFEST KURAL 2 — Konsölidasyon Yasağı (Hub-and-Spoke İptali):
   GÜNCELLEME 4 ile hub-and-spoke koordinat eşleştirmesi tamamen kaldırıldı.
   Yalnızca iki karar vardır:
     (a) Orijinal rota zaten kiralık hatsa → kiralık hat kullan
     (b) Değilse → rota P2P olarak kalır, spot_sadece=True
   Hiçbir koordinat mesafesi hesabı yapılmaz; hub merkezde birleştirme yoktur.
================================================================
"""

import pandas as pd
import numpy as np
import warnings
import os
import math

warnings.filterwarnings("ignore")

print("=" * 60)
print("  KİRALIK ARAÇ AKTARMA HARİTASI — AKILLI MESAFE BAZLI")
print("=" * 60)


# ─────────────────────────────────────────────────────────────
# Yardımcı fonksiyon: Haversine (iki nokta arası km)
# ─────────────────────────────────────────────────────────────
def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Haversine formülüyle iki koordinat arasındaki kuş uçuşu mesafesini hesaplar (km).

    NOT: GÜNCELLEME 4 sonrasında bu fonksiyon artık rota atama kararında
    aktif olarak kullanılmamaktadır. Hub-and-Spoke koordinat eşleştirmesi
    iptal edildiğinden fonksiyon sadece referans amaçlı bırakılmıştır.
    """
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return R * 2 * math.asin(math.sqrt(a))


try:
    # ── 1. Verileri Oku ───────────────────────────────────────
    df_tahmin = pd.read_excel(
        "data/tahmin_FINAL_11_17_mayis.xlsx",
        sheet_name="📦 TM Tahminler"
    )

    kiralik_dosya = [f for f in os.listdir("data") if "Kiral" in f and f.endswith(".xlsx")][0]
    df_kiralik    = pd.read_excel(os.path.join("data", kiralik_dosya))
    # Kiralık hatları bir küme olarak tut (O(1) arama için)
    kiralik_rotalar: set = set(
        zip(df_kiralik["Çıkış Transfer Merkezi"], df_kiralik["Varış Transfer Merkezi"])
    )

    koor_dosya = [f for f in os.listdir("data") if "Koor" in f and f.endswith(".xlsx")][0]
    df_koor    = pd.read_excel(os.path.join("data", koor_dosya))

    # Koordinat sözlüğü: Şehir → (Lat, Lon)
    coords: dict = {}
    for _, row in df_koor.iterrows():
        cols  = df_koor.columns
        sehir = str(row[cols[0]]).strip()
        coords[sehir] = (float(row[cols[1]]), float(row[cols[2]]))

    # Eksik koordinat uyarısı
    tum_sehirler = set(df_tahmin["Çıkış TM"]).union(set(df_tahmin["Varış TM"]))
    for s in tum_sehirler:
        if s not in coords:
            print(f"  ⚠ Uyarı: '{s}' için koordinat yok → (0,0) varsayılıyor.")
            coords[s] = (0.0, 0.0)

    # ── 2. Her Orijinal Rota İçin P2P Karar Üret ────────────────────────
    # GÜNCELLEME 4: Koordinat mesafesi hesaplaması kaldırıldı.
    # Yalnızca: (a) rota zaten kiralık hatsa → kiralık hat kullan;
    #           (b) değilse → doğrudan spot'a bırak (spot_sadece=True).
    rota_karari: dict = {}

    for cikis_tm in df_tahmin["Çıkış TM"].unique():
        for varis_tm in df_tahmin["Varış TM"].unique():
            orijinal = (cikis_tm, varis_tm)

            if orijinal in kiralik_rotalar:
                # Rota zaten kiralık hat — doğrudan kullan
                rota_karari[orijinal] = {
                    "kir_cikis": cikis_tm,
                    "kir_varis":  varis_tm,
                    "spot_sadece": False
                }
            else:
                # GÜNCELLEME 4: Hub-and-Spoke mapping yok — rota P2P olarak kalıyor.
                rota_karari[orijinal] = {
                    "kir_cikis":   cikis_tm,
                    "kir_varis":   varis_tm,
                    "spot_sadece": True
                }

    # ── 3. Veriyi Dönüştür ────────────────────────────────────
    mapped_data = []
    for _, row in df_tahmin.iterrows():
        c = row["Çıkış TM"]
        v = row["Varış TM"]
        karar = rota_karari[(c, v)]

        gun_val = row.get("Gün", "")
        mapped_data.append({
            "Tarih":              row["Tarih"],
            "Gün":               gun_val,
            "Kiralık Çıkış TM":  karar["kir_cikis"],
            "Kiralık Varış TM":  karar["kir_varis"],
            "Orijinal Çıkış TM": c,
            "Orijinal Varış TM": v,
            "Tahmin Edilen Desi": row["Tahmin Edilen Desi"],
            "Spot Sadece":       karar["spot_sadece"]
        })

    df_mapped = pd.DataFrame(mapped_data)

    # ── 4. Kiralık Hat Toplamları (Spot Sadece = False olanlar) ──
    df_kiralik_hat = df_mapped[~df_mapped["Spot Sadece"]].copy()
    df_opt = df_kiralik_hat.groupby(
        ["Tarih", "Gün", "Kiralık Çıkış TM", "Kiralık Varış TM"]
    ).agg({
        "Tahmin Edilen Desi": "sum"   # optimizasyon bu sütunu kullanır
    }).reset_index()

    # Kiralık araç bilgilerini birleştir
    df_opt = df_opt.merge(
        df_kiralik,
        left_on=["Kiralık Çıkış TM", "Kiralık Varış TM"],
        right_on=["Çıkış Transfer Merkezi", "Varış Transfer Merkezi"],
        how="left"
    )
    df_opt.drop(columns=["Çıkış Transfer Merkezi", "Varış Transfer Merkezi"], inplace=True)

    # ── 5. Spot Sadece Rotalar (Kiraliktan bağımsız) ─────────
    df_spot_sadece = df_mapped[df_mapped["Spot Sadece"]].copy()
    df_spot_opt = df_spot_sadece.groupby(
        ["Tarih", "Gün", "Orijinal Çıkış TM", "Orijinal Varış TM"]
    ).agg({
        "Tahmin Edilen Desi": "sum"
    }).reset_index()

    # ── 6. Haftalık Özet ────────────────────────────────────
    haftalik = df_opt.groupby(
        ["Kiralık Çıkış TM", "Kiralık Varış TM", "Araç sayısı", "Araç Türü"]
    ).agg({
        "Tahmin Edilen Desi": "sum"    # NET ORTALAMA — nokta tahmini
    }).reset_index().sort_values("Tahmin Edilen Desi", ascending=False)

    # ── 7. Harita Özet DataFrame ─────────────────────────────
    harita_df = pd.DataFrame([
        {
            "Orijinal Çıkış":       k[0],
            "Orijinal Varış":        k[1],
            "Atanan Kiralık Çıkış": v["kir_cikis"],
            "Atanan Kiralık Varış": v["kir_varis"],
            "Spot Sadece (Detour)": v["spot_sadece"]
        }
        for k, v in rota_karari.items()
    ])

    # ── 8. Excel'e Yaz ────────────────────────────────────────
    cikis_dosyasi = "data/Optimizasyon_Kiralik_11_17_mayis.xlsx"
    with pd.ExcelWriter(cikis_dosyasi, engine="openpyxl") as writer:
        # Ana optimizasyon verisi: kiralık hatlara atanan günlük talepler
        df_opt.to_excel(writer, sheet_name="Günlük Optimizasyon Verisi", index=False)
        # Spot sadece: kiralık hatta bağlamak detour yarattığı için es geçilmiş rotalar
        df_spot_opt.to_excel(writer, sheet_name="Spot Sadece Rotalar", index=False)
        # Haftalık kiralık hat yükleri (NET ORTALAMA üzerinden)
        haftalik.to_excel(writer, sheet_name="Haftalık Araç Yükleri", index=False)
        # Hangi şehir nereye atandı / spot'a bırakıldı
        harita_df.to_excel(writer, sheet_name="Aktarma Haritası", index=False)

    # ── 9. Terminal Özet ──────────────────────────────────────
    n_kiralik_hat = (~harita_df["Spot Sadece (Detour)"]).sum()
    n_spot_sadece = harita_df["Spot Sadece (Detour)"].sum()

    print(f"\n  [BAŞARILI] P2P rota kararı oluşturuldu (Hub-and-Spoke iptal edildi).")
    print(f"  ✓ Dosya: {cikis_dosyasi}")
    print(f"\n  ── Karar Özeti ────────────────────────────────────")
    print(f"  Kiralık hatta atılan rota sayısı : {n_kiralik_hat}")
    print(f"  P2P Spot'a bırakılan rota sayısı  : {n_spot_sadece}")
    print(f"\n  [GÜNCELLEME 4] Hub-and-Spoke mapping DEVRE DIŞI. Tüm olmayan kiralık")
    print(f"  rotalar doğrudan Point-to-Point Spot VRP'ye aktarılır.")
    print(f"\n  Kiralık Rota İçin Haftalık Toplam Yük (Net Tahmin):")
    print(haftalik.to_string(index=False))

except Exception as e:
    import traceback
    print("\n  HATA:")
    traceback.print_exc()
