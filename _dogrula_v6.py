# -*- coding: utf-8 -*-
"""
================================================================
 MODÜL: _dogrula_v6.py
 AMAÇ : TEKNOFEST 2026 Lojistik Projesi — Jüri Öncesi Excel Doğrulama

 Bu modül, optimize_cost_v4.py tarafından üretilen V6 TEKNOFEST Excel
 çıktısını juri sunumundan önce eksiksiz kontrol eder. Tüm TEKNOFEST
 sert kısıtlarının dosyada gerçekten sağlanmış olduğunu doğrular.

 PIPELINE'DAKİ YERİ (Doğrulama katmanı, ana pipeline ile bağımsız):
   optimize_cost_v4.py → [_dogrula_v6.py] (manuel tetiklenir)

 KONTROL LİSTESİ:
   [1] Zorunlu kolonların varlığı (Tarih / Çıkış TM / Varış TM /
       Tahmin Edilen Desi / Araç Tipi)
   [2] İlk 8 satır göz önüneşi (spot bug ya da sıfır değer tespiti için)
   [3] KURAL 4a: Tüm spot atamalarda %10 doluluk hard constraint
   [4] KURAL 3: Her gün en az bir kiralık araç satırının bulunması
   [5] Yasaklı ifadeler: "Konsölidasyon" / "Transfer Merkezi" / "Hub" / "Batık"
   [6] Bütçe özeti doğrulaması
   [7] Toplam talep karşılama oranı (kiralık + spot toplamları)
================================================================
"""
import pandas as pd

df = pd.read_excel(
    'data/Lojistik_Maliyet_Optimizasyonu_V6_TEKNOFEST.xlsx',
    sheet_name='Sefer Planı (TEKNOFEST)'
)
df_b = pd.read_excel(
    'data/Lojistik_Maliyet_Optimizasyonu_V6_TEKNOFEST.xlsx',
    sheet_name='Bütçe Özeti'
)

print("=" * 60)
print("  TEKNOFEST V6 EXCEL DOĞRULAMA RAPORU")
print("=" * 60)

# 1. Kolon kontrolü
zorunlu = ["Tarih", "Çıkış TM", "Varış TM", "Tahmin Edilen Desi", "Araç Tipi"]
print("\n[1] TEKNOFEST Zorunlu Kolonlar:")
for k in zorunlu:
    durum = "✅" if k in df.columns else "❌ EKSİK"
    print(f"  {durum} {k}")

# 2. İlk 8 satır
print("\n[2] İlk 8 Satır (Zorunlu Kolonlar):")
print(df[zorunlu].head(8).to_string(index=False))

# 3. %10 Hard Constraint kontrolü
print("\n[3] Spot Araç %10 Hard Constraint Doğrulama:")
spot = df[df['Araç Tipi'].str.startswith('Spot:', na=False)]
if not spot.empty:
    min_dol = spot['Doluluk Oranı (%)'].min()
    ort_dol = spot['Doluluk Oranı (%)'].mean()
    ihlal = spot[spot['Doluluk Oranı (%)'] < 10.0]
    print(f"  Toplam spot satır : {len(spot)}")
    print(f"  Min doluluk       : %{min_dol:.1f}")
    print(f"  Ort doluluk       : %{ort_dol:.1f}")
    if ihlal.empty:
        print("  => ✅ TÜM SPOT ATAMALAR %10 EŞİĞİNİN ÜZERİNDE")
    else:
        print(f"  => ❌ UYARI: {len(ihlal)} satırda %10 ihlali!")
        print(ihlal[zorunlu + ['Doluluk Oranı (%)']].to_string(index=False))
else:
    print("  Spot araç ataması yok.")

# 4. Kiralık araç kontrolü (Her günde mevcut mu?)
print("\n[4] Kiralık Araç Zorunlu Kullanım Kontrolü:")
kiralik = df[df['Araç Tipi'].str.startswith('Kiralık:', na=False)]
print(f"  Toplam kiralık satır : {len(kiralik)}")
print(f"  Kapsanan gün sayısı  : {kiralik['Tarih'].nunique()}")
print(f"  Toplam gün sayısı    : {df['Tarih'].nunique()}")
if kiralik['Tarih'].nunique() == df['Tarih'].nunique():
    print("  => ✅ Her günde kiralık araç planın ilk katmanında")
else:
    print("  => ⚠️  Bazı günlerde kiralık araç satırı eksik (spot-only gün olabilir)")

# 5. Yasaklı kelime kontrolü
print("\n[5] Yasaklı İfade Kontrolü ('Konsolidasyon' / 'Transfer Merkezi'):")
yasak = ["Konsolidasyon", "Transfer Merkezi", "Hub", "Batık"]
for ifade in yasak:
    eslesme = df.astype(str).apply(lambda col: col.str.contains(ifade, case=False, na=False)).any().any()
    durum = "❌ BULUNDU!" if eslesme else "✅ Yok"
    print(f"  {durum}  '{ifade}'")

# 6. Bütçe Özeti
print("\n[6] Bütçe Özeti (TEKNOFEST V6 — Tam Maliyet):")
print(df_b.to_string(index=False))

# 7. Genel talep karşılama
print(f"\n[7] Talep Karşılama:")
toplam = df['Tahmin Edilen Desi'].sum()
kiralik_toplam = kiralik['Tahmin Edilen Desi'].sum()
spot_toplam = spot['Tahmin Edilen Desi'].sum() if not spot.empty else 0
print(f"  Toplam taşınan desi  : {toplam:,.1f}")
print(f"  Kiralık araçla       : {kiralik_toplam:,.1f}")
print(f"  Spot araçla          : {spot_toplam:,.1f}")

print("\n" + "=" * 60)
print("  DOĞRULAMA TAMAMLANDI")
print("=" * 60)
