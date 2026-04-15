# ============================================================
# TRADECLEANSE — NOTEBOOK 03 : Validation du Dataset Nettoye
# DCLE821 — QuantAxis Capital
# Etudiant(s) : ___________________________________
# Date        : ___________________________________
# ============================================================
#
# Approche choisie : tests pandas + assertions Python (approche B).
# Chaque test retourne [PASS] ou [FAIL] avec le detail.
# ============================================================

import pandas as pd
import numpy as np
import re
import warnings
warnings.filterwarnings('ignore')

# Chargement du dataset nettoye
df = pd.read_csv('data/tradecleanse_clean.csv', low_memory=False)
df['trade_date'] = pd.to_datetime(df['trade_date'], errors='coerce')
df['settlement_date'] = pd.to_datetime(df['settlement_date'], errors='coerce')
print(f"Dataset nettoye charge : {df.shape[0]} lignes x {df.shape[1]} colonnes\n")

results = []

def run_test(test_id, name, passed, detail):
    """Enregistre et affiche le resultat d'un test."""
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] Expectation {test_id} — {name}")
    print(f"          {detail}\n")
    results.append({
        'test_id': test_id,
        'name': name,
        'status': status,
        'detail': detail
    })

print("=" * 60)
print("SUITE DE VALIDATION — 14 EXPECTATIONS")
print("=" * 60 + "\n")

# ============================================================
# EXPECTATION 1 — Unicite de trade_id
# ============================================================
dups = df['trade_id'].duplicated().sum()
run_test(1, "Unicite de trade_id",
         dups == 0,
         f"{dups} doublons trouves sur trade_id")

# ============================================================
# EXPECTATION 2 — Colonnes obligatoires non nulles
# ============================================================
# counterparty_id a ete pseudonymise en counterparty_id_hash
mandatory = ['trade_id', 'counterparty_id_hash', 'isin', 'trade_date',
             'asset_class', 'price', 'quantity', 'default_flag']
null_counts = {col: df[col].isnull().sum() for col in mandatory}
total_nulls = sum(null_counts.values())
detail_nulls = ", ".join([f"{c}={v}" for c, v in null_counts.items() if v > 0])
run_test(2, "Colonnes obligatoires non nulles",
         total_nulls == 0,
         f"NaN dans colonnes obligatoires : {detail_nulls if detail_nulls else 'aucun'}")

# ============================================================
# EXPECTATION 3 — settlement_date >= trade_date
# ============================================================
mask_3 = df['settlement_date'].notna() & df['trade_date'].notna()
violations_3 = (df.loc[mask_3, 'settlement_date'] < df.loc[mask_3, 'trade_date']).sum()
run_test(3, "settlement_date >= trade_date",
         violations_3 == 0,
         f"{violations_3} lignes ou settlement_date < trade_date")

# ============================================================
# EXPECTATION 4 — bid < ask sur toutes les lignes
# ============================================================
mask_4 = df['bid'].notna() & df['ask'].notna()
violations_4 = (df.loc[mask_4, 'bid'] >= df.loc[mask_4, 'ask']).sum()
run_test(4, "bid < ask",
         violations_4 == 0,
         f"{violations_4} lignes ou bid >= ask")

# ============================================================
# EXPECTATION 5 — price dans [bid * 0.995, ask * 1.005]
# ============================================================
mask_5 = df['price'].notna() & df['bid'].notna() & df['ask'].notna()
sub = df[mask_5]
violations_5 = ((sub['price'] < sub['bid'] * 0.995) |
                (sub['price'] > sub['ask'] * 1.005)).sum()
run_test(5, "price dans fourchette [bid*0.995, ask*1.005]",
         violations_5 == 0,
         f"{violations_5} lignes hors fourchette")

# ============================================================
# EXPECTATION 6 — mid_price coherent avec (bid + ask) / 2
# ============================================================
mask_6 = df['mid_price'].notna() & df['bid'].notna() & df['ask'].notna()
mid_theo = (df.loc[mask_6, 'bid'] + df.loc[mask_6, 'ask']) / 2
ecart = (abs(df.loc[mask_6, 'mid_price'] - mid_theo) / mid_theo)
violations_6 = (ecart > 0.01).sum()
run_test(6, "mid_price coherent avec (bid+ask)/2 (tolerance 1%)",
         violations_6 == 0,
         f"{violations_6} lignes avec ecart > 1%")

# ============================================================
# EXPECTATION 7 — asset_class dans le referentiel
# ============================================================
valid_ac = {'equity', 'bond', 'derivative', 'fx'}
ac_values = set(df['asset_class'].dropna().unique())
invalids_7 = ac_values - valid_ac
run_test(7, "asset_class dans {equity, bond, derivative, fx}",
         len(invalids_7) == 0,
         f"Valeurs presentes : {ac_values}. Invalides : {invalids_7 if invalids_7 else 'aucune'}")

# ============================================================
# EXPECTATION 8 — Pas de contradiction rating + defaut
# ============================================================
invest_grades = {'aaa', 'aa', 'a'}
mask_8 = df['credit_rating'].isin(invest_grades) & (df['default_flag'] == 1)
violations_8 = mask_8.sum()
run_test(8, "Pas de rating investissement (AAA/AA/A) + default_flag=1",
         violations_8 == 0,
         f"{violations_8} contradictions trouvees")

# ============================================================
# EXPECTATION 9 — notional_eur > 0
# ============================================================
violations_9 = (df['notional_eur'] <= 0).sum()
run_test(9, "notional_eur > 0",
         violations_9 == 0,
         f"{violations_9} lignes avec notional_eur <= 0")

# ============================================================
# EXPECTATION 10 — country_risk dans [0, 100]
# ============================================================
mask_10 = df['country_risk'].notna()
violations_10 = ((df.loc[mask_10, 'country_risk'] < 0) |
                 (df.loc[mask_10, 'country_risk'] > 100)).sum()
run_test(10, "country_risk dans [0, 100]",
         violations_10 == 0,
         f"{violations_10} lignes hors [0, 100]")

# ============================================================
# EXPECTATION 11 — Format ISIN valide
# ============================================================
isin_regex = r'^[A-Z]{2}[A-Z0-9]{10}$'
mask_11 = df['isin'].notna()
isin_valid = df.loc[mask_11, 'isin'].apply(lambda x: bool(re.match(isin_regex, str(x))))
violations_11 = (~isin_valid).sum()
run_test(11, "Format ISIN valide (2 lettres + 10 alphanumeriques)",
         violations_11 == 0,
         f"{violations_11} ISIN invalides sur {mask_11.sum()} testes")

# ============================================================
# EXPECTATION 12 — volatility_30d dans [0.1, 200]
# ============================================================
mask_12 = df['volatility_30d'].notna()
violations_12 = ((df.loc[mask_12, 'volatility_30d'] < 0.1) |
                 (df.loc[mask_12, 'volatility_30d'] > 200)).sum()
run_test(12, "volatility_30d dans [0.1, 200]",
         violations_12 == 0,
         f"{violations_12} lignes hors [0.1, 200]")

# ============================================================
# EXPECTATION 13 — Completude globale > 90%
# ============================================================
total_cells = df.shape[0] * df.shape[1]
total_nan = df.isnull().sum().sum()
completude = (1 - total_nan / total_cells) * 100
run_test(13, "Completude globale > 90%",
         completude > 90,
         f"Completude = {completude:.2f}% ({total_nan} NaN sur {total_cells} cellules)")

# ============================================================
# EXPECTATION 14 — Absence de PII en clair
# ============================================================
pii_cols = ['counterparty_name', 'trader_id', 'counterparty_id']
pii_present = [col for col in pii_cols if col in df.columns]
hash_cols = [col for col in df.columns if col.endswith('_hash')]
run_test(14, "Absence de PII en clair",
         len(pii_present) == 0,
         f"PII en clair : {pii_present if pii_present else 'aucune'}. "
         f"Colonnes hashees presentes : {hash_cols}")

# ============================================================
# SCORE FINAL
# ============================================================
results_df = pd.DataFrame(results)
passed = (results_df['status'] == 'PASS').sum()
total = len(results_df)

print("=" * 60)
print(f"SCORE FINAL : {passed}/{total} expectations passees")
print("=" * 60)

if passed == total:
    print("Le dataset est CERTIFIE pour l'entrainement du modele de risque.")
else:
    failed = results_df[results_df['status'] == 'FAIL']
    print("Tests echoues :")
    for _, row in failed.iterrows():
        print(f"  - Expectation {row['test_id']}: {row['name']}")

results_df.to_csv('ge_validation_report.csv', index=False)
print(f"\nRapport exporte dans ge_validation_report.csv")
