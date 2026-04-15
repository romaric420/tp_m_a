# ============================================================
# TRADECLEANSE — NOTEBOOK 01 : Audit & Profiling Initial
# DCLE821 — QuantAxis Capital
# Etudiant    : Hippolyte Romaric TCHOFFO SAMBO
# Date        : 15 avril 2026
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sqlite3
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CELLULE 1 — Chargement multi-sources
# ============================================================

SENTINELS = ['#N/A', 'N/A', '#VALUE!', '-', 'nd', 'null', 'None', 'na',
             'NaN', 'missing', '99999']

df = pd.read_csv(
    'data/tradecleanse_raw.csv',
    low_memory=False,
    encoding='utf-8',
    sep=',',
    na_values=SENTINELS,
    keep_default_na=True
)
print(f"Dataset charge : {df.shape[0]} lignes x {df.shape[1]} colonnes\n")

# --- Simulation des 3 sources selon le README ---

bloomberg_cols = ['isin', 'price', 'bid', 'ask', 'mid_price',
                  'volume_j', 'volatility_30d']
murex_cols = ['trade_id', 'counterparty_id', 'trade_date', 'settlement_date',
              'asset_class', 'notional_eur', 'quantity', 'trader_id']
refinitiv_cols = ['counterparty_name', 'credit_rating', 'default_flag',
                  'sector', 'country_risk']

df_bloomberg = df[bloomberg_cols].copy()
df_bloomberg['source'] = 'Bloomberg'

df_murex = df[murex_cols].copy()
df_murex['source'] = 'Murex'

df_refinitiv = df[refinitiv_cols].copy()
df_refinitiv['source'] = 'Refinitiv'

# Simulation Murex en base SQL (sqlite3 en memoire)
conn = sqlite3.connect(':memory:')
df_murex.to_sql('murex_trades', conn, index=False, if_exists='replace')
df_murex_sql = pd.read_sql('SELECT * FROM murex_trades', conn)
conn.close()

print(f"Bloomberg  : {df_bloomberg.shape}")
print(f"Murex (SQL): {df_murex_sql.shape}")
print(f"Refinitiv  : {df_refinitiv.shape}")

# Reconsolidation des 3 sources
df_consolidated = pd.concat(
    [df_bloomberg.reset_index(), df_murex.reset_index(), df_refinitiv.reset_index()],
    axis=1
)
df_consolidated = df_consolidated.loc[:, ~df_consolidated.columns.duplicated()]
print(f"\nDataset reconsolide : {df_consolidated.shape}")

# ============================================================
# CELLULE 2 — Profiling initial
# ============================================================

print("\n" + "=" * 60)
print("PROFILING INITIAL")
print("=" * 60)

# Shape
print(f"\n--- Shape ---")
print(f"Lignes  : {df.shape[0]}")
print(f"Colonnes: {df.shape[1]}")

# Types detectes
print(f"\n--- Types pandas ---")
print(df.dtypes)

# Valeurs manquantes
print(f"\n--- Valeurs manquantes ---")
nan_count = df.isnull().sum()
nan_pct = (df.isnull().sum() / len(df) * 100).round(2)
nan_report = pd.DataFrame({'nb_missing': nan_count, 'pct_missing': nan_pct})
nan_report = nan_report.sort_values('pct_missing', ascending=False)
print(nan_report[nan_report['nb_missing'] > 0])

# Statistiques descriptives (numeriques)
print(f"\n--- Statistiques descriptives (numeriques) ---")
print(df.describe().round(2))

# Cardinalite des colonnes categorielles
print(f"\n--- Cardinalite (categorielles) ---")
cat_cols = df.select_dtypes(include='object').columns
for col in cat_cols:
    print(f"  {col}: {df[col].nunique()} valeurs uniques")

# Distribution des colonnes categorielles
print(f"\n--- Distribution des categorielles ---")
for col in ['asset_class', 'credit_rating', 'sector']:
    print(f"\n  [{col}]")
    print(df[col].value_counts().to_string())

# Doublons
exact_dups = df.duplicated().sum()
tid_dups = df['trade_id'].duplicated().sum()
print(f"\n--- Doublons ---")
print(f"  Doublons exacts     : {exact_dups}")
print(f"  Doublons sur trade_id: {tid_dups}")

# ============================================================
# CELLULE 3 — Detection des anomalies
# ============================================================

print("\n" + "=" * 60)
print("DETECTION DES ANOMALIES")
print("=" * 60)

anomalies = []

# 1. Doublons sur trade_id
n = df['trade_id'].duplicated().sum()
anomalies.append({
    'type': 'Doublon',
    'colonne': 'trade_id',
    'nb_lignes': n,
    'criticite': 'HAUTE — un trade_id duplique fausse le calcul du risque net'
})

# 2. Valeurs manquantes significatives
for col in df.columns:
    nb = df[col].isnull().sum()
    if nb > 0:
        pct = round(nb / len(df) * 100, 2)
        crit = 'HAUTE' if col in ['trade_id', 'price', 'default_flag'] else 'MOYENNE'
        anomalies.append({
            'type': 'Valeur manquante',
            'colonne': col,
            'nb_lignes': nb,
            'criticite': f'{crit} — {pct}% de NaN'
        })

# 3. settlement_date < trade_date (regle T+2)
df['trade_date_dt'] = pd.to_datetime(df['trade_date'], errors='coerce')
df['settlement_date_dt'] = pd.to_datetime(df['settlement_date'], errors='coerce')
mask_settle = df['settlement_date_dt'] < df['trade_date_dt']
n = mask_settle.sum()
anomalies.append({
    'type': 'Incoherence',
    'colonne': 'settlement_date < trade_date',
    'nb_lignes': n,
    'criticite': 'HAUTE — viole la regle T+2, regle de reglement obligatoire'
})

# 4. bid > ask (fourchette inversee)
mask_bid_ask = df['bid'] > df['ask']
n = mask_bid_ask.sum()
anomalies.append({
    'type': 'Incoherence',
    'colonne': 'bid > ask',
    'nb_lignes': n,
    'criticite': 'HAUTE — fourchette inversee, rend mid_price et spread invalides'
})

# 5. mid_price incoherent avec (bid + ask) / 2 (tolerance 1%)
mid_theo = (df['bid'] + df['ask']) / 2
mask_mid = (abs(df['mid_price'] - mid_theo) / mid_theo > 0.01)
mask_mid = mask_mid & df['mid_price'].notna() & mid_theo.notna()
n = mask_mid.sum()
anomalies.append({
    'type': 'Incoherence',
    'colonne': 'mid_price vs (bid+ask)/2',
    'nb_lignes': n,
    'criticite': 'MOYENNE — fausse le mark-to-market des positions'
})

# 6. price hors fourchette [bid*0.995, ask*1.005]
mask_price = (df['price'] < df['bid'] * 0.995) | (df['price'] > df['ask'] * 1.005)
mask_price = mask_price & df['price'].notna() & df['bid'].notna() & df['ask'].notna()
n = mask_price.sum()
anomalies.append({
    'type': 'Incoherence',
    'colonne': 'price hors fourchette bid/ask',
    'nb_lignes': n,
    'criticite': 'HAUTE — prix d\'execution suspect, possible erreur de saisie'
})

# 7. notional_eur negatif
n = (df['notional_eur'] < 0).sum()
anomalies.append({
    'type': 'Incoherence',
    'colonne': 'notional_eur < 0',
    'nb_lignes': n,
    'criticite': 'HAUTE — un notionnel negatif fausse l\'exposition de risque'
})

# 8. asset_class non standard (variantes non normalisees)
valid_ac = {'equity', 'bond', 'derivative', 'fx'}
mask_ac = ~df['asset_class'].str.lower().str.strip().isin(valid_ac)
n = mask_ac.sum()
anomalies.append({
    'type': 'Format',
    'colonne': 'asset_class (variantes non normalisees)',
    'nb_lignes': n,
    'criticite': 'MOYENNE — empeche les agregations par classe d\'actif'
})

# 9. country_risk hors [0, 100]
cr_numeric = pd.to_numeric(df['country_risk'], errors='coerce')
mask_cr = (cr_numeric < 0) | (cr_numeric > 100)
n = mask_cr.sum()
anomalies.append({
    'type': 'Hors plage',
    'colonne': 'country_risk hors [0, 100]',
    'nb_lignes': n,
    'criticite': 'MOYENNE — score de risque pays invalide pour le modele'
})

# 10. volatility_30d hors [0.1, 200]
vol_numeric = pd.to_numeric(df['volatility_30d'], errors='coerce')
mask_vol = (vol_numeric < 0.1) | (vol_numeric > 200)
n = mask_vol.sum()
anomalies.append({
    'type': 'Hors plage',
    'colonne': 'volatility_30d hors [0.1, 200]',
    'nb_lignes': n,
    'criticite': 'MOYENNE — volatilite aberrante, biais le modele de risque'
})

# 11. default_flag != 0 ou 1
mask_df = ~df['default_flag'].isin([0, 1])
n = mask_df.sum()
anomalies.append({
    'type': 'Format',
    'colonne': 'default_flag (valeur invalide)',
    'nb_lignes': n,
    'criticite': 'HAUTE — variable cible du modele ML'
})

# 12. Contradiction rating investissement (AAA/AA/A) + default_flag = 1
mask_rating_default = (
    df['credit_rating'].isin(['AAA', 'AA', 'A']) & (df['default_flag'] == 1)
)
n = mask_rating_default.sum()
anomalies.append({
    'type': 'Incoherence',
    'colonne': 'credit_rating (AAA/AA/A) + default_flag=1',
    'nb_lignes': n,
    'criticite': 'HAUTE — contradiction metier : investment grade en defaut impossible'
})

# Construction du rapport
anomalies_report = pd.DataFrame(anomalies)
anomalies_report = anomalies_report[anomalies_report['nb_lignes'] > 0]
anomalies_report = anomalies_report.sort_values('nb_lignes', ascending=False).reset_index(drop=True)

print("\n--- Rapport d'anomalies ---")
print(anomalies_report.to_string(index=False))
print(f"\nTotal : {len(anomalies_report)} types d'anomalies detectes")

# Nettoyage des colonnes temporaires
df.drop(columns=['trade_date_dt', 'settlement_date_dt'], inplace=True)

# ============================================================
# CELLULE 4 — Visualisations
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('TradeCleanse — Rapport de Profiling Initial\nQuantAxis Capital',
             fontsize=14, fontweight='bold', y=1.01)

# --- Graphique 1 : Taux de valeurs manquantes par colonne ---
ax1 = axes[0, 0]
nan_pct_all = (df.isnull().sum() / len(df) * 100).sort_values(ascending=True)
colors_nan = ['#e74c3c' if v > 10 else '#f39c12' if v > 0 else '#2ecc71'
              for v in nan_pct_all]
nan_pct_all.plot(kind='barh', ax=ax1, color=colors_nan, edgecolor='black', linewidth=0.5)
ax1.set_title('Taux de valeurs manquantes par colonne (%)', fontweight='bold')
ax1.set_xlabel('% de NaN')
ax1.axvline(x=10, color='red', linestyle='--', alpha=0.5, label='Seuil 10%')
ax1.legend()

# --- Graphique 2 : Distribution de asset_class (toutes variantes) ---
ax2 = axes[0, 1]
ac_counts = df['asset_class'].value_counts()
valid_labels = {'equity', 'bond', 'derivative', 'fx'}
colors_ac = ['#2ecc71' if lbl.lower().strip() in valid_labels else '#e74c3c'
             for lbl in ac_counts.index]
ac_counts.plot(kind='barh', ax=ax2, color=colors_ac, edgecolor='black', linewidth=0.5)
ax2.set_title('Distribution asset_class (vert=standard, rouge=variante)',
              fontweight='bold')
ax2.set_xlabel('Nombre de lignes')

# --- Graphique 3 : Scatter bid vs ask (inversions en evidence) ---
ax3 = axes[1, 0]
normal = df[df['bid'] <= df['ask']]
inverted = df[df['bid'] > df['ask']]
ax3.scatter(normal['bid'], normal['ask'], alpha=0.15, s=8,
            color='#3498db', label=f'Normal ({len(normal)})')
ax3.scatter(inverted['bid'], inverted['ask'], alpha=0.8, s=20,
            color='#e74c3c', marker='x', label=f'Inversee ({len(inverted)})')
max_val = max(df['bid'].max(), df['ask'].max())
ax3.plot([0, max_val], [0, max_val], 'k--', alpha=0.3, label='bid = ask')
ax3.set_title('Bid vs Ask (X rouge = fourchette inversee)', fontweight='bold')
ax3.set_xlabel('Bid')
ax3.set_ylabel('Ask')
ax3.legend()

# --- Graphique 4 : Distribution du delai settlement - trade_date ---
ax4 = axes[1, 1]
td = pd.to_datetime(df['trade_date'], errors='coerce')
sd = pd.to_datetime(df['settlement_date'], errors='coerce')
delai = (sd - td).dt.days
delai_clean = delai.dropna()
colors_hist = ['#e74c3c' if x < 0 else '#2ecc71' for x in sorted(delai_clean.unique())]
ax4.hist(delai_clean, bins=range(int(delai_clean.min()) - 1, int(delai_clean.max()) + 2),
         color='#3498db', edgecolor='black', linewidth=0.5, alpha=0.7)
ax4.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Seuil (delai=0)')
ax4.axvline(x=2, color='green', linestyle='--', linewidth=2, label='T+2 attendu')
negative_count = (delai_clean < 0).sum()
ax4.set_title(f'Delai reglement - trade (jours) | {negative_count} negatifs',
              fontweight='bold')
ax4.set_xlabel('Jours')
ax4.set_ylabel('Frequence')
ax4.legend()

plt.tight_layout()
plt.savefig('01_profiling_report.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nGraphiques sauvegardes dans 01_profiling_report.png")
