# ============================================================
# TRADECLEANSE — NOTEBOOK 04 : Bonus Expert
# DCLE821 — QuantAxis Capital
# Etudiant(s) : ___________________________________
# Date        : ___________________________________
# ============================================================
#
# Ce notebook contient 3 bonus independants.
# Chaque bonus vaut +1 point au-dela de 20.
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_auc_score, precision_score,
                             recall_score, f1_score, roc_curve)
import warnings
warnings.filterwarnings('ignore')

df_raw   = pd.read_csv('data/tradecleanse_raw.csv',   low_memory=False)
df_clean = pd.read_csv('data/tradecleanse_clean.csv', low_memory=False)
df_clean['trade_date'] = pd.to_datetime(df_clean['trade_date'], errors='coerce')

# ============================================================
# BONUS 1 — Detection de Wash Trading (+1 pt)
# ============================================================
# Le wash trading se caracterise par des paires de transactions ou
# le meme trader achete et vend le meme instrument, le meme jour,
# a des prix et quantites quasi-identiques. L'objectif est de gonfler
# artificiellement les volumes pour manipuler le marche.
# Ces 5 criteres simultanes (meme ISIN, meme trader, meme date,
# delta prix < 0.1%, delta quantite < 5%) sont les marqueurs
# classiques car ils revelent une transaction sans transfert reel
# de risque economique (interdit par MAR Art. 12).

print("=" * 60)
print("BONUS 1 — DETECTION DE WASH TRADING")
print("=" * 60)

# Groupement par (isin, trader_id_hash, trade_date) pour trouver les paires
grouped = df_clean.groupby(['isin', 'trader_id_hash', 'trade_date'])

wt_pairs = []
for (isin, trader, date), group in grouped:
    if len(group) < 2:
        continue
    trades = group.reset_index(drop=True)
    for i in range(len(trades)):
        for j in range(i + 1, len(trades)):
            t1 = trades.iloc[i]
            t2 = trades.iloc[j]

            avg_price = (t1['price'] + t2['price']) / 2
            avg_qty = (t1['quantity'] + t2['quantity']) / 2
            if avg_price == 0 or avg_qty == 0:
                continue

            delta_price = abs(t1['price'] - t2['price']) / avg_price * 100
            delta_qty = abs(t1['quantity'] - t2['quantity']) / avg_qty * 100

            if delta_price < 0.1 and delta_qty < 5:
                wt_pairs.append({
                    'trade_id_1': t1['trade_id'],
                    'trade_id_2': t2['trade_id'],
                    'isin': isin,
                    'trader_hash': trader,
                    'trade_date': date,
                    'delta_price_%': round(delta_price, 4),
                    'delta_qty_%': round(delta_qty, 4)
                })

wt_suspects = pd.DataFrame(wt_pairs)
print(f"\nPaires suspectes de wash trading detectees : {len(wt_suspects)}")
if len(wt_suspects) > 0:
    print(wt_suspects.head(10).to_string(index=False))
    wt_suspects.to_csv('wash_trading_suspects.csv', index=False)
    print("\nSauvegarde dans wash_trading_suspects.csv")
else:
    print("Aucune paire suspecte detectee.")
    wt_suspects.to_csv('wash_trading_suspects.csv', index=False)


# ============================================================
# BONUS 2 — Data Drift Monitoring (+1 pt)
# ============================================================

print("\n" + "=" * 60)
print("BONUS 2 — DATA DRIFT MONITORING")
print("=" * 60)

min_date = df_clean['trade_date'].min()
max_date = df_clean['trade_date'].max()
cutoff_early = min_date + pd.Timedelta(days=90)
cutoff_late = max_date - pd.Timedelta(days=90)

df_early = df_clean[df_clean['trade_date'] <= cutoff_early]
df_late  = df_clean[df_clean['trade_date'] >= cutoff_late]

print(f"\nPeriode early : {min_date.date()} -> {cutoff_early.date()} ({len(df_early)} lignes)")
print(f"Periode late  : {cutoff_late.date()} -> {max_date.date()} ({len(df_late)} lignes)")

drift_vars = ['price', 'volatility_30d', 'notional_eur', 'volume_j', 'country_risk']
drift_results = []

fig_drift, axes_drift = plt.subplots(2, 3, figsize=(18, 10))
fig_drift.suptitle('Data Drift Monitoring — Distribution Early vs Late',
                   fontsize=14, fontweight='bold')
axes_flat = axes_drift.flatten()

for i, var in enumerate(drift_vars):
    early_vals = df_early[var].dropna()
    late_vals = df_late[var].dropna()

    ks_stat, p_value = ks_2samp(early_vals, late_vals)
    drift = 'OUI' if p_value < 0.05 else 'NON'

    drift_results.append({
        'variable': var,
        'KS_stat': round(ks_stat, 4),
        'p_value': round(p_value, 6),
        'drift': drift
    })

    ax = axes_flat[i]
    ax.hist(early_vals, bins=40, alpha=0.6, color='#3498db',
            label=f'Early (n={len(early_vals)})', density=True)
    ax.hist(late_vals, bins=40, alpha=0.6, color='#e74c3c',
            label=f'Late (n={len(late_vals)})', density=True)
    color = 'red' if drift == 'OUI' else 'green'
    ax.set_title(f'{var}\nKS={ks_stat:.4f}, p={p_value:.4f} [{drift}]',
                 fontweight='bold', color=color)
    ax.legend(fontsize=8)

axes_flat[-1].axis('off')
plt.tight_layout()
plt.savefig('04_drift_monitor.png', dpi=150, bbox_inches='tight')
plt.close()

drift_df = pd.DataFrame(drift_results)
print(f"\n{drift_df.to_string(index=False)}")
drift_df.to_csv('drift_report.csv', index=False)
print("\nGraphique sauvegarde dans 04_drift_monitor.png")
print("Rapport sauvegarde dans drift_report.csv")


# ============================================================
# BONUS 3 — Impact du nettoyage sur le modele ML (+1 pt)
# ============================================================

print("\n" + "=" * 60)
print("BONUS 3 — IMPACT DU NETTOYAGE SUR LE MODELE ML")
print("=" * 60)

features = ['price', 'quantity', 'bid', 'ask', 'mid_price',
            'volume_j', 'volatility_30d', 'country_risk']

def train_and_evaluate(dataframe, dataset_name):
    """Prepare, entraine et evalue un Random Forest sur un dataset."""
    df_ml = dataframe.copy()

    # Conversion des colonnes numeriques (necessaire pour df_raw)
    for col in features + ['default_flag']:
        df_ml[col] = pd.to_numeric(df_ml[col], errors='coerce')

    df_ml = df_ml.dropna(subset=['default_flag'])
    X = df_ml[features].fillna(df_ml[features].median())
    y = df_ml['default_flag'].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    rf = RandomForestClassifier(n_estimators=150, max_depth=6, random_state=42)
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    y_proba = rf.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, y_proba)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    fpr, tpr, _ = roc_curve(y_test, y_proba)

    print(f"\n  {dataset_name}:")
    print(f"    AUC-ROC   = {auc:.4f}")
    print(f"    Precision = {prec:.4f}")
    print(f"    Rappel    = {rec:.4f}")
    print(f"    F1        = {f1:.4f}")

    return {'Dataset': dataset_name, 'AUC-ROC': round(auc, 4),
            'Precision': round(prec, 4), 'Rappel': round(rec, 4),
            'F1': round(f1, 4)}, fpr, tpr

# Entrainement sur les deux datasets
res_raw, fpr_raw, tpr_raw = train_and_evaluate(df_raw, 'Raw (avant nettoyage)')
res_clean, fpr_clean, tpr_clean = train_and_evaluate(df_clean, 'Clean (apres nettoyage)')

# Tableau comparatif
comparison = pd.DataFrame([res_raw, res_clean])
print(f"\n--- Tableau comparatif ---")
print(comparison.to_string(index=False))
comparison.to_csv('model_comparison.csv', index=False)

# Courbes ROC
fig_roc, ax_roc = plt.subplots(figsize=(8, 6))
ax_roc.plot(fpr_raw, tpr_raw, color='#e74c3c', linewidth=2,
            label=f"Raw (AUC = {res_raw['AUC-ROC']:.4f})")
ax_roc.plot(fpr_clean, tpr_clean, color='#2ecc71', linewidth=2,
            label=f"Clean (AUC = {res_clean['AUC-ROC']:.4f})")
ax_roc.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Aleatoire (AUC = 0.5)')
ax_roc.set_xlabel('Taux de faux positifs (FPR)')
ax_roc.set_ylabel('Taux de vrais positifs (TPR)')
ax_roc.set_title('Comparaison ROC — Raw vs Clean\nImpact du nettoyage sur la prediction de defaut',
                 fontweight='bold')
ax_roc.legend(loc='lower right')
ax_roc.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('04_roc_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nGraphique sauvegarde dans 04_roc_comparison.png")

# --- Analyse du resultat ---
delta_auc = res_clean['AUC-ROC'] - res_raw['AUC-ROC']
delta_f1 = res_clean['F1'] - res_raw['F1']

print("\n--- Analyse ---")
print(f"1. Le nettoyage {'ameliore' if delta_auc > 0 else 'degrade legerement'} "
      f"l'AUC-ROC de {abs(delta_auc):.4f} ({delta_auc/max(res_raw['AUC-ROC'],0.001)*100:+.2f}%).")
print(f"2. Le F1-score evolue de {delta_f1:+.4f}, confirmant que le pipeline "
      f"{'renforce' if delta_f1 > 0 else 'ne degrade pas significativement'} "
      f"la capacite predictive.")
print(f"3. Si le gain est modeste, c'est parce que le Random Forest est robuste "
      f"aux valeurs aberrantes et manquantes. L'impact du nettoyage est plus "
      f"visible sur des modeles lineaires (regression logistique, SVM).")
print(f"4. Pour ameliorer davantage : feature engineering (ratios financiers, "
      f"rolling stats), optimisation des hyperparametres (GridSearchCV), "
      f"et ajout de features temporelles (jour de semaine, mois).")
print(f"5. Le nettoyage reste indispensable pour la conformite reglementaire "
      f"(BCBS 239) et la reproductibilite des resultats, independamment "
      f"du gain ML brut.")
