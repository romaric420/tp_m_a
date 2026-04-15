# ============================================================
# TRADECLEANSE — NOTEBOOK 02 : Pipeline de Nettoyage Complet
# DCLE821 — QuantAxis Capital
# Etudiant(s) : ___________________________________
# Date        : ___________________________________
# ============================================================
#
# CONTRAINTES OBLIGATOIRES :
#   - Ne jamais modifier tradecleanse_raw.csv
#   - Toujours travailler sur une copie : df = pd.read_csv(...).copy()
#   - Chaque etape doit etre loggee : nb lignes avant / apres / supprimees
#   - Chaque decision doit etre justifiee en commentaire (raison METIER)
#   - Le dataset final doit etre sauvegarde dans : tradecleanse_clean.csv
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
import os
import hashlib
import warnings
from sklearn.ensemble import IsolationForest
warnings.filterwarnings('ignore')

# Configuration du logging (ne pas modifier)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler('tradecleanse_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ============================================================
# CHARGEMENT (ne pas modifier)
# ============================================================
df_raw = pd.read_csv('data/tradecleanse_raw.csv', low_memory=False)
df = df_raw.copy()
logger.info(f"Dataset charge : {df.shape[0]} lignes, {df.shape[1]} colonnes")

# ============================================================
# ETAPE 1 — Remplacement des valeurs sentinelles
# ============================================================
# CONSIGNE :
# Identifiez et remplacez TOUTES les valeurs sentinelles par NaN.
# Une sentinelle est une valeur utilisee a la place d'un NaN reel :
# textuelles (#N/A, N/A, #VALUE!, -, nd, null...) ET numeriques
# (ex: 99999 utilise comme code "donnee manquante" sur country_risk).
#
# ATTENTION : certaines colonnes sont en type "object" a cause des
# sentinelles textuelles melangees a des valeurs numeriques.
# Pensez a gerer le cast des colonnes apres nettoyage.
#
# Loggez le nb de NaN total avant et apres.

# ============================================================
# ETAPE 1 — Remplacement des valeurs sentinelles
# ============================================================
# Raison metier : les 3 sources (Bloomberg, Murex, Refinitiv) utilisent
# des codes differents pour signaler une donnee absente. On les unifie
# en NaN pandas pour garantir un traitement homogene.

before = len(df)
nan_before = df.isnull().sum().sum()

# Sentinelles textuelles presentes dans les colonnes object
text_sentinels = ['#N/A', 'N/A', '#VALUE!', '-', 'nd', 'null',
                  'None', 'na', 'NaN', 'missing', 'n/a', '#NA']
df.replace(text_sentinels, np.nan, inplace=True)

# Sentinelle numerique : 99999 utilise par Refinitiv sur country_risk
df['country_risk'] = df['country_risk'].replace('99999', np.nan)
df['country_risk'] = df['country_risk'].replace(99999, np.nan)
df['country_risk'] = df['country_risk'].replace(99999.0, np.nan)

nan_after = df.isnull().sum().sum()
logger.info(f"[Sentinelles] NaN avant={nan_before}, NaN apres={nan_after} "
            f"(+{nan_after - nan_before} detectes). {before} -> {len(df)} lignes")


# ============================================================
# ETAPE 2 — Suppression des doublons
# ============================================================
# Raison metier : dans Murex, le premier enregistrement correspond a la
# saisie initiale du trade (booking). Les doublons sont des erreurs
# d'extraction. On garde "first" car c'est l'enregistrement original.

before = len(df)
exact_dups = df.duplicated().sum()
tid_dups = df['trade_id'].duplicated().sum()
logger.info(f"[Doublons] Doublons exacts: {exact_dups}, doublons trade_id: {tid_dups}")

df.drop_duplicates(subset='trade_id', keep='first', inplace=True)
df.reset_index(drop=True, inplace=True)
logger.info(f"[Doublons] {before} -> {len(df)} lignes (-{before - len(df)} supprimees)")


# ============================================================
# ETAPE 3 — Conversion et normalisation des types
# ============================================================
# Raison metier : les types corrects sont necessaires pour appliquer
# les regles metier (comparaisons de dates, calculs numeriques, etc.)

before = len(df)
nan_before_conv = df.isnull().sum().sum()

# Dates : conversion avec errors='coerce' pour les formats mixtes
df['trade_date'] = pd.to_datetime(df['trade_date'], errors='coerce')
df['settlement_date'] = pd.to_datetime(df['settlement_date'], errors='coerce')

# Colonnes numeriques qui peuvent contenir des residus textuels
numeric_cols = ['bid', 'ask', 'mid_price', 'price', 'notional_eur',
                'quantity', 'volume_j', 'volatility_30d', 'country_risk']
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Colonnes categorielles : minuscule + strip
cat_cols = ['asset_class', 'credit_rating', 'sector']
for col in cat_cols:
    df[col] = df[col].astype(str).str.strip().str.lower()
    df[col] = df[col].replace('nan', np.nan)

nan_after_conv = df.isnull().sum().sum()
new_nans = nan_after_conv - nan_before_conv
logger.info(f"[Types] {new_nans} nouvelles valeurs NaN par coercion. "
            f"{before} -> {len(df)} lignes")


# ============================================================
# ETAPE 4 — Normalisation du referentiel asset_class
# ============================================================
# Raison metier : les 3 sources nomment differemment les classes
# d'actifs. Il faut un referentiel unique pour les agregations
# de risque par classe d'actif (exigence BCBS 239 principe 2).

before = len(df)
asset_class_mapping = {
    # equity
    'equity': 'equity', 'eq': 'equity', 'equities': 'equity',
    # bond
    'bond': 'bond', 'fixed income': 'bond', 'fi': 'bond',
    # derivative
    'derivative': 'derivative', 'deriv': 'derivative',
    'derivatives': 'derivative', 'opt': 'derivative',
    # fx
    'fx': 'fx', 'foreign exchange': 'fx', 'forex': 'fx',
}

# Normalisation prealable pour le mapping
df['asset_class'] = df['asset_class'].str.lower().str.strip()
df['asset_class'] = df['asset_class'].map(asset_class_mapping)

valid_ac = df['asset_class'].value_counts()
logger.info(f"[asset_class] Apres normalisation : {valid_ac.to_dict()}. "
            f"NaN (non mappees): {df['asset_class'].isnull().sum()}. "
            f"{before} -> {len(df)} lignes")


# ============================================================
# ETAPE 5 — Incoherences structurelles financieres
# ============================================================

before = len(df)

# 5a. settlement_date < trade_date
# Raison metier : regle T+2 (CSDR) — le reglement arrive toujours apres
# le trade. On recalcule settlement = trade_date + 2 jours ouvres.
mask_5a = df['settlement_date'] < df['trade_date']
mask_5a = mask_5a & df['settlement_date'].notna() & df['trade_date'].notna()
n_5a = mask_5a.sum()
df.loc[mask_5a, 'settlement_date'] = (
    df.loc[mask_5a, 'trade_date'] + pd.tseries.offsets.BDay(2)
)
logger.info(f"[5a] settlement < trade : {n_5a} lignes corrigees (recalcul T+2)")

# 5b. bid > ask (fourchette inversee)
# Raison metier : la fourchette bid/ask reflete l'offre et la demande.
# Une inversion est une erreur de saisie Bloomberg. On swap les valeurs.
mask_5b = df['bid'] > df['ask']
mask_5b = mask_5b & df['bid'].notna() & df['ask'].notna()
n_5b = mask_5b.sum()
df.loc[mask_5b, ['bid', 'ask']] = df.loc[mask_5b, ['ask', 'bid']].values
logger.info(f"[5b] bid > ask : {n_5b} lignes corrigees (swap bid/ask)")

# 5c. mid_price incoherent avec (bid + ask) / 2
# Raison metier : le mid_price est le prix de reference pour le mark-to-market.
# On le recalcule systematiquement a partir de bid et ask (source de verite).
mid_theo = (df['bid'] + df['ask']) / 2
mask_5c = (abs(df['mid_price'] - mid_theo) / mid_theo > 0.01)
mask_5c = mask_5c & df['mid_price'].notna() & mid_theo.notna()
n_5c = mask_5c.sum()
df['mid_price'] = mid_theo
logger.info(f"[5c] mid_price recalcule sur toutes les lignes ({n_5c} etaient incoherentes)")

# 5d. price hors de la fourchette [bid * 0.995, ask * 1.005]
# Raison metier : un prix d'execution hors fourchette signale une erreur
# de saisie ou un trade off-market. On remplace par le mid_price.
mask_5d = (df['price'] < df['bid'] * 0.995) | (df['price'] > df['ask'] * 1.005)
mask_5d = mask_5d & df['price'].notna() & df['bid'].notna() & df['ask'].notna()
n_5d = mask_5d.sum()
df.loc[mask_5d, 'price'] = df.loc[mask_5d, 'mid_price']
logger.info(f"[5d] price hors fourchette : {n_5d} lignes corrigees (remplace par mid_price)")

# 5e. notional_eur negatif
# Raison metier : le notionnel represente le montant nominal de la transaction,
# toujours positif. Un signe negatif est une erreur de signe dans l'extraction Murex.
# On prend la valeur absolue pour conserver l'information.
mask_5e = df['notional_eur'] < 0
n_5e = mask_5e.sum()
df.loc[mask_5e, 'notional_eur'] = df.loc[mask_5e, 'notional_eur'].abs()
logger.info(f"[5e] notional negatif : {n_5e} lignes corrigees (valeur absolue)")

# 5f. credit_rating AAA/AA/A avec default_flag = 1
# Raison metier : un emetteur en defaut ne peut conserver un rating
# investment grade (convention S&P/Moody's). Le default_flag provient
# de Refinitiv (source fiable). On retrogade le rating a 'bbb' (premier
# echelon speculative grade) pour maintenir la coherence.
mask_5f = df['credit_rating'].isin(['aaa', 'aa', 'a']) & (df['default_flag'] == 1)
n_5f = mask_5f.sum()
df.loc[mask_5f, 'credit_rating'] = 'bbb'
logger.info(f"[5f] rating investissement + defaut : {n_5f} lignes corrigees "
            f"(rating retrogade a BBB)")

logger.info(f"[Incoherences financieres] {before} -> {len(df)} lignes")


# ============================================================
# ETAPE 6 — Regles metier (valeurs hors plage valide)
# ============================================================
# Raison metier : chaque colonne a une plage de validite definie par
# les normes du metier financier. Les valeurs hors plage sont mises
# a NaN (pas de suppression de ligne) pour etre imputees a l'etape 8.

before = len(df)

# country_risk : score Refinitiv entre 0 et 100
mask_cr = (df['country_risk'] < 0) | (df['country_risk'] > 100)
n_cr = mask_cr.sum()
df.loc[mask_cr, 'country_risk'] = np.nan
logger.info(f"[6] country_risk hors [0,100] : {n_cr} -> NaN")

# volatility_30d : la volatilite historique 30j doit etre dans [0.1, 200]%
mask_vol = (df['volatility_30d'] < 0.1) | (df['volatility_30d'] > 200)
n_vol = mask_vol.sum()
df.loc[mask_vol, 'volatility_30d'] = np.nan
logger.info(f"[6] volatility_30d hors [0.1,200] : {n_vol} -> NaN")

# default_flag : binaire 0 ou 1 uniquement
mask_dflag = ~df['default_flag'].isin([0, 1])
n_dflag = mask_dflag.sum()
df.loc[mask_dflag, 'default_flag'] = np.nan
logger.info(f"[6] default_flag invalide : {n_dflag} -> NaN")

# quantity : doit etre strictement positive
mask_qty = df['quantity'] <= 0
n_qty = mask_qty.sum()
df.loc[mask_qty, 'quantity'] = np.nan
logger.info(f"[6] quantity <= 0 : {n_qty} -> NaN")

logger.info(f"[Regles metier] {before} -> {len(df)} lignes")


# ============================================================
# ETAPE 7 — Detection et traitement des outliers
# ============================================================

before = len(df)

# --- 7.1 Methode IQR ---
# Strategie : winsorisation (capping aux bornes IQR).
# Raison metier : en finance, les valeurs extremes sont frequentes
# (gros trades, pics de volatilite). La suppression perdrait de
# l'information pertinente. La winsorisation conserve les lignes
# en ramenent les extremes aux bornes de la distribution.

iqr_cols = ['notional_eur', 'volatility_30d', 'volume_j']

fig_box, axes_box = plt.subplots(1, 3, figsize=(15, 5))
fig_box.suptitle('Boxplots avant winsorisation (IQR)', fontweight='bold')

for i, col in enumerate(iqr_cols):
    series = df[col].dropna()
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = ((series < lower) | (series > upper)).sum()
    logger.info(f"[7-IQR] {col}: Q1={Q1:.2f}, Q3={Q3:.2f}, IQR={IQR:.2f}, "
                f"bornes=[{lower:.2f}, {upper:.2f}], outliers={outliers}")

    axes_box[i].boxplot(series, vert=True)
    axes_box[i].set_title(f'{col}\n({outliers} outliers)')

    # Winsorisation
    df.loc[df[col] < lower, col] = lower
    df.loc[df[col] > upper, col] = upper

plt.tight_layout()
plt.savefig('02_boxplots_outliers.png', dpi=150, bbox_inches='tight')
plt.close()

# --- 7.2 Isolation Forest (anomalies multivariees) ---
# Raison metier : certaines combinaisons de valeurs sont suspectes
# meme si chaque valeur individuelle est dans la plage valide.
# Le Risk Officer doit examiner ces lignes (pas de suppression).

iso_cols = ['price', 'volume_j', 'volatility_30d', 'notional_eur']
df_iso = df[iso_cols].copy()
df_iso = df_iso.fillna(df_iso.median())

iso_forest = IsolationForest(
    n_estimators=200, contamination=0.03, random_state=42
)
preds = iso_forest.fit_predict(df_iso)
df['is_anomaly_multivariate'] = (preds == -1).astype(int)

n_anomalies = df['is_anomaly_multivariate'].sum()
logger.info(f"[7-IsoForest] {n_anomalies} anomalies multivariees detectees "
            f"(flaggees, non supprimees)")
logger.info(f"[Outliers] {before} -> {len(df)} lignes")


# ============================================================
# ETAPE 8 — Traitement des valeurs manquantes
# ============================================================
# Strategie generale :
#   < 20% NaN  : imputer (mediane pour num, mode pour cat) + flag
#   20-70% NaN : imputer + flag
#   > 70% NaN  : supprimer la colonne

before = len(df)

# Cas particulier : trade_id NaN -> on supprime la ligne
# Raison metier : sans identifiant, la transaction est inutilisable
# pour l'agregation de risque.
n_tid_nan = df['trade_id'].isnull().sum()
if n_tid_nan > 0:
    df = df.dropna(subset=['trade_id'])
    logger.info(f"[8] trade_id NaN : {n_tid_nan} lignes supprimees")

# Analyse du taux de NaN par colonne
nan_pct = (df.isnull().sum() / len(df) * 100)
logger.info(f"[8] Taux de NaN par colonne avant imputation :")
for col in df.columns:
    pct = nan_pct[col]
    if pct > 0:
        logger.info(f"     {col}: {pct:.2f}%")

# Suppression des colonnes > 70% NaN
cols_to_drop = nan_pct[nan_pct > 70].index.tolist()
if cols_to_drop:
    df.drop(columns=cols_to_drop, inplace=True)
    logger.info(f"[8] Colonnes supprimees (>70% NaN) : {cols_to_drop}")

# Cas particulier : settlement_date NaT -> trade_date + 2 jours ouvres
# Raison metier : la regle T+2 (CSDR) s'applique par defaut pour les
# actions europeennes. C'est la meilleure estimation raisonnable.
mask_settle_nan = df['settlement_date'].isnull() & df['trade_date'].notna()
n_settle = mask_settle_nan.sum()
if n_settle > 0:
    df['settlement_date_was_missing'] = df['settlement_date'].isnull().astype(int)
    df.loc[mask_settle_nan, 'settlement_date'] = (
        df.loc[mask_settle_nan, 'trade_date'] + pd.tseries.offsets.BDay(2)
    )
    logger.info(f"[8] settlement_date NaT : {n_settle} imputees (trade_date + T+2)")

# Cas particulier : credit_rating NaN
# Raison metier : imputer le mode global n'est pas pertinent car le rating
# depend fortement du secteur et de la contrepartie. On impute le mode
# par secteur pour plus de precision.
if df['credit_rating'].isnull().sum() > 0:
    df['credit_rating_was_missing'] = df['credit_rating'].isnull().astype(int)
    mode_by_sector = df.groupby('sector')['credit_rating'].transform(
        lambda x: x.mode()[0] if len(x.mode()) > 0 else np.nan
    )
    df['credit_rating'] = df['credit_rating'].fillna(mode_by_sector)
    remaining = df['credit_rating'].isnull().sum()
    if remaining > 0:
        global_mode = df['credit_rating'].mode()[0]
        df['credit_rating'] = df['credit_rating'].fillna(global_mode)
    logger.info(f"[8] credit_rating : impute par mode sectoriel + flag")

# Imputation generique des colonnes restantes
numeric_cols_final = df.select_dtypes(include=[np.number]).columns
cat_cols_final = df.select_dtypes(include='object').columns

for col in df.columns:
    n_nan = df[col].isnull().sum()
    if n_nan == 0:
        continue
    pct = n_nan / len(df) * 100
    flag_col = f"{col}_was_missing"
    if flag_col not in df.columns:
        df[flag_col] = df[col].isnull().astype(int)

    if col in numeric_cols_final:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)
        logger.info(f"[8] {col} ({pct:.1f}% NaN) : impute par mediane ({median_val:.2f}) + flag")
    elif col in cat_cols_final:
        mode_val = df[col].mode()[0]
        df[col] = df[col].fillna(mode_val)
        logger.info(f"[8] {col} ({pct:.1f}% NaN) : impute par mode ('{mode_val}') + flag")
    else:
        # Colonnes datetime
        df[col] = df[col].fillna(method='ffill')
        logger.info(f"[8] {col} ({pct:.1f}% NaN) : impute par forward fill + flag")

logger.info(f"[Valeurs manquantes] {before} -> {len(df)} lignes")


# ============================================================
# ETAPE 9 — Pseudonymisation RGPD / BCBS 239
# ============================================================
# Colonnes PII identifiees :
#   - counterparty_name : nom de personne/entreprise (Art. 4(1) RGPD)
#   - trader_id : identifiant personnel du trader (Art. 4(1) RGPD)
#   - counterparty_id : identifiant indirect de personne morale (Art. 4(1) RGPD)
# Ces donnees permettent l'identification directe ou indirecte d'une
# personne physique, d'ou l'obligation de pseudonymisation (Art. 25 RGPD).

before = len(df)
salt = os.environ.get('CLEANSE_SALT', 'default_salt_dev')

def hash_sha256(value, salt):
    """Hash SHA-256 irreversible avec salt."""
    if pd.isnull(value):
        return np.nan
    return hashlib.sha256(f"{salt}{value}".encode('utf-8')).hexdigest()

pii_columns = ['counterparty_name', 'trader_id', 'counterparty_id']

for col in pii_columns:
    if col in df.columns:
        df[f'{col}_hash'] = df[col].apply(lambda x: hash_sha256(x, salt))
        df.drop(columns=[col], inplace=True)
        logger.info(f"[9] {col} -> {col}_hash (SHA-256 + salt)")

logger.info(f"[Pseudonymisation RGPD] {before} -> {len(df)} lignes. "
            f"Colonnes PII supprimees : {pii_columns}")


# ============================================================
# ETAPE 10 — Rapport de qualite final
# ============================================================

# Statistiques du dataset brut (avant nettoyage)
raw_rows, raw_cols = df_raw.shape
raw_nan_pct = df_raw.isnull().sum().sum() / (raw_rows * raw_cols) * 100
raw_dups = df_raw['trade_id'].duplicated().sum()

# Statistiques du dataset nettoye
clean_rows, clean_cols = df.shape
clean_nan_pct = df.isnull().sum().sum() / (clean_rows * clean_cols) * 100
clean_dups = 0
if 'trade_id' in df.columns:
    clean_dups = df['trade_id'].duplicated().sum()

# Data Quality Score
completude = 1 - (df.isnull().sum().sum() / (clean_rows * clean_cols))
unicite = 1 - (clean_dups / clean_rows) if clean_rows > 0 else 0
DQS = (completude * 0.6 + unicite * 0.4) * 100

print("\n" + "=" * 60)
print("RAPPORT DE QUALITE — AVANT / APRES")
print("=" * 60)
print(f"{'Metrique':<35} {'AVANT':>12} {'APRES':>12}")
print("-" * 60)
print(f"{'Nb lignes':<35} {raw_rows:>12} {clean_rows:>12}")
print(f"{'Nb colonnes':<35} {raw_cols:>12} {clean_cols:>12}")
print(f"{'Taux NaN global (%)':<35} {raw_nan_pct:>11.2f}% {clean_nan_pct:>11.2f}%")
print(f"{'Doublons trade_id':<35} {raw_dups:>12} {clean_dups:>12}")
print("-" * 60)
print(f"{'Completude':<35} {completude:>12.4f}")
print(f"{'Unicite':<35} {unicite:>12.4f}")
print(f"{'DATA QUALITY SCORE (DQS)':<35} {DQS:>11.2f}%")
print("=" * 60)

logger.info(f"[RAPPORT] DQS = {DQS:.2f}% | Completude = {completude:.4f} "
            f"| Unicite = {unicite:.4f}")

# Sauvegarde
df.to_csv('data/tradecleanse_clean.csv', index=False)
logger.info(f"Dataset nettoye sauvegarde : tradecleanse_clean.csv")
print(f"\nShape finale : {df.shape}")
