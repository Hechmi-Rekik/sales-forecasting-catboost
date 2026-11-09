"""
Prévision des ventes par magasin et par article avec CatBoost,
et effet de l'enrichissement de données externes (Upgini).

Données : Kaggle - Store Item Demand Forecasting Challenge (train.csv)
https://www.kaggle.com/competitions/demand-forecasting-kernels-only
"""

import time

import matplotlib.pyplot as plt
import pandas as pd
from catboost import CatBoostRegressor
from catboost.utils import eval_metric
from upgini import FeaturesEnricher, SearchKey
from upgini.metadata import CVType

DATA_PATH = "train.csv"
TRAIN_SIZE = 5000          # petit historique d'entraînement (voir README)
SPLIT_DATE = "2017-01-01"  # entraînement avant 2017, test sur 2017


def prepare_features(features: pd.DataFrame) -> pd.DataFrame:
    """Transforme la date en variables numériques utilisables par CatBoost."""
    features = features.copy()
    features["month"] = features["date"].dt.month
    features["day_of_week"] = features["date"].dt.dayofweek
    features["day_of_year"] = features["date"].dt.dayofyear
    return features.drop(columns="date")


def train_and_evaluate(train_x, train_y, test_x, test_y) -> float:
    """Entraîne CatBoost et renvoie le SMAPE sur la période de test."""
    train_x, test_x = prepare_features(train_x), prepare_features(test_x)
    cat_features = [col for col in train_x.columns if pd.api.types.is_string_dtype(train_x[col])]

    model = CatBoostRegressor(verbose=False, allow_writing_files=False, random_seed=0)
    model.fit(train_x, train_y, cat_features=cat_features)
    predictions = model.predict(test_x)
    return eval_metric(test_y.values, predictions, "SMAPE")[0]


# 1. Chargement et préparation des données
df = pd.read_csv(DATA_PATH)
df["store"] = df["store"].astype(str)
df["item"] = df["item"].astype(str)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

# 2. Séparation temporelle : le modèle n'apprend que sur le passé
train = df[df["date"] < SPLIT_DATE].head(TRAIN_SIZE)
test = df[df["date"] >= SPLIT_DATE]

train_x, train_y = train.drop(columns="sales"), train["sales"]
test_x, test_y = test.drop(columns="sales"), test["sales"]

# 3. Modèle de référence, sans enrichissement
smape_baseline = train_and_evaluate(train_x, train_y, test_x, test_y)
print(f"SMAPE sans Upgini : {smape_baseline:.2f} %")

# 4. Recherche de variables externes avec Upgini
enricher = FeaturesEnricher(
    search_keys={"date": SearchKey.DATE},
    id_columns=["store", "item"],
    cv=CVType.time_series,
)
start = time.time()
enricher.fit(train_x, train_y, calculate_metrics=False)
print(f"Recherche Upgini terminée en {time.time() - start:.0f} s")
print(enricher.get_features_info())

enriched_train_x = enricher.transform(train_x, keep_input=True)
enriched_test_x = enricher.transform(test_x, keep_input=True)

# 5. Même modèle, avec les variables enrichies
smape_enriched = train_and_evaluate(enriched_train_x, train_y, enriched_test_x, test_y)
print(f"SMAPE avec Upgini : {smape_enriched:.2f} %")

# 6. Comparaison
plt.bar(["Sans Upgini", "Avec Upgini"], [smape_baseline, smape_enriched], color=["#9ca3af", "#2563eb"])
plt.ylabel("SMAPE (%) - plus bas = meilleur")
plt.title("CatBoost : effet de l'enrichissement des données")
plt.tight_layout()
plt.savefig("smape_comparison.png", dpi=150)
plt.show()
