# Prévision des ventes avec CatBoost et enrichissement de données (Upgini)

Prévoir les ventes quotidiennes de 50 articles dans 10 magasins, et mesurer ce que l'ajout de **données externes** apporte à un modèle entraîné sur un **historique très court**.

![Comparaison du SMAPE](smape_comparison.png)

## Objectif

En entreprise, on doit souvent prévoir avec peu d'historique : nouveau produit, nouveau magasin, données récentes. La question de ce projet est simple : **quand les données internes manquent, des variables externes (calendrier, jours fériés, indicateurs économiques…) peuvent-elles améliorer la prévision ?**

## Données

- Source : [Kaggle - Store Item Demand Forecasting Challenge](https://www.kaggle.com/competitions/demand-forecasting-kernels-only)
- 913 000 lignes : ventes quotidiennes par magasin et par article, de 2013 à 2017
- Colonnes : `date`, `store`, `item`, `sales`

Le fichier `train.csv` n'est pas inclus dans ce dépôt : il faut le télécharger sur Kaggle et le placer à la racine du projet.

## Méthode

1. **Séparation temporelle** : entraînement sur les données antérieures à 2017, test sur toute l'année 2017. Le modèle ne voit jamais le futur.
2. **Historique court volontaire** : l'entraînement n'utilise que les 5 000 premières observations, soit environ 10 jours de janvier 2013.
3. **Modèle de référence** : CatBoostRegressor avec le magasin et l'article en variables catégorielles, et des variables tirées de la date (jour de la semaine, mois, jour de l'année).
4. **Enrichissement** : recherche automatique de variables externes liées à la date avec [Upgini](https://github.com/upgini/upgini), en validation croisée temporelle.
5. **Évaluation** : même modèle, mêmes paramètres, comparé avec le **SMAPE** (erreur moyenne en pourcentage, plus bas = meilleur).

## Résultats

| Modèle | SMAPE (test 2017) |
|---|---|
| CatBoost sans enrichissement | [X] % |
| CatBoost avec Upgini | [Y] % |

**Lecture** : [à compléter après exécution, par exemple « l'enrichissement réduit l'erreur de X à Y %, car les variables externes apportent l'information de saisonnalité absente de 10 jours d'historique »].

## Limites et pistes d'amélioration

- L'historique d'entraînement est volontairement très court : avec 4 ans de données, l'écart entre les deux modèles serait probablement plus faible.
- Ajouter des variables de retard (ventes de la semaine précédente, moyennes mobiles) et comparer avec un modèle de séries temporelles classique (SARIMA).
- Tester d'autres modèles de gradient boosting (XGBoost, LightGBM).

## Lancer le projet

```bash
pip install -r requirements.txt
python sales_forecasting.py
```

La recherche Upgini nécessite une connexion internet.

## Outils

Python, pandas, CatBoost, Upgini, Matplotlib

Projet inspiré de l'exemple de démonstration d'Upgini sur ce jeu de données.
