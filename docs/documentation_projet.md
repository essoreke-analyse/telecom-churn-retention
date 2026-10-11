# Documentation du projet : Prédiction du churn télécom et campagne de rétention

| | |
|---|---|
| **Auteur** | Essoréké Kiheou |
| **Dépôt** | github.com/essoreke-analyse/telecom-churn-retention |
| **Langage / outils** | Python 3.12, pandas, scikit-learn, matplotlib, pytest, Git |
| **Origine** | Reprise d'un projet académique (Master MBFA, SAS + Python, 2024-2025) |
| **Graine aléatoire** | 42 (tous les résultats sont reproductibles) |

---

## 1. Contexte et objectif

### 1.1 Contexte business
Un opérateur télécom perd des clients qui résilient leur abonnement (le « churn »). Garder un client coûte moins cher qu'en reconquérir un. L'opérateur peut envoyer une offre de rétention (remise, geste commercial), mais :
- il ne sait pas à l'avance qui va partir ;
- chaque offre coûte de l'argent ;
- l'offre ne fonctionne pas à tous les coups.

### 1.2 Question posée
> **À quels clients proposer une offre de rétention pour que la campagne rapporte plus qu'elle ne coûte ?**

### 1.3 Deux livrables
1. **Un modèle de prédiction** : probabilité de résiliation dans les 3 mois pour chaque client.
2. **Une règle de décision chiffrée** : liste des clients à contacter, avec le profit estimé.

### 1.4 Critères de succès
- Retrouver la plupart des départs en ciblant peu de clients (rappel dans le top 20 %, PR-AUC).
- Profit net positif et robuste aux hypothèses.
- Résultats reproductibles et non gonflés (pas de fuite de données).

### 1.5 Ce qui a changé par rapport à la version académique
| Problème trouvé | Correction |
|---|---|
| Accuracy de la régression logistique erronée (85,1 % affiché, 90,0 % réel d'après la matrice de confusion) : erreur de code SAS | Accuracy abandonnée, remplacée par des métriques adaptées au déséquilibre |
| 300 doublons (9,5 %) non détectés | Détectés, conservés, jamais séparés entre apprentissage et test ; test de sensibilité |
| Modèles comparés sur des jeux de test différents, sans graine fixe | Mêmes folds pour tous, graine 42 |
| `age` traité comme une vraie variable continue | Reconnu comme un recodage de `age_group` (5 valeurs), supprimé |
| `age_group` et `charge_amount` traités comme continus | Traités comme catégories (effet non monotone ; `charge_amount` va de 0 à 10, pas 0 à 9) |
| Variables écartées sur un test univarié | Conservées, car elles pèsent une fois l'usage contrôlé |
| Pas de réglage des hyperparamètres | Réglage par validation croisée imbriquée |
| Aucune traduction en décision business | Couche de décision (gain attendu, profit, sensibilité) |

---

## 2. Données

### 2.1 Source
- **Jeu de données** : *Iranian Churn*, UCI Machine Learning Repository (2020). https://doi.org/10.24432/C5JW3Z
- **Contenu** : échantillon tiré aléatoirement de la base d'un opérateur télécom iranien, sur 12 mois.
- **Fichier utilisé** : `Customer Churn.csv` (non versionné dans le dépôt, à télécharger et placer dans `data/raw/`).

### 2.2 Structure
- **3 150 clients** (une ligne par client), **14 colonnes**.
- Toutes les variables explicatives sont **agrégées sur les 9 premiers mois**. La cible `Churn` indique l'état du client **au 12e mois**. Les 3 mois intermédiaires sont la fenêtre de prédiction.
- **495 clients churners (15,71 %)** : la cible est déséquilibrée. Prédire « personne ne part » donnerait déjà 84,29 % de bonnes réponses, d'où le rejet de l'accuracy.

### 2.3 Dictionnaire des variables
| Variable d'origine | Type | Description |
|---|---|---|
| Call  Failure | Quantitative | Nombre d'échecs d'appel |
| Complains | Binaire (0/1) | Le client a déposé une plainte |
| Subscription  Length | Quantitative | Mois d'abonnement (3 à 47) |
| Charge  Amount | Ordinale (0 à 10) | Niveau de montant facturé |
| Seconds of Use | Quantitative | Total de secondes d'appels |
| Frequency of use | Quantitative | Nombre total d'appels |
| Frequency of SMS | Quantitative | Nombre total de SMS |
| Distinct Called Numbers | Quantitative | Nombre de numéros distincts appelés |
| Age Group | Ordinale (1 à 5) | Tranche d'âge |
| Tariff Plan | Binaire (1/2) | 1 = pay as you go, 2 = contrat |
| Status | Binaire (1/2) | 1 = actif, 2 = inactif |
| Age | Quantitative | Âge (en réalité 5 valeurs : 15, 25, 30, 45, 55) |
| Customer Value | Quantitative | Valeur calculée du client (définition et unité non documentées) |
| Churn | Binaire (cible) | 1 = a résilié, 0 = reste |

Note technique : les noms d'origine contiennent des **doubles espaces** (`Call  Failure`). Ils sont normalisés au chargement (`call_failure`).

### 2.4 Contrôles de qualité réalisés
| Contrôle | Résultat |
|---|---|
| Valeurs manquantes | 0 |
| Valeurs autorisées (churn, complains, status, tariff_plan) | Conformes |
| Cohérence appels / secondes | Aucun client avec 0 appel mais des secondes (ni l'inverse) |
| Doublons exacts | **300 lignes (9,5 %)**, surtout des groupes de 3 (117 groupes de 3, 37 paires) |
| Profils identiques avec label différent | 14 profils, surtout des clients sans aucune activité (probablement des clients distincts indiscernables) |
| `age` vs `age_group` | Corrélation de rang = 1,00 : `age` n'apporte rien |
| Valeurs extrêmes | Comportements réels (gros utilisateurs), non supprimés |
| Clients à usage nul | 154, dont 121 inactifs (62 % de churn) |

---

## 3. Analyse exploratoire : ce que disent les données

### 3.1 Méthode : une méthode par type de variable
| Type de variable | Méthode |
|---|---|
| Catégorielle / ordinale vs churn | Tableaux croisés, V de Cramér, test du χ² |
| Quantitative vs churn | Médianes, boxplots, test de Mann-Whitney, AUC univariée |
| Quantitative vs quantitative | Corrélation de Spearman (variables asymétriques) |

Les p-values sont toutes très petites avec 3 150 lignes ; on juge donc la **taille d'effet**.

### 3.2 Résultats clés
- **Plaintes** : 83 % de churn chez ceux qui se plaignent, contre 10,1 % (V de Cramér 0,53). Mais seuls 40 % des churners se sont plaints.
- **Statut** : 47,3 % de churn chez les inactifs, contre 5,3 % (V = 0,50). 75 % des churners sont déjà inactifs.
- **Contrat** : 2,4 % de churn (6 sur 245), contre 16,8 % en pay as you go.
- **Charge Amount** : 23,8 % de churn au niveau 0, 7,0 % au niveau 1, moins de 4 % à partir du niveau 3.
- **Age Group** : effet non monotone (groupe 4 le plus élevé à 20,0 %, groupes 1 et 5 presque nuls).
- **Usage** : les churners consomment environ 3 fois moins (médiane 1 182 contre 3 530 secondes ; 25 contre 63 appels).
- **Pouvoir séparateur seul** (1 − AUC) : `seconds_of_use` 0,79, `customer_value` 0,79, `frequency_of_use` 0,77, `distinct_called_numbers` 0,74, `frequency_of_sms` 0,68. `subscription_length`, `call_failure` et `age` ≈ 0,5 (ne séparent pas seuls).
- **Corrélations** : le bloc usage est très redondant (jusqu'à 0,94 entre secondes et appels).
- **Valeur** : les churners sont **15,7 % des clients mais 4,2 % de la valeur totale** (valeur moyenne 125 contre 536).

### 3.3 Décisions de modélisation tirées de l'EDA
1. Supprimer `age`, encoder `age_group` en catégories.
2. Regrouper `charge_amount` en 0 / 1 / 2 / 3+.
3. Recoder `status` et `tariff_plan` en indicateurs 0/1 clairs.
4. Conserver doublons et valeurs extrêmes, utiliser un découpage groupé.
5. Régulariser les modèles linéaires (variables d'usage très corrélées).
6. Évaluer avec PR-AUC et rappel dans le top k, pas l'accuracy.
7. Évaluer aussi séparément sur les clients encore actifs.

---

## 4. Variables du modèle

12 variables explicatives après transformation (`src/churn/features.py`). Transformations **déterministes** uniquement (aucun apprentissage sur les données, donc aucune fuite).

| Groupe | Variables | Traitement dans le modèle |
|---|---|---|
| Numériques (7) | call_failure, subscription_length, seconds_of_use, frequency_of_use, frequency_of_sms, distinct_called_numbers, customer_value | Passées telles quelles aux arbres ; log + standardisation pour les modèles linéaires |
| Binaires (3) | complains, is_inactive, is_contract | Telles quelles |
| Catégorielles (2) | age_group, charge_band (0 / 1 / 2 / 3+) | Encodage one-hot pour les modèles linéaires |

Supprimée : `age`. Recodées : `status` → `is_inactive`, `tariff_plan` → `is_contract`, `charge_amount` → `charge_band`.

### Deux versions du modèle
`complains` et `is_inactive` sont très prédictives mais **peut-être mesurées trop tard** (le jeu de données ne dit pas quand). On produit donc :

| Version | Variables | À utiliser si |
|---|---|---|
| **Complète** (12 variables) | toutes | plaintes et statut connus au moment du scoring |
| **Alerte précoce** (10 variables) | sans `complains` ni `is_inactive` | doute sur la date de mesure, ou scoring plus tôt |

---

## 5. Modélisation

### 5.1 Protocole d'évaluation (très important)
- **Validation croisée stratifiée et groupée à 5 folds** (`StratifiedGroupKFold`) : même proportion de churners dans chaque fold, et les lignes identiques ne sont jamais séparées entre apprentissage et test.
- **Mêmes folds pour tous les modèles** : la comparaison est équitable.
- **Pipelines scikit-learn** : prétraitement appris dans chaque fold uniquement, donc pas de fuite.
- **Scores hors échantillon** : chaque client est noté par un modèle qui ne l'a pas vu.

### 5.2 Métriques
| Métrique | Sens | Repère |
|---|---|---|
| ROC-AUC | Capacité à classer churners avant non-churners | 0,5 = hasard |
| **PR-AUC** | Idem, centrée sur la classe rare | **0,157 = hasard** |
| Rappel top 20 % | Part des churners trouvés en contactant les 20 % les plus risqués | 20 % = hasard |
| Lift top 20 % | Rappel divisé par 0,2 | 1 = hasard |

### 5.3 Modèles comparés (valeurs hand-picked, avant réglage)
| Modèle | ROC-AUC | PR-AUC | Rappel top 20 % |
|---|---|---|---|
| Régression logistique (baseline, L2) | 0,939 | 0,778 | 0,792 |
| SVM noyau RBF | 0,967 | 0,864 | 0,887 |
| Random Forest | 0,982 | 0,907 | 0,939 |
| Gradient boosting (HistGradientBoosting) | 0,980 | 0,916 | 0,917 |

Les modèles à arbres battent la baseline de plus de 0,12 de PR-AUC, pour un bruit entre folds d'environ 0,03 : la complexité est justifiée (non-linéarités et interactions). Le SVM est dominé et écarté.

### 5.4 Contrôles de robustesse
**a) Les doublons gonflent-ils les scores ?** On compare découpage groupé et découpage naïf. Écart de PR-AUC : −0,001 à +0,003, donc **pas de fuite** mesurable. Le découpage groupé est conservé par principe.

**b) Ablation (sans `complains` ni `is_inactive`)**
| Modèle | PR-AUC complet | PR-AUC sans | Variation |
|---|---|---|---|
| Logistique | 0,778 | 0,563 | −0,215 |
| SVM | 0,864 | 0,762 | −0,103 |
| Random Forest | 0,907 | 0,842 | −0,065 |
| Gradient boosting | 0,916 | 0,836 | −0,080 |

La logistique s'effondre, les modèles à arbres retrouvent l'information dans l'usage.

### 5.5 Réglage des hyperparamètres : validation croisée imbriquée
- Boucle externe 5 folds (évaluation), boucle interne 3 folds (réglage), recherche aléatoire de 12 combinaisons, critère PR-AUC.
- Sans cette imbrication, on règle et on note sur les mêmes données et le score est optimiste.

| Version | Modèle | ROC-AUC | PR-AUC | Écart-type PR-AUC | Rappel top 20 % |
|---|---|---|---|---|---|
| Complète | Random Forest | 0,985 | 0,932 | 0,031 | 0,953 |
| Complète | **Gradient boosting** | 0,987 | **0,947** | 0,028 | 0,951 |
| Alerte précoce | Random Forest | 0,977 | 0,867 | 0,074 | 0,949 |
| Alerte précoce | **Gradient boosting** | 0,978 | **0,906** | 0,039 | 0,927 |

### 5.6 Modèle retenu
**Gradient boosting (`HistGradientBoostingClassifier`)** pour les deux versions : meilleure PR-AUC, plus stable en alerte précoce, meilleur sur les clients encore actifs en alerte précoce (PR-AUC 0,854 contre 0,811). En version complète, il est à égalité statistique avec la Random Forest.
Hyperparamètres du modèle final : learning_rate 0,1 ; max_iter 400 ; max_depth 3 ; min_samples_leaf 10 ; l2_regularization 1,0 (valeurs choisies le plus souvent par la validation imbriquée).

---

## 6. Couche de décision : du score au profit

### 6.1 Principe
Un score n'est pas une décision. On contacte un client si :

> **probabilité de départ × taux de succès de l'offre × valeur du client − coût de l'offre > 0**

### 6.2 Hypothèses (inventées, modifiables dans `config/config.yaml`)
| Paramètre | Valeur | Justification |
|---|---|---|
| Coût d'une offre | 10 | ≈ 10 % de la valeur médiane d'un churner (97) |
| Taux de succès | 30 % | Hypothèse prudente, testée de 10 à 50 % |
| Valeur d'un client sauvé | sa `customer_value` | Seule mesure de valeur disponible |
| Unité | celle de `customer_value` | Non documentée, donc pas appelée « euros » |

### 6.3 Résultats (scores hors échantillon, gradient boosting réglé)
| Politique | Clients contactés | Churners atteints | Profit net |
|---|---|---|---|
| Contacter tout le monde | 100 % | 100 % | **−12 966** |
| Top 19 % par probabilité (k choisi a posteriori, référence non déployable) | 19 % | 94 % | +11 330 |
| **Règle du gain attendu, version complète** | **12 %** | **66 %** | **+13 054** |
| Règle du gain attendu, alerte précoce | 13 % | 63 % | +12 004 |

La règle du gain attendu fait mieux que le classement par probabilité car elle tient aussi compte de la valeur du client.

### 6.4 Robustesse aux hypothèses
Le profit reste positif sur toute la grille (succès 10 à 50 %, coût 5 à 20). Pire cas (succès 10 %, coût 20) : **+863** en version complète (+532 en alerte précoce), mais avec seulement 2 % de clients contactés. Le profit dépend surtout du taux de succès de l'offre.

### 6.5 Calibration
La règle suppose des probabilités fiables. Vérification : dans les déciles à haut risque, 0,49 prédit pour 0,52 observé, et 0,96 pour 0,98. Score de Brier 0,030 (0,132 pour une prédiction constante). La version alerte précoce est un peu moins bien calibrée (Brier 0,039).

---

## 7. Interprétation

### 7.1 Importance des variables (permutation, hors échantillon)
| Variable | Version complète | Alerte précoce |
|---|---|---|
| frequency_of_use | 0,267 | 0,353 |
| complains | 0,204 | (retirée) |
| subscription_length | 0,120 | 0,233 |
| call_failure | 0,115 | 0,268 |
| seconds_of_use | 0,111 | 0,241 |
| is_inactive | 0,089 | (retirée) |
| charge_band | 0,000 | 0,048 |

À retenir :
- **L'usage domine.** `frequency_of_use` est n°1 dans les deux versions.
- **`call_failure` et `subscription_length` ne séparent pas seules, mais pèsent beaucoup une fois l'usage pris en compte.** Les écarter sur un test univarié aurait fait perdre du signal.
- L'importance est partagée entre variables corrélées : à lire en ordre de grandeur. Elle indique « aide à prédire », pas « cause ».

### 7.2 Profil des clients ciblés (règle du gain attendu, version complète)
| | Non ciblés | Ciblés |
|---|---|---|
| Clients | 2 766 | 384 |
| Taux de churn réel | 6,0 % | **85,7 %** |
| Inactifs | 18,5 % | 70,6 % |
| Ayant porté plainte | 2,0 % | 48,7 % |
| Usage médian (secondes) | 3 346 | 1 782 |
| Valeur médiane | 252 | 146 |

La règle cible avant tout un **risque très élevé** ; la valeur départage les cas limites.

---

## 8. Recommandations business
1. **Cibler environ 12 % des clients** avec la règle du gain attendu plutôt que tout le monde.
2. **Traiter les plaintes vite** (83 % de churn), en sachant que 60 % des churners ne se plaignent jamais : action nécessaire mais insuffisante.
3. **Surveiller la baisse d'usage** comme signal d'alerte précoce, observable sans plainte.
4. **Étudier les échecs d'appel avec les équipes réseau** (signal qui pèse une fois l'usage contrôlé).
5. **Tester l'effet des contrats par A/B test** : leur faible churn (2,4 %) ne prouve pas qu'ils causent la fidélité.
6. **Garder les offres peu coûteuses** : les churners ne pèsent que 4,2 % de la valeur.

---

## 9. Limites et risques
| Limite | Conséquence |
|---|---|
| Date de mesure de `status` et `complains` inconnue | Deux versions du modèle ; la version complète peut reconnaître des clients déjà partis |
| Coût, taux de succès et unité de `customer_value` inventés ou inconnus | Les profits montrent une sensibilité, pas une prévision |
| 15,7 % est un taux d'échantillon | Pas forcément le taux réel de l'opérateur |
| Association ≠ causalité | Les recommandations (réseau, contrats) nécessitent des tests |
| Ni dates ni identifiant client | Pas de validation temporelle ; clients repérés par numéro de ligne ; doublons indiscernables |
| Un seul pays et une seule période | Généralisation non démontrée |
| Hyperparamètres en bord de grille | L'optimum est peut-être un peu plus loin |
| 125 churners encore actifs seulement | Métriques sur les clients actifs bruitées |
| Taux de succès supposé identique pour tous | Probablement faux en réalité |
| Modèle final entraîné sur toutes les données | Aucun score hors échantillon ; les performances annoncées sont celles de la validation imbriquée |

---

## 10. Architecture technique

```
config/config.yaml     chemins, graine, hypothèses business
data/raw/              CSV d'origine (non versionné, jamais modifié)
data/processed/        sorties générées (listes de clients scorés)
notebooks/             01_eda, 02_feature_analysis, 03_model_comparison
tools/                 make_notebooks.py (régénère les notebooks 02 et 03)
src/churn/
  data.py              chargement, normalisation des noms, validation
  features.py          variables dérivées, groupes de doublons
  train.py             pipelines, CV stratifiée et groupée, métriques
  compare.py           comparaison des modèles, fuite par doublons, ablation
  tune.py              réglage par CV imbriquée
  evaluate.py          gain attendu, courbe de profit, sensibilité
  explain.py           calibration, importance par permutation, profils
  final.py             entraînement et sauvegarde des modèles finaux
  predict.py           scoring de clients, liste à contacter
tests/                 39 tests pytest
models/                modèles sauvegardés (générés, non versionnés)
reports/figures/       graphiques utilisés par le README et les notebooks
docs/                  documentation
```

### Choix techniques notables
- **Package installable** (`pip install -e .`) : chaque notebook importe le code de `src/`, aucune logique copiée.
- **Tests automatisés** : validation des données, features, pipelines, scoring ; un test vérifie que le code détecte un fichier cassé.
- **Données non versionnées** : `.gitignore` exclut `data/raw`, `models` ; le README indique où télécharger le jeu de données.
- **Historique Git** : un commit par étape logique, adresse e-mail noreply GitHub.

### Reproduire le projet (Windows PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
pytest -q                       # 39 tests
python -m churn.train           # baseline
python -m churn.compare         # modèles, fuite, ablation
python -m churn.tune            # validation imbriquée (plusieurs minutes)
python -m churn.evaluate        # profit et sensibilité
python -m churn.explain         # calibration et importance
python -m churn.final           # entraîne et sauvegarde les modèles
python -m churn.predict         # liste de clients à contacter
```
`python -m churn.predict --version early_warning` utilise la version alerte précoce. `--input fichier.csv` score de nouveaux clients (même schéma que le CSV d'origine).

---

## 11. Pistes d'évolution
1. Obtenir la **date de mesure** de `status` et `complains` pour trancher entre les deux versions.
2. **Mesurer le vrai coût et le vrai taux de succès** par un test A/B de la campagne.
3. **Valider dans le temps** (entraîner sur une période, tester sur la suivante) avec des données datées.
4. Ajouter un **identifiant client** pour distinguer les doublons.
5. Surveiller la **dérive** du modèle (distribution des scores, calibration) après déploiement.
6. Étendre la grille d'hyperparamètres au-delà des bords atteints.
7. Estimer l'**effet causal** de l'offre (modèles d'uplift) pour cibler ceux sur qui l'offre change vraiment la décision, et non seulement ceux qui risquent de partir.

---

## 12. Glossaire
| Terme | Définition simple |
|---|---|
| Churn | Résiliation d'un abonnement |
| Variable cible | Ce qu'on veut prédire (ici : le client part-il ?) |
| Déséquilibre | Une classe est bien plus rare que l'autre (16 % contre 84 %) |
| Validation croisée | Découper les données en folds ; chacun sert tour à tour de test |
| Validation imbriquée | Un réglage dans une boucle interne, une évaluation dans une boucle externe |
| Fuite de données | Information du test qui contamine l'apprentissage, et gonfle les scores |
| PR-AUC | Qualité du classement centrée sur la classe rare (hasard = 0,157) |
| Rappel top 20 % | Part des départs retrouvés en contactant les 20 % les plus risqués |
| Calibration | Les probabilités prédites correspondent-elles aux fréquences observées ? |
| Ablation | Retirer des variables pour mesurer ce qui reste |
| Importance par permutation | Perte de performance quand on mélange une variable au hasard |
| Gain attendu | Profit espéré d'une offre : proba × succès × valeur − coût |