# FICHE DE CADRAGE : SYNOPSIS DU PROJET MLOPS (SESSION 2026–2027)

---

## 1. Informations Générales & Équipe

* **Titre du Projet :** Plateforme MLOps de prédiction des annulations de réservations hôtelières
* **Taille du groupe :** 2 étudiants (binôme)
* **Membres de l'équipe :**
  1. **Étudiant 1 (Lead Data & DVC, Modeling & MLflow, Drift) :** Gragba Rayen — Email : `rayengragba18@gmail.com`
  2. **Étudiant 2 (Lead Serving & CI/CD, Orchestration Prefect) :** Trabelsi Chedi — Email : `cheditrabelsi10@gmail.com`
* **Lien vers le Dépôt GitHub :** <https://github.com/RAYENGR18/hotel-cancellation-mlops>

---

## 2. Problématique Métier & Cas d'Usage

* **Contexte Business :** Hôtellerie et revenue management. Dans les données, 37 % des réservations sont annulées (41,7 % pour l'hôtel de ville, 27,8 % pour le resort). Chaque annulation tardive laisse une chambre vide et fausse les prévisions d'occupation.
* **Décision assistée ou automatisée :** À chaque nouvelle réservation, l'API renvoie une probabilité d'annulation. Le revenue manager s'en sert pour (1) ajuster le niveau de surbooking, (2) exiger un acompte sur les réservations à risque, (3) relancer le client avant la date d'arrivée.
* **Objectif métier quantifié (KPI) :** Détecter au moins 75 % des annulations avec une précision d'au moins 65 %, mesuré sur des réservations postérieures à l'entraînement (année 2017).

---

## 3. Formulation Machine Learning & Choix du Dataset

* **Type de problème ML :** Classification binaire
* **Variable Cible (`Target`) :** `is_canceled` — 1 = réservation annulée ou no-show, 0 = séjour honoré
* **Source du Dataset :** *Hotel Booking Demand* — <https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand>. Article d'origine : Antonio, Almeida & Nunes, *Data in Brief* 22 (2019), <https://doi.org/10.1016/j.dib.2018.11.126>. Licence CC BY 4.0.
* **Volume et dimensions :** 119 390 lignes, 32 colonnes, CSV de 17 Mo. Deux hôtels au Portugal, arrivées de juillet 2015 à août 2017.
* **Variables prédictives clés :**
  * *Variables numériques :* `lead_time`, `adr` (prix moyen par nuit), `stays_in_weekend_nights`, `stays_in_week_nights`, `adults`, `children`, `previous_cancellations`, `days_in_waiting_list`, `total_of_special_requests`
  * *Variables catégorielles :* `hotel`, `meal`, `market_segment`, `distribution_channel`, `deposit_type`, `customer_type`, `reserved_room_type`, `country`
  * *Colonnes exclues (fuite de données) :* `reservation_status` et `reservation_status_date` (connues après l'issue), `assigned_room_type` et `booking_changes` (connues après la réservation)
* **Découpage :** temporel et non aléatoire. Entraînement sur les arrivées 2015–2016 (78 590 lignes après nettoyage), flux de production simulé avec 2017 (40 619 lignes).
* **Métriques ML cibles :** ROC-AUC ≥ 0,85 ; F1-Score ≥ 0,70 ; Rappel ≥ 0,75 ; latence d'inférence < 150 ms

---

## 4. Architecture MLOps Prévisionnelle

| Composant MLOps | Outil & Configuration retenue |
|---|---|
| **Versioning Données** | DVC, pipeline `dvc.yaml` (ingest → preprocess → train → evaluate), remote local puis DagsHub pour la CI |
| **Tracking & Registry** | MLflow Tracking Server local (`http://127.0.0.1:5000`) & Model Registry ; comparaison LogReg / RandomForest / XGBoost |
| **Microservice Inférence** | API REST FastAPI : `/predict` (unitaire et batch), `/health`, Swagger automatique |
| **Validation des données** | Schémas Pydantic stricts pour les payloads d'entrée |
| **Packaging & Isolation** | Image Docker multi-stage non-root (`python:3.11-slim`) & `docker-compose.yml` |
| **Intégration Continue (CI)** | GitHub Actions : Linting (`flake8`, `black`) + Pytest (couverture > 75 %) |
| **Rapports CML** | Publication automatique des métriques, matrice de confusion et courbe ROC en commentaire de PR |
| **Orchestration** | Prefect 3.x (Tasks atomiques avec retries, Flow d'entraînement, Work Pool, planification CRON) |
| **Déploiement Continu** | Blue/Green avec `docker-compose` *(extension, hors périmètre obligatoire du binôme)* |
| **Surveillance de Dérive** | Kolmogorov-Smirnov (numérique), χ² (catégoriel) et PSI, calculés par lot mensuel. Les variables de calendrier sont exclues du suivi. |
| **Export Métriques** | Format OpenMetrics Prometheus (`/metrics`) *(extension)* |
| **Boucle de Rétroaction** | Déclenchement automatique du Flow de réentraînement si `drift_share >= 0.30` |

---

## 5. Scénario de Dérive de Données (Data Drift) Simulé

* **Quelles variables vont dériver en production ?** Dérive réelle déjà présente en 2017 : `adr` passe de 95 € à 115 € en moyenne (PSI = 0,20) et la part du segment « Online TA » dans `market_segment` monte de 43 % à 56 %. Dérive injectée (scénario de crise) : `adr` +25 %, `lead_time` raccourci (réservations de dernière minute), bascule de `market_segment` vers « Online TA » et de `deposit_type` vers « No Deposit ».
* **Méthode d'injection de la dérive :** Module `DriftSimulator` avec deux modes : (1) rejeu des réservations 2017 mois par mois vers l'API ; (2) choc paramétrable : facteur multiplicatif et bruit gaussien sur les variables numériques, réaffectation de catégories avec une probabilité donnée.
* **Seuil d'alerte :** Une variable est en dérive si son PSI ≥ 0,20 (les p-values KS et χ² sont rapportées en complément). Alerte critique si la proportion de variables en dérive atteint 30 %.
* **Comportement autonome du système :** Le rapport de dérive est généré par un Flow Prefect planifié. Si le seuil critique est dépassé, il appelle le Flow de réentraînement sur les données récentes, compare le nouveau modèle au modèle en production dans MLflow et le promeut dans le Model Registry s'il est meilleur.

---

## 6. Planning Prévisionnel & Validation Enseignant

* [ ] **Jalon 1 (S2) :** Dépôt initialisé, Synopsis validé, Dataset importé sous DVC
* [ ] **Jalon 2 (S4) :** Pipeline DVC + MLflow fonctionnels, 1ers tests Pytest
* [ ] **Jalon 3 (S5) :** API FastAPI opérationnelle, Dockerfile validé, Workflow CI actif
* [ ] **Jalon 4 (S6) :** Flow Prefect orchestré, Détection de drift & Prometheus
* [ ] **Jalon Final (S8) :** Tag `v1.0-release`, Rapport PDF et Soutenance

> **Avis de l'Enseignant :** `[  ] VALIDÉ  /  [  ] À REVOIR`  
> **Commentaires :**
