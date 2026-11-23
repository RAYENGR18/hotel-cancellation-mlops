# Plateforme MLOps de prédiction des annulations de réservations hôtelières

Projet de fin de module MLOps (session 2026-2027), réalisé en binôme par Rayen Gragba et Chedi Trabelsi.

## Objectif

Prédire, pour chaque nouvelle réservation, la probabilité qu'elle soit annulée. L'hôtel peut ainsi demander un acompte, relancer le client ou ajuster son surbooking. Le cadrage complet est dans [SYNOPSIS.md](SYNOPSIS.md).

## Données

*Hotel Booking Demand* (Antonio, Almeida & Nunes, 2019), licence CC BY 4.0 : 119 390 réservations et 32 colonnes, pour deux hôtels au Portugal entre juillet 2015 et août 2017.

Source : <https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand>

Le découpage est temporel : les arrivées 2015-2016 servent à l'entraînement, celles de 2017 simulent le flux de production.

## Structure du projet

```
configs/     Fichiers de configuration
data/        Données brutes et préparées (suivies par DVC, pas par Git)
models/      Modèles entraînés
notebooks/   Analyses exploratoires
src/         Code source (data, features, models)
tests/       Tests unitaires Pytest
dvc.yaml     Étapes du pipeline de données
```

## Installation

```bash
git clone https://github.com/RAYENGR18/hotel-cancellation-mlops.git
cd hotel-cancellation-mlops
poetry install
```

## Utilisation

```bash
poetry run dvc repro    # relance le pipeline de données
poetry run pytest       # lance les tests
```

## État d'avancement

- [x] Structure du projet et fiche de cadrage
- [x] Dataset suivi par DVC et étape de préparation des données
- [ ] Entraînement et suivi des modèles avec MLflow
- [ ] API FastAPI et image Docker
- [ ] Intégration continue GitHub Actions et rapports CML
- [ ] Orchestration Prefect
- [ ] Détection de dérive et réentraînement automatique
