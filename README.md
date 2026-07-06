# NLP — NER sur Documents Administratifs (spaCy)

Pipeline complet d'entraînement d'un modèle de **Reconnaissance d'Entités Nommées (NER)** custom sur des textes administratifs français. Entraîné de zéro (`spacy blank("fr")`) sur 480 exemples annotés, évalué par F1-score par type d'entité, avec 17 tests unitaires couvrant la génération de données, l'entraînement et l'évaluation.

---

## Entités extraites

| Entité | Exemples |
|--------|---------|
| `MONTANT` | `500 euros`, `2 millions d euros`, `une aide de 2 500 euros` |
| `ORGANISME` | `la CAF`, `Pôle emploi`, `le ministère du Travail` |
| `DISPOSITIF` | `le RSA`, `le contrat d engagement jeune`, `la prime d activité` |
| `DUREE` | `6 mois`, `24 mois`, `deux ans` |
| `ZONE_GEO` | `en Ile-de-France`, `dans les quartiers prioritaires`, `en zone rurale` |

---

## Structure du projet

```
nlp-ner-admin/
├── src/
│   ├── generate_training_data.py  # Générateur de dataset annoté (600 exemples)
│   ├── train_ner.py               # Entraînement spaCy NER from scratch
│   ├── evaluate_ner.py            # F1 / Précision / Rappel par entité
│   └── run_ner_pipeline.py        # Pipeline principal (point d'entrée)
├── data/
│   └── ner_dataset.json           # Dataset annoté (600 exemples)
├── outputs/
│   ├── ner_model/                 # Modèle entraîné (spaCy format)
│   ├── training_history.json      # Historique de loss
│   └── evaluation_metrics.json    # Métriques par entité
├── tests/
│   └── test_ner_pipeline.py       # 17 tests unitaires
├── requirements.txt
└── README.md
```

---

## Pipeline

```
Dataset annoté (JSON)
  600 exemples · 5 types d'entités
        │
        ▼
   Split 80/20
  Train: 480 | Test: 120
        │
        ▼
  Entraînement spaCy
  blank("fr") + composant NER
  30 itérations · dropout 0.3
        │
        ▼
   Évaluation Test
  F1 · Précision · Rappel par entité
        │
        ▼
  Inférence sur nouveaux textes
```

---

## Résultats

| Entité | Précision | Rappel | F1 | Support |
|--------|-----------|--------|-----|---------|
| MONTANT | 1.00 | 1.00 | 1.00 | 92 |
| ORGANISME | 1.00 | 1.00 | 1.00 | 110 |
| DISPOSITIF | 1.00 | 1.00 | 1.00 | 120 |
| DUREE | 1.00 | 1.00 | 1.00 | 83 |
| ZONE_GEO | 1.00 | 1.00 | 1.00 | 58 |
| **Macro avg** | **1.00** | **1.00** | **1.00** | **463** |

> F1 = 1.0 sur corpus synthétique (vocabulaire contrôlé). Sur un corpus réel de rapports administratifs, les entités seraient plus variées et le modèle nécessiterait des données supplémentaires via few-shot ou annotation manuelle.

---


## Lancer le dashboard Streamlit

```bash
streamlit run app.py
```

Le dashboard s'ouvre sur `http://localhost:8501`

## Installation & lancement

```bash
pip install -r requirements.txt

# Générer les données + entraîner + évaluer
python3 src/run_ner_pipeline.py
```

---

## Tests

```bash
python3 -m pytest tests/ -v
```

Sortie attendue : `17 passed`

---

## Stack technique

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![spaCy](https://img.shields.io/badge/spaCy-NER-09A3D5?style=flat-square)
![pytest](https://img.shields.io/badge/pytest-17%20tests-red?style=flat-square)

---

## Auteur

**Emmanuel KOURAOGO** 
[GitHub](https://github.com/EKOURAOGO) · [Email](mailto:ekouraogo73@gmail.com)
