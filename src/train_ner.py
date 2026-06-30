"""
train_ner.py
------------
Entraînement d'un modèle NER custom avec spaCy (blank fr).
Architecture : modèle blank + composant NER entraîné from scratch.
"""

import json
import random
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Any

import spacy
from spacy.training import Example
from spacy.tokens import DocBin
from spacy.util import minibatch, compounding

warnings.filterwarnings("ignore")

LABELS = ["MONTANT", "ORGANISME", "DISPOSITIF", "DUREE", "ZONE_GEO"]


def load_dataset(path: str) -> List[Dict]:
    """Charge le dataset annoté depuis un fichier JSON."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def split_dataset(
    dataset: List[Dict],
    train_ratio: float = 0.8,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Divise le dataset en train/test de façon stratifiée.

    Args:
        dataset: Liste d'exemples annotés.
        train_ratio: Proportion du train.

    Returns:
        (train_data, test_data)
    """
    random.seed(42)
    data = dataset.copy()
    random.shuffle(data)
    split = int(len(data) * train_ratio)
    return data[:split], data[split:]


def create_spacy_examples(
    data: List[Dict],
    nlp: spacy.Language,
) -> List[Example]:
    """
    Convertit les exemples annotés au format spaCy Example.

    Args:
        data: Liste de {'text': str, 'entities': [(start, end, label)]}
        nlp: Pipeline spaCy.

    Returns:
        Liste de spaCy Example.
    """
    examples = []
    for item in data:
        text = item["text"]
        entities = item["entities"]
        doc = nlp.make_doc(text)
        try:
            example = Example.from_dict(doc, {"entities": entities})
            examples.append(example)
        except Exception:
            # Ignorer les exemples avec spans malformés
            pass
    return examples


def train_ner_model(
    train_data: List[Dict],
    n_iter: int = 30,
    dropout: float = 0.3,
) -> Tuple[spacy.Language, List[Dict]]:
    """
    Entraîne le modèle NER spaCy from scratch.

    Args:
        train_data: Données d'entraînement annotées.
        n_iter: Nombre d'itérations.
        dropout: Taux de dropout.

    Returns:
        (modèle entraîné, historique de loss)
    """
    nlp = spacy.blank("fr")

    # Ajouter le composant NER
    ner = nlp.add_pipe("ner", last=True)
    for label in LABELS:
        ner.add_label(label)

    # Préparer les exemples
    examples = create_spacy_examples(train_data, nlp)

    # Initialiser le modèle
    nlp.initialize(lambda: examples)

    # Entraînement
    history = []
    optimizer = nlp.create_optimizer()

    for iteration in range(n_iter):
        random.shuffle(examples)
        losses = {}
        batches = minibatch(examples, size=compounding(4.0, 32.0, 1.001))
        for batch in batches:
            nlp.update(batch, drop=dropout, losses=losses, sgd=optimizer)

        loss_val = round(float(losses.get("ner", 0.0)), 4)
        history.append({"iter": iteration + 1, "loss": float(loss_val)})

        if (iteration + 1) % 10 == 0:
            print(f"  Iter {iteration+1:3d} | Loss NER: {loss_val:.4f}")

    return nlp, history


def save_model(nlp: spacy.Language, output_path: str) -> None:
    """Sauvegarde le modèle entraîné sur disque."""
    Path(output_path).mkdir(parents=True, exist_ok=True)
    nlp.to_disk(output_path)
    print(f"Modèle sauvegardé dans {output_path}")


if __name__ == "__main__":
    base = Path(__file__).parent.parent
    dataset = load_dataset(str(base / "data" / "ner_dataset.json"))
    train_data, test_data = split_dataset(dataset)

    print(f"Train: {len(train_data)} | Test: {len(test_data)}")
    print("Entraînement NER...")

    nlp, history = train_ner_model(train_data, n_iter=30)

    save_model(nlp, str(base / "outputs" / "ner_model"))

    # Sauvegarder historique
    with open(base / "outputs" / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print("Entraînement terminé.")
