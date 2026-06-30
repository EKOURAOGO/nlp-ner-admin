"""
evaluate_ner.py
---------------
Évaluation du modèle NER sur le jeu de test.
Calcule le F1-score, la précision et le rappel par type d'entité.
"""

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Any

import spacy
from spacy.training import Example


LABELS = ["MONTANT", "ORGANISME", "DISPOSITIF", "DUREE", "ZONE_GEO"]


def load_model(model_path: str) -> spacy.Language:
    """Charge un modèle NER depuis le disque."""
    return spacy.load(model_path)


def compute_entity_metrics(
    predictions: List[List[Tuple[int, int, str]]],
    references: List[List[Tuple[int, int, str]]],
    labels: List[str],
) -> Dict[str, Dict[str, float]]:
    """
    Calcule précision, rappel et F1 par type d'entité.

    Args:
        predictions: Entités prédites [(start, end, label)] par doc.
        references: Entités de référence [(start, end, label)] par doc.
        labels: Liste des labels à évaluer.

    Returns:
        Dict {label: {precision, recall, f1, support}}
    """
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)

    for preds, refs in zip(predictions, references):
        pred_set = set(preds)
        ref_set = set(refs)
        for ent in pred_set:
            if ent in ref_set:
                tp[ent[2]] += 1
            else:
                fp[ent[2]] += 1
        for ent in ref_set:
            if ent not in pred_set:
                fn[ent[2]] += 1

    metrics = {}
    for label in labels:
        p_denom = tp[label] + fp[label]
        r_denom = tp[label] + fn[label]
        precision = tp[label] / p_denom if p_denom > 0 else 0.0
        recall = tp[label] / r_denom if r_denom > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0 else 0.0
        )
        support = tp[label] + fn[label]
        metrics[label] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": support,
        }

    # Macro F1
    macro_f1 = sum(m["f1"] for m in metrics.values()) / len(labels)
    metrics["macro_avg"] = {
        "precision": round(sum(m["precision"] for m in metrics.values()) / len(labels), 4),
        "recall": round(sum(m["recall"] for m in metrics.values()) / len(labels), 4),
        "f1": round(macro_f1, 4),
        "support": sum(m["support"] for m in metrics.values()),
    }

    return metrics


def evaluate_on_test(
    nlp: spacy.Language,
    test_data: List[Dict],
) -> Dict:
    """
    Évalue le modèle sur les données de test.

    Args:
        nlp: Modèle NER chargé.
        test_data: Liste d'exemples de test annotés.

    Returns:
        Dictionnaire de métriques par entité.
    """
    predictions = []
    references = []

    for item in test_data:
        text = item["text"]
        ref_entities = [(s, e, l) for s, e, l in item["entities"]]

        doc = nlp(text)
        pred_entities = [(ent.start_char, ent.end_char, ent.label_) for ent in doc.ents]

        predictions.append(pred_entities)
        references.append(ref_entities)

    return compute_entity_metrics(predictions, references, LABELS)


def print_evaluation_report(metrics: Dict) -> None:
    """Affiche le rapport d'évaluation formaté."""
    print(f"\n{'='*60}")
    print("RAPPORT D'ÉVALUATION NER")
    print(f"{'='*60}")
    print(f"{'Entité':<20} {'Précision':>10} {'Rappel':>10} {'F1':>10} {'Support':>10}")
    print(f"{'-'*60}")
    for label, m in metrics.items():
        if label != "macro_avg":
            print(
                f"{label:<20} {m['precision']:>10.4f} {m['recall']:>10.4f}"
                f" {m['f1']:>10.4f} {m['support']:>10}"
            )
    print(f"{'-'*60}")
    m = metrics["macro_avg"]
    print(
        f"{'MACRO AVG':<20} {m['precision']:>10.4f} {m['recall']:>10.4f}"
        f" {m['f1']:>10.4f} {m['support']:>10}"
    )
    print(f"{'='*60}\n")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from train_ner import load_dataset, split_dataset

    base = Path(__file__).parent.parent
    dataset = load_dataset(str(base / "data" / "ner_dataset.json"))
    _, test_data = split_dataset(dataset)

    nlp = load_model(str(base / "outputs" / "ner_model"))
    metrics = evaluate_on_test(nlp, test_data)

    print_evaluation_report(metrics)

    with open(base / "outputs" / "evaluation_metrics.json", "w") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print("Métriques sauvegardées dans outputs/evaluation_metrics.json")
