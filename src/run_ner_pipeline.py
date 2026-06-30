"""
run_ner_pipeline.py
-------------------
Pipeline NER complet : génération -> entraînement -> évaluation -> inférence.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from generate_training_data import generate_dataset
from train_ner import load_dataset, split_dataset, train_ner_model, save_model
from evaluate_ner import evaluate_on_test, print_evaluation_report


def run_inference_examples(nlp, examples: list) -> None:
    """Affiche quelques exemples d'inférence."""
    print("\n── Exemples d'inférence ──────────────────────────────────")
    for text in examples:
        doc = nlp(text)
        print(f"\nTexte : {text}")
        if doc.ents:
            for ent in doc.ents:
                print(f"  [{ent.label_}] → '{ent.text}'")
        else:
            print("  (aucune entité détectée)")


def main() -> None:
    base = Path(__file__).parent.parent

    # ── 1. Génération du dataset ─────────────────────────────────────────────
    corpus_path = base / "data" / "ner_dataset.json"
    if not corpus_path.exists():
        print("Génération du dataset d'entraînement...")
        dataset = generate_dataset(n_examples=600)
        corpus_path.parent.mkdir(exist_ok=True)
        with open(corpus_path, "w", encoding="utf-8") as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)
        print(f"  {len(dataset)} exemples générés.")
    else:
        dataset = load_dataset(str(corpus_path))
        print(f"Dataset chargé : {len(dataset)} exemples.")

    # ── 2. Split train / test ────────────────────────────────────────────────
    train_data, test_data = split_dataset(dataset, train_ratio=0.8)
    print(f"Train: {len(train_data)} | Test: {len(test_data)}")

    # ── 3. Entraînement ──────────────────────────────────────────────────────
    print("\nEntraînement du modèle NER (30 itérations)...")
    nlp, history = train_ner_model(train_data, n_iter=30)

    # ── 4. Sauvegarde du modèle ──────────────────────────────────────────────
    model_path = base / "outputs" / "ner_model"
    save_model(nlp, str(model_path))

    with open(base / "outputs" / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    # ── 5. Évaluation ────────────────────────────────────────────────────────
    metrics = evaluate_on_test(nlp, test_data)
    print_evaluation_report(metrics)

    with open(base / "outputs" / "evaluation_metrics.json", "w") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    # ── 6. Exemples d'inférence ──────────────────────────────────────────────
    inference_examples = [
        "La CAF verse 500 euros aux bénéficiaires du RSA en Ile-de-France pendant 6 mois.",
        "Le ministère du Travail alloue 2 millions d euros au contrat d engagement jeune sur 24 mois.",
        "Pôle emploi accompagne les demandeurs dans les zones rurales grâce à la prime d activité.",
    ]
    run_inference_examples(nlp, inference_examples)

    print("\nPipeline NER terminé avec succès.")
    print(f"Macro F1 : {metrics['macro_avg']['f1']:.4f}")


if __name__ == "__main__":
    main()
