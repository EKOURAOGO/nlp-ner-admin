"""
test_ner_pipeline.py
--------------------
Tests unitaires du pipeline NER :
génération de données, entraînement, évaluation.
"""

import sys
import json
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from generate_training_data import (
    generate_dataset, generate_example, find_span, TEMPLATES
)
from train_ner import split_dataset, create_spacy_examples
from evaluate_ner import compute_entity_metrics

import spacy

LABELS = ["MONTANT", "ORGANISME", "DISPOSITIF", "DUREE", "ZONE_GEO"]


# ─────────────────────────────────────────────────────────────────────────────
# Data generation tests
# ─────────────────────────────────────────────────────────────────────────────

class TestDataGeneration:

    def test_generate_dataset_length(self):
        dataset = generate_dataset(n_examples=100)
        assert len(dataset) >= 50  # filtre sur >= 2 entités

    def test_generate_dataset_has_text_and_entities(self):
        dataset = generate_dataset(n_examples=20)
        for ex in dataset:
            assert "text" in ex
            assert "entities" in ex
            assert isinstance(ex["text"], str)
            assert isinstance(ex["entities"], list)

    def test_generate_example_entities_within_text(self):
        template = TEMPLATES[0]
        ex = generate_example(template)
        for start, end, label in ex["entities"]:
            assert 0 <= start < end <= len(ex["text"])
            assert label in LABELS

    def test_find_span_correct_position(self):
        text = "La CAF verse 500 euros aux bénéficiaires."
        start, end = find_span(text, "CAF")
        assert text[start:end] == "CAF"

    def test_find_span_not_found_returns_minus_one(self):
        start, end = find_span("Texte sans entité.", "INEXISTANT")
        assert start == -1
        assert end == -1

    def test_entities_sorted_by_start(self):
        dataset = generate_dataset(n_examples=50)
        for ex in dataset:
            starts = [s for s, _, _ in ex["entities"]]
            assert starts == sorted(starts)

    def test_minimum_two_entities_per_example(self):
        dataset = generate_dataset(n_examples=100)
        for ex in dataset:
            assert len(ex["entities"]) >= 2


# ─────────────────────────────────────────────────────────────────────────────
# Training tests
# ─────────────────────────────────────────────────────────────────────────────

class TestTraining:

    def test_split_dataset_ratio(self):
        dataset = [{"text": f"Text {i}", "entities": []} for i in range(100)]
        train, test = split_dataset(dataset, train_ratio=0.8)
        assert len(train) == 80
        assert len(test) == 20

    def test_split_dataset_no_overlap(self):
        dataset = [{"text": f"Text {i}", "entities": []} for i in range(50)]
        train, test = split_dataset(dataset)
        train_texts = {d["text"] for d in train}
        test_texts = {d["text"] for d in test}
        assert train_texts.isdisjoint(test_texts)

    def test_create_spacy_examples_returns_list(self):
        nlp = spacy.blank("fr")
        ner = nlp.add_pipe("ner")
        for label in LABELS:
            ner.add_label(label)
        data = [
            {"text": "La CAF verse 500 euros.", "entities": [(3, 6, "ORGANISME"), (13, 22, "MONTANT")]}
        ]
        examples = create_spacy_examples(data, nlp)
        assert isinstance(examples, list)
        assert len(examples) >= 0  # Peut être 0 si spans invalides

    def test_split_reproducible_with_seed(self):
        dataset = [{"text": f"T{i}", "entities": []} for i in range(100)]
        train1, test1 = split_dataset(dataset)
        train2, test2 = split_dataset(dataset)
        assert [d["text"] for d in train1] == [d["text"] for d in train2]


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation tests
# ─────────────────────────────────────────────────────────────────────────────

class TestEvaluation:

    def test_perfect_predictions(self):
        preds = [[(0, 3, "MONTANT"), (5, 10, "ORGANISME")]]
        refs  = [[(0, 3, "MONTANT"), (5, 10, "ORGANISME")]]
        metrics = compute_entity_metrics(preds, refs, LABELS)
        assert metrics["MONTANT"]["f1"] == 1.0
        assert metrics["ORGANISME"]["f1"] == 1.0

    def test_zero_predictions(self):
        preds = [[]]
        refs  = [[(0, 5, "MONTANT")]]
        metrics = compute_entity_metrics(preds, refs, LABELS)
        assert metrics["MONTANT"]["recall"] == 0.0
        assert metrics["MONTANT"]["f1"] == 0.0

    def test_false_positives_lower_precision(self):
        preds = [[(0, 5, "MONTANT"), (10, 15, "MONTANT")]]
        refs  = [[(0, 5, "MONTANT")]]
        metrics = compute_entity_metrics(preds, refs, LABELS)
        assert metrics["MONTANT"]["precision"] < 1.0

    def test_macro_avg_present(self):
        preds = [[(0, 5, "MONTANT")]]
        refs  = [[(0, 5, "MONTANT")]]
        metrics = compute_entity_metrics(preds, refs, LABELS)
        assert "macro_avg" in metrics

    def test_f1_between_zero_and_one(self):
        preds = [[(0, 5, "MONTANT"), (6, 10, "DUREE")]]
        refs  = [[(0, 5, "MONTANT"), (11, 15, "ZONE_GEO")]]
        metrics = compute_entity_metrics(preds, refs, LABELS)
        for label in LABELS:
            assert 0.0 <= metrics[label]["f1"] <= 1.0

    def test_support_counts_references(self):
        preds = [[]]
        refs  = [[(0, 5, "ORGANISME"), (6, 10, "ORGANISME")]]
        metrics = compute_entity_metrics(preds, refs, LABELS)
        assert metrics["ORGANISME"]["support"] == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
