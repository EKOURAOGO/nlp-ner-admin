"""
generate_training_data.py
--------------------------
Génère un dataset d'entraînement annoté pour la NER (Named Entity Recognition)
sur des textes administratifs français.

Entités extraites :
  - MONTANT      : montants financiers (€, euros, budget)
  - ORGANISME    : organismes publics, ministères, agences
  - DISPOSITIF   : dispositifs d'aide, programmes, contrats
  - DUREE        : durées (mois, ans, semaines)
  - ZONE_GEO     : zones géographiques, territoires
"""

import json
import random
from pathlib import Path
from typing import List, Tuple, Dict, Any

random.seed(42)

# ── Gabarits d'entités ──────────────────────────────────────────────────────

MONTANTS = [
    "1 500 euros", "3 000 euros", "5 000 euros", "10 000 euros", "25 000 euros",
    "50 000 euros", "100 000 euros", "500 000 euros", "1,5 million d euros",
    "2 millions d euros", "150 euros par mois", "300 euros mensuels",
    "un budget de 80 000 euros", "une aide de 2 500 euros",
    "une enveloppe de 750 000 euros",
]

ORGANISMES = [
    "la DREES", "l INSEE", "Pôle emploi", "la CAF", "la MSA",
    "le ministère du Travail", "le ministère de la Santé",
    "l Agence nationale de cohésion des territoires",
    "la Caisse nationale d assurance maladie", "l ANAH",
    "le Conseil départemental", "la Région Ile-de-France",
    "l Agence régionale de santé", "la DREETS",
    "France Travail", "la Sécurité sociale",
]

DISPOSITIFS = [
    "le RSA", "l APL", "la prime d activité", "le contrat d engagement jeune",
    "le plan pauvreté", "le service civique", "l ARE",
    "le dispositif Action Logement", "la garantie jeunes",
    "le contrat aidé", "le Pass Culture", "le MaPrimeRénov",
    "le plan de relance", "le CEJ", "la prestation de compensation du handicap",
]

DUREES = [
    "6 mois", "12 mois", "18 mois", "24 mois", "3 ans",
    "deux ans", "un an", "six semaines", "trois mois",
    "une période de 18 mois", "sur 24 mois", "pendant 6 mois",
]

ZONES_GEO = [
    "en Ile-de-France", "en Nouvelle-Aquitaine", "dans le département du Nord",
    "en zone rurale", "dans les quartiers prioritaires",
    "dans les zones de revitalisation rurale", "en Seine-Saint-Denis",
    "dans les territoires ultramarins", "en Occitanie",
    "dans le département de la Gironde", "en zone périurbaine",
    "dans les QPV", "dans les zones blanches",
]

# ── Gabarits de phrases avec placeholders ──────────────────────────────────

TEMPLATES = [
    "{ORGANISME} verse {MONTANT} aux bénéficiaires de {DISPOSITIF} {ZONE_GEO}.",
    "Le bénéficiaire perçoit {MONTANT} dans le cadre de {DISPOSITIF} pendant {DUREE}.",
    "{ORGANISME} accompagne {ZONE_GEO} avec une enveloppe de {MONTANT} pour {DISPOSITIF}.",
    "L aide de {MONTANT} accordée par {ORGANISME} concerne {DISPOSITIF} sur {DUREE}.",
    "Les habitants {ZONE_GEO} peuvent bénéficier de {DISPOSITIF} versé par {ORGANISME}.",
    "{DISPOSITIF} permet d obtenir {MONTANT} auprès de {ORGANISME} pendant {DUREE}.",
    "{ORGANISME} déploie {DISPOSITIF} {ZONE_GEO} avec un budget de {MONTANT} sur {DUREE}.",
    "Une allocation de {MONTANT} est versée par {ORGANISME} dans le cadre de {DISPOSITIF}.",
    "{ZONE_GEO}, {ORGANISME} finance {DISPOSITIF} à hauteur de {MONTANT} pendant {DUREE}.",
    "Le montant de {MONTANT} alloué à {DISPOSITIF} par {ORGANISME} couvre {DUREE}.",
    "L accompagnement de {ORGANISME} via {DISPOSITIF} dure {DUREE} {ZONE_GEO}.",
    "Grâce à {DISPOSITIF}, {ORGANISME} verse {MONTANT} sur une durée de {DUREE}.",
]


def find_span(text: str, entity: str) -> Tuple[int, int]:
    """Trouve la position (start, end) d'une entité dans un texte."""
    idx = text.find(entity)
    if idx == -1:
        return (-1, -1)
    return (idx, idx + len(entity))


def generate_example(template: str) -> Dict[str, Any]:
    """
    Génère un exemple annoté à partir d'un gabarit.

    Returns:
        {"text": str, "entities": [(start, end, label), ...]}
    """
    montant = random.choice(MONTANTS)
    organisme = random.choice(ORGANISMES)
    dispositif = random.choice(DISPOSITIFS)
    duree = random.choice(DUREES)
    zone = random.choice(ZONES_GEO)

    text = (template
            .replace("{MONTANT}", montant)
            .replace("{ORGANISME}", organisme)
            .replace("{DISPOSITIF}", dispositif)
            .replace("{DUREE}", duree)
            .replace("{ZONE_GEO}", zone))

    entity_map = {
        montant: "MONTANT",
        organisme: "ORGANISME",
        dispositif: "DISPOSITIF",
        duree: "DUREE",
        zone: "ZONE_GEO",
    }

    entities = []
    for ent_text, label in entity_map.items():
        if ent_text in text:
            start, end = find_span(text, ent_text)
            if start != -1:
                entities.append((start, end, label))

    # Trier par position de début
    entities.sort(key=lambda x: x[0])
    return {"text": text, "entities": entities}


def generate_dataset(n_examples: int = 300) -> List[Dict]:
    """Génère n_examples exemples annotés."""
    examples = []
    for _ in range(n_examples):
        template = random.choice(TEMPLATES)
        example = generate_example(template)
        # Garder seulement les exemples avec au moins 2 entités
        if len(example["entities"]) >= 2:
            examples.append(example)
    return examples


if __name__ == "__main__":
    dataset = generate_dataset(n_examples=600)
    out = Path(__file__).parent.parent / "data" / "ner_dataset.json"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    label_counts: Dict[str, int] = {}
    for ex in dataset:
        for _, _, label in ex["entities"]:
            label_counts[label] = label_counts.get(label, 0) + 1

    print(f"Dataset généré : {len(dataset)} exemples annotés")
    print("Distribution des entités :")
    for label, count in sorted(label_counts.items()):
        print(f"  {label}: {count}")
