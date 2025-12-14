from pymongo import MongoClient
from collections import defaultdict
import csv

# === 1) Connexion à MongoDB (serveur de l'école) ===
MONGO_URI = "mongodb://tncy:etjustelaunpetitgeranium!@mongodb.telecomnancy.univ-lorraine.fr:27017/?authSource=admin"
client = MongoClient(MONGO_URI)

db = client["protein_db_kenza"]
collection = db["proteins_kenza"]

print("Nombre total de documents :", collection.count_documents({}))

# === 2) Charger les protéines avec au moins 1 domaine InterPro ===
proteins = []
cursor = collection.find(
    {"interpro_domains": {"$ne": []}},  # au moins un domaine
    {
        "entry": 1,
        "entry_name": 1,
        "interpro_domains": 1,
        "ec_numbers": 1,
        "go_terms": 1,      # ✅ AJOUT
        "_id": 0,
    },
)

for doc in cursor:
    entry = doc.get("entry")
    if not entry:
        continue

    entry_name = doc.get("entry_name", "")
    domains = set(doc.get("interpro_domains", []) or [])
    ec_numbers = doc.get("ec_numbers", []) or []
    go_terms = doc.get("go_terms", []) or []   # ✅ AJOUT

    if not domains:
        continue

    proteins.append({
        "entry": entry,
        "entry_name": entry_name,
        "domains": domains,
        "ec_numbers": ec_numbers,
        "go_terms": go_terms,   # ✅ AJOUT
    })

print(f"Protéines avec au moins 1 domaine InterPro : {len(proteins)}")

# Map: entry -> set(domains)
entry_to_domains = {p["entry"]: p["domains"] for p in proteins}

# === 3) Construire domain -> liste de protéines ===
domain_to_entries = defaultdict(list)
for p in proteins:
    for d in p["domains"]:
        domain_to_entries[d].append(p["entry"])

print("Domaines InterPro distincts :", len(domain_to_entries))

# === 4) Construire les paires qui partagent au moins 1 domaine ===
pair_to_intersection = defaultdict(int)

for d, entries in domain_to_entries.items():
    if len(entries) < 2:
        continue
    entries = list(entries)
    for i in range(len(entries)):
        for j in range(i + 1, len(entries)):
            e1 = entries[i]
            e2 = entries[j]
            if e1 == e2:
                continue
            key = tuple(sorted((e1, e2)))
            pair_to_intersection[key] += 1

print("Paires avec ≥1 domaine commun :", len(pair_to_intersection))

# === 5) Calcul de la similarité de Jaccard pour chaque paire ===
edges = []
THRESHOLD = 0.3  # on ne garde que les liens avec une similarité suffisante

for (e1, e2), inter_count in pair_to_intersection.items():
    dom1 = entry_to_domains[e1]
    dom2 = entry_to_domains[e2]
    union_size = len(dom1) + len(dom2) - inter_count
    if union_size == 0:
        continue
    jaccard = inter_count / union_size
    if jaccard >= THRESHOLD:
        edges.append({
            "source": e1,
            "target": e2,
            "weight": jaccard,
        })

print(f"Arêtes gardées (Jaccard ≥ {THRESHOLD}): {len(edges)}")

# === 6) Export des nœuds (labelled = EC OU GO) ===
with open("nodes.csv", "w", newline="", encoding="utf-8") as f_nodes:
    writer = csv.writer(f_nodes)
    writer.writerow(["id", "entry_name", "is_labelled"])

    for p in proteins:
        # ✅ Correction : labelled si EC non vide OU GO non vide
        is_labelled = 1 if (len(p["ec_numbers"]) > 0 or len(p["go_terms"]) > 0) else 0
        writer.writerow([p["entry"], p["entry_name"], is_labelled])

print("nodes.csv écrit (is_labelled = EC ou GO).")

# === 7) Export des arêtes ===
with open("edges.csv", "w", newline="", encoding="utf-8") as f_edges:
    writer = csv.writer(f_edges)
    writer.writerow(["source", "target", "weight"])
    for e in edges:
        writer.writerow([e["source"], e["target"], e["weight"]])

print("edges.csv écrit.")
