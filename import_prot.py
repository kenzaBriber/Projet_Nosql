import pandas as pd
from pymongo import MongoClient
import math
import requests
import time
from collections import OrderedDict

UNIPROT_URL = "https://rest.uniprot.org/uniprotkb/{}.json"
HEADERS = {"Accept": "application/json"}

def fetch_go_terms(uniprot_id: str) -> list[str]:
    """Retourne la liste des GO:xxxxxxx pour un Uniprot ID via UniProt REST."""
    if not uniprot_id or not isinstance(uniprot_id, str):
        return []

    url = UNIPROT_URL.format(uniprot_id.strip())
    r = requests.get(url, headers=HEADERS, timeout=30)
    if r.status_code != 200:
        return []

    data = r.json()
    gos = set()
    for ref in data.get("uniProtKBCrossReferences", []):
        if ref.get("database") == "GO":
            go_id = ref.get("id")
            if go_id and go_id.startswith("GO:"):
                gos.add(go_id)
    return sorted(gos)



# === 1) Charger le TSV ===
tsv_path = "prot.tsv"  # ton fichier doit être dans le même dossier
df = pd.read_csv(tsv_path, sep="\t")

print("Colonnes détectées :")
print(df.columns)

# === 2) Connexion à MongoDB DISTANT (serveur de l'école) ===
# Très important: on se connecte au serveur mongo de Telecom Nancy
MONGO_URI = "mongodb://tncy:etjustelaunpetitgeranium!@mongodb.telecomnancy.univ-lorraine.fr:27017/?authSource=admin"

client = MongoClient(MONGO_URI)
db = client["protein_db_kenza"]
collection = db["proteins_kenza"]

# On vide la collection si elle existe déjà (pour éviter les doublons)
deleted = collection.delete_many({})
print(f"Collection 'proteins' vidée ({deleted.deleted_count} anciens documents supprimés).")

# === 3) Fonctions utilitaires pour nettoyer les champs ===
def split_interpro(value):
    """
    'IPR001128;IPR017972;IPR002401;IPR008069;IPR036396;'
    -> ['IPR001128','IPR017972',...]
    """
    if isinstance(value, float) and math.isnan(value):
        return []
    s = str(value).strip()
    if not s:
        return []
    parts = [p.strip() for p in s.split(";") if p.strip() != ""]
    return parts

def split_ec(value):
    """
    '1.14.14.1' -> ['1.14.14.1']
    '' ou NaN -> []
    """
    if isinstance(value, float) and math.isnan(value):
        return []
    s = str(value).strip()
    if not s:
        return []
    parts = [p.strip() for p in s.split(";") if p.strip() != ""]
    return parts

# === 4) Construction des documents à insérer ===
docs = []
for _, row in df.iterrows():
    doc = {
        "entry": row.get("Entry"),
        "entry_name": row.get("Entry Name"),
        "protein_names": row.get("Protein names"),
        "organism": row.get("Organism"),
        "sequence": row.get("Sequence"),
        "ec_numbers": split_ec(row.get("EC number")),
        "interpro_domains": split_interpro(row.get("InterPro")),
        "go_terms": [],   # sera rempli après via l'API

    }
    docs.append(doc)

print(f"{len(docs)} documents préparés.")

# === 5) Insertion en bulk dans MongoDB ===
if docs:
    result = collection.insert_many(docs)
    print(f"{len(result.inserted_ids)} documents insérés dans MongoDB (protein_db.proteins).")
else:
    print("Aucun document à insérer.")


# === 7) Enrichissement GO via UniProt API ===
# Bonnes pratiques: cache + pause (throttling) pour éviter de spam l’API

SLEEP_SEC = 0.25       # ajuste 0.2 à 0.5
BATCH_PRINT = 200
MAX_CACHE = 20000      # cache en mémoire (OK pour ~20k)

cache = OrderedDict()  # uniprot_id -> list(go_terms)

updated = 0
processed = 0

cursor = collection.find({}, {"entry": 1, "go_terms": 1})
for doc in cursor:
    processed += 1
    uid = doc.get("entry")

    # skip si déjà rempli
    if doc.get("go_terms"):
        continue

    if uid in cache:
        go_terms = cache[uid]
    else:
        go_terms = fetch_go_terms(uid)
        cache[uid] = go_terms
        if len(cache) > MAX_CACHE:
            cache.popitem(last=False)
        time.sleep(SLEEP_SEC)

    collection.update_one(
        {"_id": doc["_id"]},
        {"$set": {"go_terms": go_terms}}
    )
    updated += 1

    if processed % BATCH_PRINT == 0:
        print(f"GO enrich: processed={processed}, updated={updated}")

print(f"✅ GO enrich terminé: processed={processed}, updated={updated}")

# === 6) Petit check ===
count = collection.count_documents({})
print(f"Vérification: la collection contient maintenant {count} documents.")

