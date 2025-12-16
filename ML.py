# ML.py
from __future__ import annotations

from neo4j import GraphDatabase
import numpy as np


# ======================================================
# Charger TOUS les embeddings depuis Neo4j (pour training)
# ======================================================
def load_embeddings_from_neo4j(
    uri: str,
    user: str,
    password: str,
    database: str = "neo4j",
    limit: int | None = None,
):
    driver = GraphDatabase.driver(uri, auth=(user, password))

    query = """
    MATCH (p:Protein)
    WHERE p.embedding IS NOT NULL
    RETURN p.id AS id,
           p.entry_name AS entry_name,
           p.embedding AS emb,
           p.is_labelled AS y
    """

    if limit is not None:
        query += "\nLIMIT $limit"

    ids, names, X, y = [], [], [], []

    try:
        with driver.session(database=database) as session:
            result = session.run(query, limit=limit) if limit else session.run(query)
            for r in result:
                ids.append(r["id"])
                names.append(r["entry_name"])
                X.append(r["emb"])
                y.append(r["y"])
    finally:
        driver.close()

    if not X:
        raise RuntimeError(
            "Aucun embedding trouvé dans Neo4j. "
            "As-tu bien exécuté gds.fastRP.write(...) ?"
        )

    X = np.asarray(X, dtype=np.float32)

    y_arr = None
    if any(v is not None for v in y):
        y_arr = np.array([int(v) if v is not None else -1 for v in y], dtype=np.int32)

    return ids, names, X, y_arr


# ======================================================
# Charger l'embedding d'UNE protéine (pour prédiction)
# ======================================================
def get_embedding_for_pid(
    uri: str,
    user: str,
    password: str,
    database: str,
    pid: str,
):
    driver = GraphDatabase.driver(uri, auth=(user, password))
    try:
        with driver.session(database=database) as session:
            r = session.run(
                """
                MATCH (p:Protein {id:$id})
                RETURN p.embedding AS emb
                """,
                id=pid,
            ).single()

            if r is None or r["emb"] is None:
                return None

            return np.asarray(r["emb"], dtype=np.float32)
    finally:
        driver.close()


# ======================================================
# Test local (NE S'EXÉCUTE PAS quand importé)
# ======================================================
if __name__ == "__main__":
    URI = "neo4j://localhost:7687"
    USER = "neo4j"
    PASSWORD = "Ab220398."
    DB = "protein-graph-final"

    ids, names, X, y = load_embeddings_from_neo4j(
        URI, USER, PASSWORD, DB, limit=5
    )
    print("OK:", X.shape, ids[0], names[0])
