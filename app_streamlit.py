import re
import os
import tempfile

import streamlit as st
import pandas as pd
from pymongo import MongoClient
from neo4j import GraphDatabase
from pyvis.network import Network
import streamlit.components.v1 as components

# =========================================================
# CONFIG APP
# =========================================================
st.set_page_config(page_title="Protein Explorer", layout="wide")
st.markdown("""
<style>
/* ✅ Ne PAS mettre .stApp * sinon tu casses tout */
.stApp {
  font-size: 18px;
}

/* Texte normal (paragraphes, listes, etc.) */
.stMarkdown p, .stMarkdown li, .stText, .stCaption, .stWrite, .stAlert p {
  font-size: 18px !important;
}

/* Widgets (labels + valeurs) */
label, label * {
  font-size: 17px !important;
}

/* Tabs */
button[role="tab"], button[role="tab"] * {
  font-size: 17px !important;
}
</style>
""", unsafe_allow_html=True)


# =========================
# COULEURS (graphe)
# =========================
COULEUR_CENTRE = "#371f23"   # centre
COULEUR_HOP1   = "#1024b9"   # voisins directs
COULEUR_HOP2   = "#fa6060"   # voisins des voisins

# =========================================================
# CONFIG MONGODB
# =========================================================
MONGO_HOST = "mongodb.telecomnancy.univ-lorraine.fr"
MONGO_PORT = 27017
MONGO_USER = "tncy"
MONGO_PASS = "etjustelaunpetitgeranium!"
AUTH_DB    = "admin"

DB_NAME    = "protein_db_kenza"
COL_NAME   = "proteins_kenza"

@st.cache_resource
def get_collection():
    uri = f"mongodb://{MONGO_USER}:{MONGO_PASS}@{MONGO_HOST}:{MONGO_PORT}/?authSource={AUTH_DB}"
    client = MongoClient(uri)
    return client[DB_NAME][COL_NAME]

col = get_collection()  # ✅ DOIT être global (hors onglets, hors if)


@st.cache_resource
def get_mongo_collection():
    uri = f"mongodb://{MONGO_USER}:{MONGO_PASS}@{MONGO_HOST}:{MONGO_PORT}/?authSource={AUTH_DB}"
    client = MongoClient(uri)
    return client[DB_NAME][COL_NAME]

mongo_col = get_mongo_collection()

# =========================================================
# CONFIG NEO4J
# =========================================================
NEO4J_URI  = "neo4j://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASS = "Ab220398."
NEO4J_DB   = "protein-graph-final"

@st.cache_resource
def get_neo_driver():
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))

neo = get_neo_driver()

# =========================================================
# CSS (input un peu plus propre)
# =========================================================
st.markdown("""
<style>
div[data-testid="stTextInput"] input {
    border: 2px solid black !important;
    border-radius: 8px;
    background-color: white !important;
    color: black !important;
}
div[data-testid="stTextInput"] label {
    color: black !important;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# MONGODB HELPERS
# =========================================================
def mongo_get_by_entry(entry: str):
    return mongo_col.find_one({"entry": entry})

def mongo_search(q: str, field: str, limit: int = 30):
    q = (q or "").strip()
    if not q:
        return []

    # Champs réels dans ta DB:
    # entry, entry_name, protein_names, organism, ec_numbers, interpro_domains, go_terms, sequence
    if field == "entry":
        query = {"entry": q}
    elif field == "entry_name":
        query = {"entry_name": {"$regex": q, "$options": "i"}}
    else:  # protein_names (description)
        query = {"protein_names": {"$regex": q, "$options": "i"}}

    projection = {
        "_id": 0,
        "entry": 1,
        "entry_name": 1,
        "protein_names": 1,
        "organism": 1,
        "ec_numbers": 1,
        "interpro_domains": 1,
        "go_terms": 1,
        "sequence": 1,
    }
    return list(mongo_col.find(query, projection).limit(limit))

def mongo_render_protein(doc: dict):
    if not doc:
        st.info("Aucun document à afficher.")
        return

    st.subheader("📄 Détails (MongoDB)")
    st.write("**entry** :", doc.get("entry", "—"))
    st.write("**entry_name** :", doc.get("entry_name", "—"))
    st.write("**protein_names** :", doc.get("protein_names", "—"))
    st.write("**organism** :", doc.get("organism", "—"))

    ec_count  = len(doc.get("ec_numbers", []) or [])
    ipr_count = len(doc.get("interpro_domains", []) or [])
    go_count  = len(doc.get("go_terms", []) or [])

    # ===== Grand cadre avec séparateurs verticaux =====
    st.markdown(f"""
    <div class="stats-box">
        <div style="display:flex; justify-content:space-between;">
            <div class="stats-col stats-divider" style="width:33%;">
                <h4>EC numbers</h4>
                <h2>{ec_count}</h2>
            </div>
            <div class="stats-col stats-divider" style="width:33%;">
                <h4>InterPro domains</h4>
                <h2>{ipr_count}</h2>
            </div>
            <div class="stats-col" style="width:33%;">
                <h4>GO terms</h4>
                <h2>{go_count}</h2>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # --- Listes détaillées
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("### EC numbers")
        st.write(doc.get("ec_numbers", []))
    with c2:
        st.markdown("### InterPro domains")
        st.write(doc.get("interpro_domains", []))
    with c3:
        st.markdown("### GO terms")
        st.write(doc.get("go_terms", []))

    with st.expander("🧬 Sequence"):
        st.code(doc.get("sequence", ""), language="text")

    with st.expander("🧾 JSON complet"):
        st.json(doc)



# =========================================================
# Annotation HELPERS
# =========================================================

from collections import defaultdict
import pandas as pd

# ---------- Mongo: récupérer GO/EC d'une protéine ----------
def mongo_get_labels(entry: str):
    doc = col.find_one(
        {"entry": entry},
        {"go_terms": 1, "ec_numbers": 1, "_id": 0}
    )
    if not doc:
        return [], []
    go = doc.get("go_terms", []) or []
    ec = doc.get("ec_numbers", []) or []
    return go, ec

# ---------- Neo4j: voisins 1-hop avec poids ----------
def neo_get_neighbors_1hop(pid: str, minw: float = 0.0, limit: int = 200):
    with neo.session(database=NEO4J_DB) as session:
        rows = session.run("""
            MATCH (p:Protein {id:$id})-[r:SIMILAR]-(n:Protein)
            WHERE r.weight >= $minw
            RETURN n.id AS nid, r.weight AS w
            ORDER BY w DESC
            LIMIT $lim
        """, id=pid, minw=minw, lim=limit).data()

    # liste de tuples (neighbor_id, weight)
    return [(r["nid"], float(r["w"])) for r in rows]

# ---------- Label propagation (1-hop) ----------
def propagate_labels_1hop(pid: str, minw: float = 0.0, n_neighbors: int = 100, topk_go: int = 10, topk_ec: int = 5):
    neighbors = neo_get_neighbors_1hop(pid, minw=minw, limit=n_neighbors)

    go_scores = defaultdict(float)
    ec_scores = defaultdict(float)

    for nid, w in neighbors:
        go_terms, ec_numbers = mongo_get_labels(nid)

        for go in go_terms:
            go_scores[go] += w

        for ec in ec_numbers:
            # on garde tel quel, ou tu peux préfixer "EC:" si tu veux
            ec_scores[ec] += w

    go_ranked = sorted(go_scores.items(), key=lambda x: x[1], reverse=True)[:topk_go]
    ec_ranked = sorted(ec_scores.items(), key=lambda x: x[1], reverse=True)[:topk_ec]

    go_df = pd.DataFrame(go_ranked, columns=["GO term", "score"])
    ec_df = pd.DataFrame(ec_ranked, columns=["EC number", "score"])

    return neighbors, go_df, ec_df


# =========================================================
# NEO4J GRAPH HELPERS
# =========================================================

import os
import tempfile
import streamlit as st
from pyvis.network import Network
import streamlit.components.v1 as components

# =========================
# GRAPH HELPERS (Neo4j + PyVis)
# =========================

def edge_width(w: float, minw: float) -> float:
    """
    Convertit un poids (similarité) en épaisseur d'arête.
    Retourne une valeur entre 1 et 4 (max), pour éviter les edges trop épaisses.
    """
    if w <= minw:
        return 1.0
    x = (w - minw) / max(1e-9, (1.0 - minw))
    return 1.0 + 3.0 * x  # [1..4]

def make_net():
    net = Network(height="700px", width="100%", directed=False)

    # Physique / espacement
    net.repulsion(
        node_distance=240,
        central_gravity=0.2,
        spring_length=160
    )

    # Options vis.js (via PyVis)
    net.set_options("""
    {
      "interaction": {
        "zoomView": true,
        "dragView": true,
        "zoomSpeed": 0.35,
        "hover": true,
        "hoverConnectedEdges": true,
        "selectConnectedEdges": true
      },
      "nodes": {
        "font": { "color": "#000000", "size": 18 },
        "borderWidth": 1
      },
      "edges": {
        "color": {
          "color": "#333333",
          "highlight": "#000000",
          "hover": "#000000",
          "opacity": 0.65
        },
        "smooth": false,
        "width": 1,
        "hoverWidth": 2,
        "selectionWidth": 3,
        "scaling": {
          "enabled": true,
          "min": 1,
          "max": 4
        }
      },
      "physics": {
        "enabled": true,
        "stabilization": { "iterations": 200 }
      }
    }
    """)
    return net

def show_pyvis_in_black_frame(net: Network, height: int = 740):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp:
        net.save_graph(tmp.name)
        html = open(tmp.name, "r", encoding="utf-8").read()
    os.remove(tmp.name)

    wrapper = f"""
    <div style="border: 2px solid #000; border-radius: 12px; padding: 8px; background: #fff;">
        {html}
    </div>
    """
    components.html(wrapper, height=height, scrolling=True)

def neo_render_graph(
    neo_driver,
    neo_db: str,
    pid: str,
    mode: str,               # "1-hop" ou "2-hop"
    limite_aretes: int,
    poids_min: float,
    top_n1: int,
    COULEUR_CENTRE: str,
    COULEUR_HOP1: str,
    COULEUR_HOP2: str
):
    if not pid:
        st.info("Entre un identifiant UniProt.")
        return

    net = make_net()

    # Centre
    net.add_node(pid, label=pid, size=35, color=COULEUR_CENTRE)

    hop1_nodes, hop2_nodes = set(), set()
    added_edges = set()

    def add_edge_once(a, b, **kwargs):
        key = tuple(sorted([a, b]))
        if key not in added_edges:
            net.add_edge(a, b, **kwargs)
            added_edges.add(key)

    with neo_driver.session(database=neo_db) as session:
        if mode == "1-hop":
            rows = session.run("""
                MATCH (p:Protein {id:$id})-[r:SIMILAR]-(n:Protein)
                WHERE r.weight >= $minw
                RETURN n.id AS nid, r.weight AS w
                ORDER BY w DESC
                LIMIT $lim
            """, id=pid, minw=poids_min, lim=limite_aretes).data()

            for r in rows:
                nid, w = r["nid"], float(r["w"])

                if nid not in hop1_nodes:
                    hop1_nodes.add(nid)
                    net.add_node(nid, label=nid, size=16, color=COULEUR_HOP1)

                add_edge_once(
                    pid, nid,
                    value=w,
                    width=edge_width(w, poids_min),
                    color=COULEUR_HOP1,
                    title=f"{pid} ↔ {nid}\nPoids = {w:.3f}"
                )

        else:  # "2-hop"
            direct_rows = session.run("""
                MATCH (p:Protein {id:$id})-[r:SIMILAR]-(n:Protein)
                WHERE r.weight >= $minw
                RETURN n.id AS nid, r.weight AS w
            """, id=pid, minw=poids_min).data()

            voisins_directs = {r["nid"] for r in direct_rows}
            poids_direct = {r["nid"]: float(r["w"]) for r in direct_rows}

            # ✅ vrai top_n1 (les meilleurs voisins directs par poids)
            direct_sorted = sorted(direct_rows, key=lambda x: float(x["w"]), reverse=True)
            top_n1_ids = {r["nid"] for r in direct_sorted[:top_n1]}


            rows = session.run("""
                MATCH (p:Protein {id:$id})-[r1:SIMILAR]-(n1:Protein)
                WHERE r1.weight >= $minw
                WITH p,n1,r1
                ORDER BY r1.weight DESC
                LIMIT $top
                MATCH (n1)-[r2:SIMILAR]-(n2:Protein)
                WHERE r2.weight >= $minw
                RETURN n1.id AS n1, r1.weight AS w1,
                       n2.id AS n2, r2.weight AS w2
                LIMIT $lim
            """, id=pid, minw=poids_min, top=top_n1, lim=limite_aretes).data()

            for r in rows:
                n1, w1 = r["n1"], float(r["w1"])
                n2, w2 = r["n2"], float(r["w2"])

                # n1 : voisin direct
                if n1 not in hop1_nodes:
                    hop1_nodes.add(n1)
                    net.add_node(n1, label=n1, size=16, color=COULEUR_HOP1)

                add_edge_once(
                    pid, n1,
                    value=w1,
                    width=edge_width(w1, poids_min),
                    color=COULEUR_HOP1,
                    title=f"{pid} ↔ {n1}\nPoids = {w1:.3f}"
                )

                if n2 == pid:
                    continue

                # n2 : si direct -> bleu, sinon rouge
                # ✅ ne mettre en bleu QUE si n2 est un voisin direct ET dans le top_n1
                if n2 in voisins_directs and n2 in top_n1_ids:
                    if n2 not in hop1_nodes:
                        hop1_nodes.add(n2)
                        net.add_node(n2, label=n2, size=16, color=COULEUR_HOP1)

                    w_direct = poids_direct.get(n2)
                    if w_direct is not None:
                        add_edge_once(
                            pid, n2,
                            value=w_direct,
                            width=edge_width(w_direct, poids_min),
                            color=COULEUR_HOP1,
                            title=f"{pid} ↔ {n2}\nPoids = {w_direct:.3f}"
                        )
                else:
                    if n2 != pid and n2 not in hop2_nodes:
                        hop2_nodes.add(n2)
                        net.add_node(n2, label=n2, size=12, color=COULEUR_HOP2)


                # arête du chemin n1--n2
                couleur_arete = COULEUR_HOP2 if n2 not in voisins_directs else COULEUR_HOP1
                add_edge_once(
                    n1, n2,
                    value=w2,
                    width=edge_width(w2, poids_min),
                    color=couleur_arete,
                    title=f"{n1} ↔ {n2}\nPoids = {w2:.3f}"
                )

    st.caption(f"Nombre de nœuds : {len(net.nodes)} | Nombre d'arêtes : {len(net.edges)}")
    show_pyvis_in_black_frame(net, height=760)

# =========================================================
# APP UI (TABS)
# =========================================================
st.markdown("""
<div id="main-title-wrap">
  <div id="main-title">Protein Explorer</div>
  <div id="main-subtitle">MongoDB • Neo4j • Graph • Annotation • ML</div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<style>
#main-title-wrap{
  display:flex;
  flex-direction:column;
  align-items:center;
  margin-top:-10px;
  margin-bottom:14px;
}

/* le “badge” du titre */
#main-title{
  padding: 10px 18px;
  border: 2px solid rgba(0,0,0,0.18);
  border-radius: 16px;
  background: linear-gradient(90deg, rgba(16,36,185,0.12), rgba(250,96,96,0.12));
  box-shadow: 0 8px 20px rgba(0,0,0,0.10);
  font-size: 64px !important;
  font-weight: 900;
  letter-spacing: 0.2px;
  color: #121212;
  line-height: 1.05;
  transition: transform 0.15s ease, box-shadow 0.15s ease;
}

#main-title:hover{
  transform: translateY(-2px);
  box-shadow: 0 12px 26px rgba(0,0,0,0.14);
}

/* sous-titre */
#main-subtitle{
  margin-top: 8px;
  font-size: 16px !important;
  font-weight: 600;
  color: rgba(0,0,0,0.60);
}
</style>
""", unsafe_allow_html=True)





st.markdown("Utilise les onglets pour chercher dans **MongoDB**, visualiser dans **Neo4j**, ou faire une vue combinée.")

tab_mongo, tab_neo, tab_combo, tab_stats, tab_annot,tab_ml = st.tabs(["📄 MongoDB", "🕸️ Neo4j Graphe", "🧩 Vue combinée", "📈 Statistiques","🧬 Annotation","🤖 Machine Learning Prédiction"])

# -------------------------
# TAB 1: MongoDB
# -------------------------
with tab_mongo:
    st.subheader("Recherche MongoDB")

    field = st.radio(
        "Rechercher par :",
        ["entry", "entry_name", "protein_names"],
        horizontal=True
    )

    col_input, col_btn = st.columns([6, 1])

    with col_input:
        q = st.text_input(
            "Recherche",
            placeholder="Ex: A0A087X1C5, cytochrome, kinase...",
            label_visibility="collapsed"  # 🔥 clé de l’alignement
        )

    with col_btn:
        search_clicked = st.button(
            "🔎",
            use_container_width=True
        )

    LIMIT_RESULTS = 30

    if search_clicked and q:
        st.session_state["mongo_results"] = mongo_search(
            q, field, limit=LIMIT_RESULTS
        )


    results = st.session_state.get("mongo_results", [])
    

    if results:
        labels = [
            f'{d.get("entry","?")} — {d.get("entry_name","")}'
            for d in results
        ]

        choice = st.selectbox("Choisir une protéine :", labels)
        selected = results[labels.index(choice)]

        # tableau résumé
        df = pd.DataFrame([{
            "entry": d.get("entry"),
            "entry_name": d.get("entry_name"),
            "protein_names": (d.get("protein_names") or "")[:80],
            "organism": d.get("organism")
        } for d in results])

        st.dataframe(df, use_container_width=True, hide_index=True)

        st.divider()
        mongo_render_protein(selected)

        # pour synchro avec Neo4j / combo
        st.session_state["selected_entry"] = selected.get("entry")
    else:
        st.info("Lance une recherche pour afficher des protéines.")

# -------------------------
# TAB 2: Neo4j Graph
# -------------------------
with tab_neo:
    st.subheader("Visualisation Neo4j")
    st.markdown("🟤 **Centre** · 🔵 **Voisins directs** · 🔴 **Voisins des voisins**")

    default_pid = st.session_state.get("selected_entry", "A0A087X1C5")
    pid = st.text_input("Identifiant UniProt", value=default_pid).strip()

    mode_label = st.radio("Profondeur", ["Voisins directs (1 saut)", "Voisins des voisins (2 sauts)"], horizontal=True)
    mode = "1-hop" if mode_label.startswith("Voisins directs") else "2-hop"

    c1, c2, c3 = st.columns(3)
    with c1:
        limite_aretes = st.slider("Limite arêtes", 10, 400, 120, 10)
    with c2:
        poids_min = st.slider("Poids minimal", 0.0, 1.0, 0.0, 0.05)
    with c3:
        top_n1 = st.slider("Top voisins (pour 2 sauts)", 5, 80, 20, 5)

    st.divider()
    neo_render_graph(
    neo_driver=neo,
    neo_db=NEO4J_DB,
    pid=pid,
    mode=mode,
    limite_aretes=limite_aretes,
    poids_min=poids_min,
    top_n1=top_n1,
    COULEUR_CENTRE=COULEUR_CENTRE,
    COULEUR_HOP1=COULEUR_HOP1,
    COULEUR_HOP2=COULEUR_HOP2
)


# -------------------------
# TAB 3: Combo (Mongo + Graph)
# -------------------------
with tab_combo:
    st.subheader("Vue combinée : MongoDB + Neo4j")
    

    default_pid = st.session_state.get("selected_entry", "A0A087X1C5")
    pid = st.text_input("Entry UniProt", value=default_pid).strip()

    left, right = st.columns([1, 1])

    with left:
        doc = mongo_get_by_entry(pid) if pid else None
        if doc:
            doc["_id"] = str(doc["_id"])
            mongo_render_protein(doc)
        else:
            st.info("Aucun doc Mongo trouvé pour cet entry (ou entry vide).")

    with right:
        mode = st.radio("Voisinage", ["Voisins directs", "Voisins des voisins"], horizontal=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            limite_aretes = st.slider("Limite arêtes (combo)", 10, 400, 120, 10, key="combo_lim")
        with c2:
            poids_min = st.slider("Poids minimal (combo)", 0.0, 1.0, 0.0, 0.05, key="combo_min")
        with c3:
            top_n1 = st.slider("Top voisins (combo)", 5, 80, 20, 5, key="combo_top")

        neo_render_graph(
            neo_driver=neo,
            neo_db=NEO4J_DB,
            pid=pid,
            mode=mode,
            limite_aretes=limite_aretes,
            poids_min=poids_min,
            top_n1=top_n1,
            COULEUR_CENTRE=COULEUR_CENTRE,
            COULEUR_HOP1=COULEUR_HOP1,
            COULEUR_HOP2=COULEUR_HOP2
        )


# -------------------------
# TAB 4: Stats (simple version)
# -------------------------
import matplotlib.pyplot as plt
import pandas as pd

with tab_stats:
    st.subheader("📈 Statistiques (Dashboard)")

    # --- Options interactives
    colA, colB, colC = st.columns([2, 1, 1])
    with colA:
        run_stats = st.button("🔄 Rafraîchir les stats", use_container_width=True)
    with colB:
        top_k = st.slider("Top K (tables)", 5, 30, 10, 1)
    with colC:
        show_tables = st.checkbox("Afficher tables", value=True)

    st.divider()

    # On calcule seulement quand on clique (évite de recalculer à chaque rerun)
    if run_stats:
        # =========================
        # 1) MONGO STATS
        # =========================
        total = mongo_col.count_documents({})

        # "annotés" = champ existe et non vide
        go_annot = mongo_col.count_documents({"go_terms": {"$exists": True, "$ne": []}})
        ec_annot = mongo_col.count_documents({"ec_numbers": {"$exists": True, "$ne": []}})
        ipr_annot = mongo_col.count_documents({"interpro_domains": {"$exists": True, "$ne": []}})

        # Averages via aggregation
        avg_pipe = [
            {"$project": {
                "go": {"$size": {"$ifNull": ["$go_terms", []]}},
                "ec": {"$size": {"$ifNull": ["$ec_numbers", []]}},
                "ipr": {"$size": {"$ifNull": ["$interpro_domains", []]}}
            }},
            {"$group": {
                "_id": None,
                "avg_go": {"$avg": "$go"},
                "avg_ec": {"$avg": "$ec"},
                "avg_ipr": {"$avg": "$ipr"},
                "max_go": {"$max": "$go"},
                "max_ec": {"$max": "$ec"},
                "max_ipr": {"$max": "$ipr"}
            }}
        ]
        avg_res = list(mongo_col.aggregate(avg_pipe))
        avg_res = avg_res[0] if avg_res else {"avg_go": 0, "avg_ec": 0, "avg_ipr": 0, "max_go": 0, "max_ec": 0, "max_ipr": 0}

        # Top organisms
        org_pipe = [
            {"$group": {"_id": "$organism", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}},
            {"$limit": top_k}
        ]
        top_org = list(mongo_col.aggregate(org_pipe))
        df_org = pd.DataFrame([{"organism": x["_id"] or "—", "count": x["count"]} for x in top_org])

        # =========================
        # 2) NEO4J STATS
        # =========================
        neo_stats = {}
        hubs_df = pd.DataFrame()

        try:
            with neo.session(database=NEO4J_DB) as session:
                # nodes
                neo_stats["nodes"] = session.run("MATCH (p:Protein) RETURN count(p) AS n").single()["n"]
                # edges
                neo_stats["edges"] = session.run("MATCH ()-[r:SIMILAR]-() RETURN count(r) AS m").single()["m"]
                neo_stats["edges"]=neo_stats["edges"]//2
                # isolated
                neo_stats["isolated"] = session.run("""
                    MATCH (p:Protein)
                    WHERE NOT (p)--()
                    RETURN count(p) AS isolated
                """).single()["isolated"]

                # degree avg/max
                deg_row = session.run("""
                    MATCH (p:Protein)
                    RETURN
                    avg(COUNT { (p)--() }) AS avg_degree,
                    max(COUNT { (p)--() }) AS max_degree

                """).single()
                neo_stats["avg_degree"] = float(deg_row["avg_degree"] or 0)
                neo_stats["max_degree"] = int(deg_row["max_degree"] or 0)

                # weights min/avg/max
                w_row = session.run("""
                    MATCH ()-[r:SIMILAR]-()
                    RETURN min(r.weight) AS minw, avg(r.weight) AS avgw, max(r.weight) AS maxw
                """).single()
                neo_stats["minw"] = float(w_row["minw"] or 0)
                neo_stats["avgw"] = float(w_row["avgw"] or 0)
                neo_stats["maxw"] = float(w_row["maxw"] or 0)

                # top hubs
                hubs = session.run("""
                    MATCH (p:Protein)
                    RETURN p.id AS id, COUNT { (p)--() } AS degree
                    ORDER BY degree DESC
                    LIMIT $k

                """, k=top_k).data()
                hubs_df = pd.DataFrame(hubs) if hubs else pd.DataFrame(columns=["id", "degree"])

        except Exception as e:
            st.warning(f"Neo4j stats non dispo: {e}")

        # Sauver en session_state pour afficher sans recalculer si on change juste l’UI
        st.session_state["stats_payload"] = {
            "mongo": {
                "total": total,
                "go_annot": go_annot,
                "ec_annot": ec_annot,
                "ipr_annot": ipr_annot,
                "avg": avg_res,
                "df_org": df_org
            },
            "neo": neo_stats,
            "hubs_df": hubs_df
        }

    payload = st.session_state.get("stats_payload")
    if not payload:
        st.info("Clique sur **Rafraîchir les stats** pour charger le dashboard.")
    else:

        # =========================
        # AFFICHAGE DASHBOARD
        # =========================
        mongo_s = payload["mongo"]
        neo_s = payload["neo"]
        hubs_df = payload["hubs_df"]

        # --- KPIs (ligne 1)
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Mongo: total protéines", mongo_s["total"])
        k2.metric("Mongo: GO annotées", mongo_s["go_annot"])
        k3.metric("Mongo: EC annotées", mongo_s["ec_annot"])
        k4.metric("Mongo: InterPro annotées", mongo_s["ipr_annot"])

        # --- KPIs (ligne 2)
        k5, k6, k7, k8 = st.columns(4)
        k5.metric("Neo4j: nœuds", neo_s.get("nodes", "—"))
        k6.metric("Neo4j: arêtes SIMILAR", neo_s.get("edges", "—"))
        k7.metric("Neo4j: isolées", neo_s.get("isolated", "—"))
        k8.metric("Neo4j: degré moyen", round(neo_s.get("avg_degree", 0), 2) if neo_s else "—")

        st.divider()

        # --- Charts en 2 colonnes
        left, right = st.columns(2)

        import plotly.express as px

        st.markdown("###  Couverture des annotations (MongoDB)")

        col_left, col_right = st.columns(2)

        # --- Camembert 1 : types d’annotations ---
        with col_left:
            labels = ["GO terms", "EC numbers", "InterPro domains"]
            values = [mongo_s["go_annot"], mongo_s["ec_annot"], mongo_s["ipr_annot"]]

            fig = px.bar(
                x=["GO terms", "EC numbers", "InterPro domains"],
                y=[mongo_s["go_annot"], mongo_s["ec_annot"], mongo_s["ipr_annot"]],
                labels={"x": "Type d’annotation", "y": "Nombre de protéines"},
                title="Couverture des annotations par type"
            )
            st.plotly_chart(fig, use_container_width=True)

           

        # --- Camembert 2 : labelled vs unlabelled ---
        with col_right:
            total = mongo_s["total"] or 1
            labeled = mongo_s["go_annot"]
            unlabeled = max(0, total - labeled)

            fig2 = px.pie(
                names=["Labelled (GO)", "Unlabelled"],
                values=[labeled, unlabeled],
                title="Labelled vs Unlabelled (GO)",
                hole=0.35
            )
            st.plotly_chart(fig2, use_container_width=True)



    

        st.divider()

        
        st.markdown("###  Poids des arêtes (Neo4j)")

        if neo_s:
            st.write(f"- **Minimum** : {neo_s.get('minw', 0):.3f}")
            st.write(f"- **Moyenne** : {neo_s.get('avgw', 0):.3f}")
            st.write(f"- **Maximum** : {neo_s.get('maxw', 0):.3f}")
        else:
            st.info("Poids indisponibles (Neo4j non connecté).")

            

        # --- Tables (optionnel)
        if show_tables:
            st.divider()
            st.markdown("### 🧾 Table: Top hubs")

            if hubs_df is not None and not hubs_df.empty:
                st.dataframe(hubs_df, use_container_width=True, hide_index=True)
            else:
                st.caption("—")


      

with tab_annot:
    st.subheader(" Annotation (Label Propagation)")
    st.caption("Recommande des GO/EC pour une protéine à partir des labels des voisins (pondérés par le poids SIMILAR).")

    default_pid = st.session_state.get("selected_entry", "A0A087X1C5")
    pid_annot = st.text_input("Protéine à annoter (UniProt entry)", value=default_pid, key="pid_annot").strip()

    c1, c2, c3 = st.columns(3)
    with c1:
        minw = st.slider("Seuil poids (min)", 0.0, 1.0, 0.3, 0.05, key="lp_minw")
    with c2:
        n_neighbors = st.slider("Nb voisins utilisés", 10, 300, 100, 10, key="lp_n")
    with c3:
        topk_go = st.slider("Top GO", 3, 30, 10, 1, key="lp_topgo")

    topk_ec = st.slider("Top EC", 1, 15, 5, 1, key="lp_topec")

    if st.button("✨ Recommander des labels", use_container_width=True):
        if not pid_annot:
            st.warning("Entre un identifiant UniProt.")
        else:
            neighbors, go_df, ec_df = propagate_labels_1hop(
                pid_annot,
                minw=minw,
                n_neighbors=n_neighbors,
                topk_go=topk_go,
                topk_ec=topk_ec
            )

            st.caption(f"Voisins utilisés : {len(neighbors)} (poids ≥ {minw})")

            with st.expander("Voir les voisins (id, poids)"):
                st.dataframe(
                    pd.DataFrame(neighbors, columns=["neighbor_id", "weight"]),
                    use_container_width=True,
                    hide_index=True
                )

            st.markdown("### ✅ GO recommandés")
            if not go_df.empty:
                st.dataframe(go_df, use_container_width=True, hide_index=True)
            else:
                st.info("Aucun GO recommandé (voisins non annotés ou filtre trop strict).")

            st.markdown("### ✅ EC recommandés")
            if not ec_df.empty:
                st.dataframe(ec_df, use_container_width=True, hide_index=True)
            else:
               st.info("Aucun EC recommandé (voisins non annotés ou filtre trop strict).")

with tab_ml:
    st.subheader("ML — EC ")
    st.caption("Entraînement plus rapide : prédiction multi-label des EC numbers uniquement.")

    min_freq_ec = st.number_input("Seuil fréquence EC", min_value=1, value=10, step=1)

    if st.button(" Entraîner et sauvegarder (EC)", use_container_width=True):
        from ML import load_embeddings_from_neo4j
        ids, names, X, _ = load_embeddings_from_neo4j(
            uri=NEO4J_URI,
            user=NEO4J_USER,
            password=NEO4J_PASS,
            database=NEO4J_DB,
            limit=None
        )

        from ml_gds_model import train_and_save_gds_models
        ec_m = train_and_save_gds_models(
            ids=ids, X=X, mongo_col=mongo_col, min_freq_ec=int(min_freq_ec)
        )
        st.success("✅ Modèle EC sauvegardé dans ml_artifacts/gds_multilabel.joblib")
        st.write("EC metrics:", ec_m)

    st.divider()

    st.markdown("###  Prédire EC")
    pid = st.text_input("Protein id (p.id)", value=st.session_state.get("selected_entry", "A0A087X1C5")).strip()
    k_ec = st.slider("Top-K EC", 1, 20, 5, 1)

    if st.button(" Prédire EC", use_container_width=True):
        from ml_gds_model import load_artifacts, predict_topk_from_embedding
        from ML import get_embedding_for_pid

        try:
            art = load_artifacts()
        except RuntimeError as e:
            st.warning(str(e))
            st.stop()

        emb = get_embedding_for_pid(
            uri=NEO4J_URI,
            user=NEO4J_USER,
            password=NEO4J_PASS,
            database=NEO4J_DB,
            pid=pid
        )

        if emb is None:
            st.error("❌ Aucun embedding trouvé pour cette protéine dans Neo4j (p.embedding est NULL).")
        else:
            ec_preds = predict_topk_from_embedding(emb, art["ec_model"], art["ec_mlb"], k=int(k_ec))
            st.dataframe(pd.DataFrame(ec_preds, columns=["EC", "score"]), use_container_width=True, hide_index=True)
            st.caption(f"Params: {art.get('params', {})} | Metrics: {art.get('metrics', {})}")
