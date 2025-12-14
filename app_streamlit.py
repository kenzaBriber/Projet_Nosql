import re
import streamlit as st
from neo4j import GraphDatabase
from pyvis.network import Network
import tempfile
import os
import streamlit.components.v1 as components

# =========================
# COULEURS (à toi d'ajuster)
# =========================
COULEUR_CENTRE = "#371f23"   # centre
COULEUR_HOP1   = "#1024b9"   # voisins directs
COULEUR_HOP2   = "#fa6060"   # voisins des voisins

# =========================
# CONFIG NEO4J
# =========================
NEO4J_URI  = "neo4j://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASS = "Ab220398."
NEO4J_DB   = "protein-graph-final"

st.set_page_config(page_title="Visualisation Graphe Protéines", layout="wide")
st.markdown("""
<style>
/* Champ texte (input) */
div[data-testid="stTextInput"] input {
    border: 2px solid black !important;
    border-radius: 6px;
    background-color: white !important;
    color: black !important;
}

/* Label du champ */
div[data-testid="stTextInput"] label {
    color: black !important;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

# =========================
# Connexion Neo4j
# =========================
@st.cache_resource
def get_neo_driver():
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))

neo = get_neo_driver()

# =========================
# Helpers
# =========================
def looks_like_uniprot_id(q: str) -> bool:
    return bool(re.fullmatch(r"[A-NR-Z0-9]{6,10}", q.strip()))

def make_net():
    net = Network(height="700px", width="100%", directed=False)

    # Physique / espacement (tu peux ajuster)
    net.repulsion(node_distance=240, central_gravity=0.2, spring_length=160)

    # Fond blanc + réglages interaction (zoom doux)
    net.set_options("""
    {
      "interaction": {
        "zoomView": true,
        "dragView": true,
        "zoomSpeed": 0.35
      },
      "nodes": {
        "font": { "color": "#000000" }
      },
      "edges": {
        "color": { "inherit": false },
        "smooth": false
      }
    }
    """)

    return net

def show_pyvis_in_black_frame(net: Network, height: int = 740):
    # Sauvegarde HTML
    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp:
        net.save_graph(tmp.name)
        html = open(tmp.name, "r", encoding="utf-8").read()
    os.remove(tmp.name)

    # ✅ Cadre noir + fond blanc autour de l'iframe
    wrapper = f"""
    <div style="
        border: 2px solid #000;
        border-radius: 10px;
        padding: 8px;
        background: #fff;">
        {html}
    </div>
    """
    components.html(wrapper, height=height, scrolling=True)

# =========================
# UI
# =========================
st.title("Visualisation du graphe de la protéine")
st.markdown("🟤 **Protéine centrale** &nbsp;&nbsp; 🔵 **Voisins directs** &nbsp;&nbsp; 🔴 **Voisins des voisins**")

pid = st.text_input("Identifiant UniProt (ex : A0A087X1C5)", value="A0A087X1C5").strip()

mode = st.radio(
    "Profondeur du voisinage",
    ["Voisins directs (1 saut)", "Voisins des voisins (2 sauts)"],
    horizontal=True
)

col1, col2, col3 = st.columns(3)
with col1:
    limite_aretes = st.slider("Limite totale d'arêtes affichées", 10, 400, 120, 10)
with col2:
    poids_min = st.slider("Poids minimal (filtre)", 0.0, 1.0, 0.0, 0.05)
with col3:
    top_n1 = st.slider("Top voisins directs utilisés pour explorer le 2 sauts", 5, 80, 20, 5)

if not pid:
    st.info("Entre un identifiant de protéine.")
else:
    net = make_net()

    with neo.session(database=NEO4J_DB) as session:
        # --- Ajouter le centre
        net.add_node(pid, label=pid, size=35, color=COULEUR_CENTRE)

        hop1_nodes = set()
        hop2_nodes = set()

        # Eviter les arêtes en doublon
        added_edges = set()
        def add_edge_once(a, b, **kwargs):
            key = tuple(sorted([a, b]))
            if key not in added_edges:
                net.add_edge(a, b, **kwargs)
                added_edges.add(key)

        if mode.startswith("Voisins directs"):
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

                add_edge_once(pid, nid, value=w, title=f"poids = {w:.3f}", color=COULEUR_HOP1, width=2)

        else:
            # 1) voisins directs (pour savoir qui est vraiment 1-hop + poids)
            direct_rows = session.run("""
                MATCH (p:Protein {id:$id})-[r:SIMILAR]-(n:Protein)
                WHERE r.weight >= $minw
                RETURN n.id AS nid, r.weight AS w
            """, id=pid, minw=poids_min).data()

            voisins_directs = {r["nid"] for r in direct_rows}
            poids_direct = {r["nid"]: float(r["w"]) for r in direct_rows}

            # 2) parcours 2-sauts via top_n1 meilleurs voisins
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

                # n1 est direct => bleu
                if n1 not in hop1_nodes:
                    hop1_nodes.add(n1)
                    net.add_node(n1, label=n1, size=16, color=COULEUR_HOP1)

                add_edge_once(pid, n1, value=w1, title=f"poids = {w1:.3f}", color=COULEUR_HOP1, width=2)

                # n2 : si direct => bleu + arête p--n2 ; sinon rouge
                if n2 == pid:
                    continue

                if n2 in voisins_directs:
                    if n2 not in hop1_nodes:
                        hop1_nodes.add(n2)
                        net.add_node(n2, label=n2, size=16, color=COULEUR_HOP1)

                    w_direct = poids_direct.get(n2)
                    if w_direct is not None:
                        add_edge_once(pid, n2, value=w_direct, title=f"poids = {w_direct:.3f}", color=COULEUR_HOP1, width=2)
                else:
                    if n2 not in hop2_nodes:
                        hop2_nodes.add(n2)
                        net.add_node(n2, label=n2, size=12, color=COULEUR_HOP2)

                # arête n1--n2 (chemin)
                couleur_arete = COULEUR_HOP2 if n2 not in voisins_directs else COULEUR_HOP1
                add_edge_once(n1, n2, value=w2, title=f"poids = {w2:.3f}", color=couleur_arete, width=1)

    st.write(f"Nombre de nœuds : {len(net.nodes)} | Nombre d'arêtes : {len(net.edges)}")
    show_pyvis_in_black_frame(net, height=760)
