# ========================
# 1. Importar librerías
# ========================
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

# ========================
# 2. Cargar los archivos (con rutas corregidas)
# ========================
events = pd.read_csv("data/Sample_Game_1/Sample_Game_1_RawEventsData.csv")
tracking_home = pd.read_csv("data/Sample_Game_1/Sample_Game_1_RawTrackingData_Home_Team.csv", skiprows=2, low_memory=False)
tracking_away = pd.read_csv("data/Sample_Game_1/Sample_Game_1_RawTrackingData_Away_Team.csv", skiprows=2, low_memory=False)

print("Tipos de eventos disponibles:")
print(events["Type"].unique())

# ========================
# 3. Estadísticas básicas de eventos
# ========================
# Contar eventos por tipo

event_counts = events["Type"].value_counts()
print("\nNúmero de eventos por tipo:\n", event_counts)

# Filtrar solo los pases
passes = events[events["Type"] == "PASS"]

# Pases por equipo
passes_by_team = passes["Team"].value_counts()
print("\nPases por equipo:\n", passes_by_team)

# ========================
# 4. Estadísticas de jugadores
# ========================
# Pases intentados por jugador (columna 'From')
passes_by_player = passes["From"].value_counts()
print("\nPases realizados por jugador:\n", passes_by_player)

# Pases recibidos por jugador (columna 'To')
receives_by_player = passes["To"].value_counts()
print("\nPases recibidos por jugador:\n", receives_by_player)

# ========================
# 5. Redes de pases por equipo (sin números, nodos ∝ pases totales)
# ========================

def team_of_players_from_passes(passes_df: pd.DataFrame) -> pd.Series:
    # index=Player, value=Team (modo del jugador como pasador)
    return (passes_df.dropna(subset=["From", "Team"])
            .groupby("From")["Team"]
            .agg(lambda s: s.mode().iat[0] if not s.mode().empty else np.nan))

def build_undirected_pass_graph(passes_df: pd.DataFrame) -> nx.Graph:
    """
    Grafo no dirigido con una sola arista por pareja (A,B), peso = pases A->B + B->A.
    """
    pair_weight = {}
    for _, r in passes_df.iterrows():
        a, b = r["From"], r["To"]
        if pd.isna(a) or pd.isna(b) or a == b:
            continue
        pair = tuple(sorted((a, b)))
        pair_weight[pair] = pair_weight.get(pair, 0) + 1

    G = nx.Graph()
    for (a, b), w in pair_weight.items():
        G.add_edge(a, b, weight=w)
    return G


def node_strength(G: nx.Graph) -> dict:
    # fuerza = suma de pesos de aristas incidentes
    s = {n: 0 for n in G.nodes()}
    for u, v, d in G.edges(data=True):
        w = d.get("weight", 1)
        s[u] += w
        s[v] += w
    return s

def centered_spring_layout(G: nx.DiGraph, strength: dict, seed: int = 7) -> dict:
    # “hub” invisible para centrar a los jugadores con más fuerza
    H = G.copy()
    s_vals = np.array(list(strength.values()), dtype=float)
    s_norm = {n: (strength[n] / s_vals.max()) if s_vals.max() > 0 else 0.0 for n in H.nodes()}
    HUB = "__hub__"
    H.add_node(HUB)
    for n in list(H.nodes()):
        if n == HUB: 
            continue
        w = 0.5 + 2.0 * s_norm[n]     # atrae más al centro si tiene más fuerza
        H.add_edge(HUB, n, weight=w)
        H.add_edge(n, HUB, weight=w)
    pos = nx.spring_layout(H, weight="weight", seed=seed, iterations=200)
    pos.pop(HUB, None)
    return pos

def draw_team_graph(ax, team_name: str, team_passes: pd.DataFrame, title_suffix: str = ""):
    G = build_undirected_pass_graph(team_passes)
    if len(G) == 0:
        ax.set_title(f"{team_name}: sin datos de pases")
        ax.axis("off"); return

    strength = node_strength(G)

    # tamaño del nodo ~ 200 + 25*sqrt(strength) (suave y legible)
    node_sizes = [200.0 + 100.0 * np.sqrt(strength[n]) for n in G.nodes()]

    # layout centrado por fuerza
    pos = centered_spring_layout(G, strength)

    weights = np.array([d.get("weight", 1) for _, _, d in G.edges(data=True)], dtype=float)
    if weights.size > 0:
        w_min, w_max = weights.min(), weights.max()
        widths = 1.0 + 4.0 * (weights - w_min) / (w_max - w_min + 1e-9)  # ← ajusta 4.0 para más grosor
        alpha = 0.30 + 0.55 * (weights - w_min) / (w_max - w_min + 1e-9)
    else:
        widths = []; alpha = []

    ax.set_title(f"{team_name} {title_suffix}")

    nx.draw_networkx_edges(
        G, pos, ax=ax,
        width=widths if len(widths) else 1.0,
        edge_color="gray",
        alpha=float(alpha.mean()) if len(alpha) else 0.4,
        connectionstyle="arc3,rad=0.08",  # curva suave para cruces
    )

    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_size=node_sizes,
        node_color="#8ecae6", edgecolors="#023047", linewidths=1.2
    )
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=9)
    ax.axis("off")

# --- preparar datos por equipo ---
passes = events[events["Type"] == "PASS"].copy()
passes = passes.dropna(subset=["From", "To"])

# si confías en events["Team"], úsala; si no, inferimos por el pasador:
# passes["FromTeam"] = passes["Team"]
player_team = team_of_players_from_passes(passes)
passes["FromTeam"] = passes["From"].map(player_team)

teams = sorted(passes["FromTeam"].dropna().unique())

cols = 2
rows = int(np.ceil(len(teams)/cols)) if len(teams) else 1
fig, axes = plt.subplots(rows, cols, figsize=(13, 6*rows))
if not isinstance(axes, np.ndarray):
    axes = np.array([axes])
axes = axes.flatten()

for i, team in enumerate(teams):
    team_passes = passes[passes["FromTeam"] == team]
    draw_team_graph(axes[i], team, team_passes, title_suffix="— Red de pases")

for j in range(len(teams), len(axes)):
    axes[j].axis("off")

plt.suptitle("Redes de pases por equipo (nodos ∝ pases hechos+recibidos)", y=0.98, fontsize=14)
plt.tight_layout()
plt.show()


