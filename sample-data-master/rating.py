# ==============================================
# script2.py - Ranking por equipos con nota 0-10
# ==============================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------
# 0) Configuración básica
# ------------------------------------------------
DATA_PATH = "data/Sample_Game_1/Sample_Game_1_RawEventsData.csv"
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------------------------------------
# 1) Cargar dataset de eventos
# ------------------------------------------------
events = pd.read_csv(DATA_PATH)

# Limpieza básica de columnas por si llegan con espacios
events.columns = events.columns.str.strip()

# ------------------------------------------------
# 2) Añadir columnas auxiliares de tipo de evento
#    (ajusta los nombres/valores si tu dataset usa otros)
# ------------------------------------------------
events["is_pass"] = events["Type"].astype(str).str.upper().eq("PASS")
events["is_shot"] = events["Type"].astype(str).str.upper().eq("SHOT")
events["is_recovery"] = events["Type"].astype(str).str.upper().eq("RECOVERY")
events["is_lost"] = events["Type"].astype(str).str.upper().isin(["BALL LOST", "BALL_LOST", "LOSS", "TURNOVER"])

# Pase exitoso = evento PASS con "To" no nulo (heurística simple)
events["successful_pass"] = events["is_pass"] & events["To"].notna()

# Asegurar tipos para agregaciones
for col in ["is_pass", "is_shot", "is_recovery", "is_lost", "successful_pass"]:
    events[col] = events[col].astype(int)

# ------------------------------------------------
# 3) Detección del campo "equipo" y asignación de equipo por jugador
#    Intentamos encontrar una columna de equipo en el dataset. Si no hay,
#    el script seguirá sin segmentar por equipos (pero lo normal es que haya).
# ------------------------------------------------
def pick_team_column(df: pd.DataFrame) -> str | None:
    candidates = [
        "Team", "team", "TEAM",
        "TeamFrom", "FromTeam", "From_Team",
        "TeamID", "Team_Id", "Team_Id_From"
    ]
    for c in candidates:
        if c in df.columns:
            return c
    return None

TEAM_COL = pick_team_column(events)

# Calculamos equipo por jugador como el modo del equipo en sus eventos "From"
if TEAM_COL is not None:
    # nos quedamos con filas donde existe un jugador en 'From'
    df_from = events[events["From"].notna() & events[TEAM_COL].notna()].copy()
    # equipo por jugador (modo)
    player_team = (
        df_from.groupby("From")[TEAM_COL]
        .agg(lambda s: s.mode().iat[0] if not s.mode().empty else np.nan)
        .rename("TeamOfPlayer")
    )
else:
    # no hay columna de equipo detectable
    player_team = pd.Series(dtype="object", name="TeamOfPlayer")

# ------------------------------------------------
# 4) Agregación de estadísticas por jugador
# ------------------------------------------------
agg = events.groupby("From").agg(
    total_events=("Type", "count"),
    passes=("is_pass", "sum"),
    successful_passes=("successful_pass", "sum"),
    shots=("is_shot", "sum"),
    recoveries=("is_recovery", "sum"),
    losses=("is_lost", "sum"),
)

# tasa de acierto en pase = exitosos / intentos (si no hay intentos, NaN -> 0)
agg["pass_success_rate"] = (agg["successful_passes"] / agg["passes"]).replace([np.inf, np.nan], 0.0)

# ------------------------------------------------
# 5) Puntuación base (impact_score) antes de normalizar
#    Ajusta pesos si lo deseas.
# ------------------------------------------------
WEIGHT_PASS_SUCCESS = 1.0
WEIGHT_SHOT = 2.0
WEIGHT_RECOVERY = 1.5
WEIGHT_LOSS = -1.2

agg["impact_score"] = (
    agg["successful_passes"] * WEIGHT_PASS_SUCCESS
    + agg["shots"] * WEIGHT_SHOT
    + agg["recoveries"] * WEIGHT_RECOVERY
    + agg["losses"] * WEIGHT_LOSS
)

# ------------------------------------------------
# 6) Unir equipo y preparar tabla final
# ------------------------------------------------
player_stats = agg.copy()
if not player_team.empty:
    player_stats = player_stats.join(player_team, how="left")
else:
    player_stats["TeamOfPlayer"] = np.nan

# Limpieza de índice (jugador)
player_stats = player_stats.reset_index().rename(columns={"From": "Player"})

# ------------------------------------------------
# 7) Nota 0-10 normalizada POR EQUIPO (min-max intra-equipo)
#    Si todos tienen el mismo impacto en un equipo, se asigna 5.0 a todos.
# ------------------------------------------------
def min_max_0_10(s: pd.Series) -> pd.Series:
    s = s.astype(float)
    min_v = s.min()
    max_v = s.max()
    if pd.isna(min_v) or pd.isna(max_v) or max_v == min_v:
        # todos iguales -> dar 5 a todo el equipo
        return pd.Series(5.0, index=s.index)
    return (s - min_v) / (max_v - min_v) * 10.0

if player_stats["TeamOfPlayer"].notna().any():
    player_stats["rating_0_10"] = (
        player_stats.groupby("TeamOfPlayer")["impact_score"].transform(min_max_0_10).round(2)
    )
else:
    # si no tenemos equipos, normalizamos globalmente
    player_stats["rating_0_10"] = min_max_0_10(player_stats["impact_score"]).round(2)

# ------------------------------------------------
# 8) Ordenar y mostrar rankings por equipo
# ------------------------------------------------
def print_team_rankings(df: pd.DataFrame) -> None:
    if df["TeamOfPlayer"].notna().any():
        for team, g in df.groupby("TeamOfPlayer"):
            print(f"\n=== Ranking {team} (nota 0-10) ===")
            show = (
                g.sort_values(["rating_0_10", "impact_score"], ascending=False)
                 [["Player", "rating_0_10", "impact_score", "passes", "successful_passes",
                   "pass_success_rate", "shots", "recoveries", "losses", "total_events"]]
                 .reset_index(drop=True)
            )
            print(show.to_string(index=True))
    else:
        print("\n=== Ranking global (no se detectó columna de equipo) ===")
        show = (
            df.sort_values(["rating_0_10", "impact_score"], ascending=False)
              [["Player", "rating_0_10", "impact_score", "passes", "successful_passes",
                "pass_success_rate", "shots", "recoveries", "losses", "total_events"]]
              .reset_index(drop=True)
        )
        print(show.to_string(index=True))

print_team_rankings(player_stats)

# ------------------------------------------------
# 9) Guardar resultados a CSV
# ------------------------------------------------
player_stats_sorted = player_stats.sort_values(
    ["TeamOfPlayer", "rating_0_10", "impact_score"], ascending=[True, False, False]
)

player_stats_sorted.to_csv(os.path.join(OUTPUT_DIR, "player_ratings_by_team.csv"), index=False)

# Tabla “Top N” por equipo (opcional)
TOP_N = 5
if player_stats["TeamOfPlayer"].notna().any():
    top_by_team = (
        player_stats_sorted.groupby("TeamOfPlayer")
        .head(TOP_N)
        .reset_index(drop=True)
    )
    top_by_team.to_csv(os.path.join(OUTPUT_DIR, f"top_{TOP_N}_by_team.csv"), index=False)

# ------------------------------------------------
# 10) Visualización rápida (opcional): Top N por equipo
#      - Crea una figura por equipo con barras de la nota 0-10
# ------------------------------------------------
try:
    if player_stats["TeamOfPlayer"].notna().any():
        for team, g in player_stats.groupby("TeamOfPlayer"):
            g = g.sort_values("rating_0_10", ascending=False).head(TOP_N)
            if g.empty:
                continue
            plt.figure(figsize=(9, 5))
            plt.barh(g["Player"].astype(str), g["rating_0_10"])
            plt.gca().invert_yaxis()
            plt.title(f"Top {TOP_N} - {team} (nota 0-10)")
            plt.xlabel("Nota 0-10")
            plt.tight_layout()
            plt.show()
    else:
        g = player_stats.sort_values("rating_0_10", ascending=False).head(TOP_N)
        if not g.empty:
            plt.figure(figsize=(9, 5))
            plt.barh(g["Player"].astype(str), g["rating_0_10"])
            plt.gca().invert_yaxis()
            plt.title(f"Top {TOP_N} - Global (nota 0-10)")
            plt.xlabel("Nota 0-10")
            plt.tight_layout()
            plt.show()
except Exception as e:
    # La visualización no debería romper el análisis
    print(f"[Aviso] No se pudo generar la visualización: {e}")

# ------------------ FIN DEL SCRIPT ------------------
