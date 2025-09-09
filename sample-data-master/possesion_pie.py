# ============================================================
# possession_pies_by_team.py — Posesión por jugador (Home/Away)
#   • Estima posesión por frames (jugador más cercano al balón)
#   • Normaliza coordenadas al campo 105x68
#   • Muestra dos gráficos circulares completos (no donut)
#     con % por jugador dentro de su equipo
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------------
# Configuración
# -------------------------------
HOME_PATH = "data/Sample_Game_1/Sample_Game_1_RawTrackingData_Home_Team.csv"
AWAY_PATH = "data/Sample_Game_1/Sample_Game_1_RawTrackingData_Away_Team.csv"
SKIPROWS = 2                  # en tus CSV, la fila 2 es el header real con PlayerN y Ball
SAMPLE_EVERY = 1              # submuestreo de frames (1=todos)
PITCH_W, PITCH_H = 105.0, 68.0

# Heurísticas de posesión (ajusta a tu dataset)
OWN_RADIUS_M = 1.5            # radio de control claro
LOOSE_RADIUS_M = 3.0          # radio de control probable (si balón va lento)
BALL_SPEED_THRESH = 10.0      # m/s; si más rápido, no asignamos control claro

# Mostrar etiquetas pequeñas o agrupar (opcional)
GROUP_OTHERS_BELOW_SECONDS = None  # p. ej., 8.0 para agrupar <8s como "Otros"; None para desactivar

# -------------------------------
# Utilidades de lectura
# -------------------------------
def load_tracking(path_csv: str, sample_every: int = 1) -> pd.DataFrame:
    df = pd.read_csv(path_csv, skiprows=SKIPROWS, low_memory=False)
    df.columns = df.columns.str.strip()
    if sample_every > 1:
        df = df.iloc[::sample_every].reset_index(drop=True)
    # Forzar numérico en columnas relevantes
    for c in df.columns:
        if c.startswith("Player") or c.startswith("Ball") or c in ("Time [s]", "Time[s]", "Time"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def list_player_xy_columns(df: pd.DataFrame):
    """
    Devuelve lista de tuplas (player_name, x_col, y_col) asumiendo que Y está a la derecha de X.
    """
    cols = df.columns.tolist()
    players = []
    for i, c in enumerate(cols):
        if isinstance(c, str) and c.startswith("Player"):
            if i + 1 < len(cols):
                players.append((c, c, cols[i + 1]))
    return players

def get_ball_xy(df: pd.DataFrame):
    if "Ball" not in df.columns:
        raise ValueError("No se encontró columna 'Ball' en el tracking.")
    i = df.columns.get_loc("Ball")
    if i + 1 >= len(df.columns):
        raise ValueError("No se encontró la columna Y del balón (Ball_Y).")
    return df.columns[i], df.columns[i + 1]

# -------------------------------
# Normalización a campo (105x68)
# -------------------------------
def normalize_pair_to_pitch(x: np.ndarray, y: np.ndarray):
    xmin, xmax = float(np.nanmin(x)), float(np.nanmax(x))
    ymin, ymax = float(np.nanmin(y)), float(np.nanmax(y))

    def minmax(a, lo, hi):
        amin, amax = float(np.nanmin(a)), float(np.nanmax(a))
        if not np.isfinite(amin) or not np.isfinite(amax) or amax == amin:
            return np.full_like(a, (lo + hi) / 2.0, dtype=np.float32)
        return (a - amin) / (amax - amin) * (hi - lo) + lo

    # Casos típicos
    if -1.5 <= xmin and xmax <= 1.5 and -1.5 <= ymin and ymax <= 1.5:
        x = (x + 1.0) * 0.5 * PITCH_W
        y = (y + 1.0) * 0.5 * PITCH_H
    elif -60 <= xmin <= -45 and 45 <= xmax <= 60 and -38 <= ymin <= -30 and 30 <= ymax <= 38:
        x = x + 52.5
        y = y + 34.0
    elif 0 <= xmin and 0 <= ymin and xmax <= 2 and ymax <= 2:
        x = minmax(x, 0.0, PITCH_W)
        y = minmax(y, 0.0, PITCH_H)
    elif 0 <= xmin <= PITCH_W and 0 <= ymin <= PITCH_H and xmax <= PITCH_W and ymax <= PITCH_H:
        pass  # ya en campo
    else:
        x = minmax(x, 0.0, PITCH_W)
        y = minmax(y, 0.0, PITCH_H)
    return x.astype(np.float32), y.astype(np.float32)

def normalize_tracking_inplace(df: pd.DataFrame, xy_cols: list[tuple[str, str]]):
    for cx, cy in xy_cols:
        x = pd.to_numeric(df[cx], errors="coerce").ffill().to_numpy(dtype=np.float32)
        y = pd.to_numeric(df[cy], errors="coerce").ffill().to_numpy(dtype=np.float32)
        x, y = normalize_pair_to_pitch(x, y)
        df[cx] = x
        df[cy] = y

# -------------------------------
# Cálculo de posesión
# -------------------------------
def estimate_possession_seconds(home_df: pd.DataFrame, away_df: pd.DataFrame) -> dict:
    """
    Devuelve dict {player_name: seconds} con la suma del tiempo de control estimado.
    """
    # Listas de jugadores y balón
    home_players = list_player_xy_columns(home_df)
    away_players = list_player_xy_columns(away_df)
    ball_x_col, ball_y_col = get_ball_xy(home_df)  # usamos el balón del Home (mismo tiempo)

    # Normalizar a metros (in-place)
    normalize_tracking_inplace(home_df, [(bx, by) for _, bx, by in home_players] + [(ball_x_col, ball_y_col)])
    normalize_tracking_inplace(away_df, [(bx, by) for _, bx, by in away_players])

    # Tiempo por frame
    time_col = "Time [s]" if "Time [s]" in home_df.columns else ("Time[s]" if "Time[s]" in home_df.columns else None)
    if time_col is not None:
        t = pd.to_numeric(home_df[time_col], errors="coerce").ffill().to_numpy()
        dt = np.diff(t, prepend=t[0])
        dt = np.clip(dt, 0.0, None)
        if np.all(dt == 0):
            dt = np.full(len(home_df), 0.04, dtype=float)
        else:
            zeros = dt == 0.0
            if np.any(zeros):
                dt[zeros] = np.median(dt[~zeros]) if np.any(~zeros) else 0.04
    else:
        dt = np.full(len(home_df), 0.04, dtype=float)  # ~25 Hz

    # Balón
    bx = pd.to_numeric(home_df[ball_x_col], errors="coerce").ffill().to_numpy(dtype=np.float32)
    by = pd.to_numeric(home_df[ball_y_col], errors="coerce").ffill().to_numpy(dtype=np.float32)
    v_ball = np.hypot(np.diff(bx, prepend=bx[0]), np.diff(by, prepend=by[0])) / np.clip(dt, 1e-6, None)

    # Players combinados (Home+Away)
    player_names = [p for p, _, _ in home_players] + [p for p, _, _ in away_players]
    player_x_cols = [x for _, x, _ in home_players] + [x for _, x, _ in away_players]
    player_y_cols = [y for _, _, y in home_players] + [y for _, _, y in away_players]

    poss_seconds = {p: 0.0 for p in player_names}

    # Iteración por frame
    n_frames = len(home_df)
    for i in range(n_frames):
        # Coord de todos los jugadores en frame i
        px = []
        py = []
        for cx, cy in zip(player_x_cols, player_y_cols):
            if cx in home_df.columns:
                xi = home_df[cx].iloc[i]; yi = home_df[cy].iloc[i]
            else:
                xi = away_df[cx].iloc[i] if cx in away_df.columns else np.nan
                yi = away_df[cy].iloc[i] if cy in away_df.columns else np.nan
            px.append(float(xi) if np.isfinite(xi) else np.nan)
            py.append(float(yi) if np.isfinite(yi) else np.nan)

        px = np.array(px, dtype=float)
        py = np.array(py, dtype=float)
        dists = np.hypot(px - bx[i], py - by[i])

        # Jugador más cercano
        nearest_idx = int(np.nanargmin(dists))
        nearest_dist = float(dists[nearest_idx])
        nearest_player = player_names[nearest_idx]

        # Heurística de control
        if v_ball[i] > BALL_SPEED_THRESH:
            continue  # balón viajando rápido
        if nearest_dist <= OWN_RADIUS_M:
            poss_seconds[nearest_player] += float(dt[i])
        elif nearest_dist <= LOOSE_RADIUS_M and v_ball[i] < BALL_SPEED_THRESH * 0.5:
            poss_seconds[nearest_player] += float(dt[i]) * 0.6

    return poss_seconds, [p for p, _, _ in home_players], [p for p, _, _ in away_players]

# -------------------------------
# Plot pies por equipo
# -------------------------------
def _group_small(labels, seconds, threshold):
    if threshold is None:
        return labels, seconds
    keep_L, keep_S = [], []
    others = 0.0
    for l, s in zip(labels, seconds):
        if s < threshold:
            others += s
        else:
            keep_L.append(l); keep_S.append(s)
    if others > 0:
        keep_L.append("Otros"); keep_S.append(others)
    return keep_L, keep_S

def plot_team_pies(poss_seconds: dict, home_players: list[str], away_players: list[str],
                   title_home="Posesión por jugador — Home",
                   title_away="Posesión por jugador — Away",
                   group_below_seconds: float | None = GROUP_OTHERS_BELOW_SECONDS):
    # Filtrar por equipo
    home_items = [(p, poss_seconds.get(p, 0.0)) for p in home_players]
    away_items = [(p, poss_seconds.get(p, 0.0)) for p in away_players]
    # Ordenar
    home_items.sort(key=lambda kv: kv[1], reverse=True)
    away_items.sort(key=lambda kv: kv[1], reverse=True)

    # Extraer y agrupar si procede
    h_labels = [p for p, s in home_items]
    h_secs = [s for p, s in home_items]
    a_labels = [p for p, s in away_items]
    a_secs = [s for p, s in away_items]

    h_labels, h_secs = _group_small(h_labels, h_secs, group_below_seconds)
    a_labels, a_secs = _group_small(a_labels, a_secs, group_below_seconds)

    # Totales por equipo para % internos
    h_total = sum(h_secs) if h_secs else 0.0
    a_total = sum(a_secs) if a_secs else 0.0

    # Plot side-by-side (círculos completos, con porcentajes)
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    # Home
    if h_total > 0:
        axes[0].pie(h_secs, labels=h_labels, autopct=lambda v: f"{v:.1f}%",
                    startangle=90, counterclock=False)
        axes[0].set_title(title_home)
    else:
        axes[0].text(0.5, 0.5, "Sin posesión detectada (Home)", ha="center", va="center")
        axes[0].axis("off")
    # Away
    if a_total > 0:
        axes[1].pie(a_secs, labels=a_labels, autopct=lambda v: f"{v:.1f}%",
                    startangle=90, counterclock=False)
        axes[1].set_title(title_away)
    else:
        axes[1].text(0.5, 0.5, "Sin posesión detectada (Away)", ha="center", va="center")
        axes[1].axis("off")

    plt.tight_layout()
    plt.show()

# -------------------------------
# Main
# -------------------------------
if __name__ == "__main__":
    home_df = load_tracking(HOME_PATH, sample_every=SAMPLE_EVERY)
    away_df = load_tracking(AWAY_PATH, sample_every=SAMPLE_EVERY)

    poss_seconds, home_players, away_players = estimate_possession_seconds(home_df, away_df)

    # Dos gráficos (uno por equipo), círculos completos con %
    plot_team_pies(
        poss_seconds,
        home_players, away_players,
        title_home="Posesión por jugador — Home",
        title_away="Posesión por jugador — Away",
        group_below_seconds=GROUP_OTHERS_BELOW_SECONDS
    )
