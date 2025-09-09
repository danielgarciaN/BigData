# ==============================================
# heatmap_fast.py — Mapa de calor rápido (formato real de tracking)
# ==============================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------------
# Configuración
# -------------------------------
TRACKING_HOME_PATH = "data/Sample_Game_1/Sample_Game_1_RawTrackingData_Away_Team.csv"
PLAYER_NUMBER = 21          # jugador a visualizar (1-14)
SAMPLE_EVERY = 2            # submuestreo de frames (1=todo, 2=la mitad, etc.)
GRID_RES_M = 1
GAUSS_SIGMA = 1.2
ALPHA = 0.9
CMAP = "Reds"

# -------------------------------
# Funciones auxiliares
# -------------------------------
def load_player_tracking(path_csv: str, player: str, sample_every: int = 1):
    # Ahora fila 2 será el header real
    df = pd.read_csv(path_csv, skiprows=2)
    df.columns = df.columns.str.strip()

    if player not in df.columns:
        raise ValueError(f"No se encontró '{player}' en {path_csv}")

    idx = df.columns.get_loc(player)
    x = pd.to_numeric(df.iloc[:, idx], errors="coerce").ffill().to_numpy()
    y = pd.to_numeric(df.iloc[:, idx+1], errors="coerce").ffill().to_numpy()

    if sample_every > 1:
        x = x[::sample_every]
        y = y[::sample_every]

    return x.astype(np.float32), y.astype(np.float32)


def normalize_to_pitch(x: np.ndarray, y: np.ndarray):
    xmin, xmax = np.nanmin(x), np.nanmax(x)
    ymin, ymax = np.nanmin(y), np.nanmax(y)

    if xmax <= 1.5 and xmin >= -1.5:
        x = (x + 1) * 0.5 * 105
        y = (y + 1) * 0.5 * 68
    elif xmax <= 52.6 and xmin >= -52.6:
        x = x + 52.5
        y = y + 34
    elif xmax <= 105 and xmin >= 0:
        pass
    else:
        x = (x - xmin) / (xmax - xmin) * 105
        y = (y - ymin) / (ymax - ymin) * 68

    return x, y

def draw_pitch(ax=None, lw=1.2, color="#222"):
    if ax is None:
        ax = plt.gca()
    ax.plot([0, 0], [0, 68], color=color, lw=lw)
    ax.plot([105, 105], [0, 68], color=color, lw=lw)
    ax.plot([0, 105], [0, 0], color=color, lw=lw)
    ax.plot([0, 105], [68, 68], color=color, lw=lw)
    ax.plot([52.5, 52.5], [0, 68], color=color, lw=lw)
    centre = plt.Circle((52.5, 34), 9.15, color=color, fill=False, lw=lw)
    ax.add_patch(centre)
    ax.set_xlim(0, 105)
    ax.set_ylim(0, 68)
    ax.set_aspect("equal")
    ax.axis("off")
    return ax

def gaussian_blur(img, sigma):
    if sigma <= 0:
        return img
    try:
        from scipy.ndimage import gaussian_filter
        return gaussian_filter(img, sigma=sigma, mode="nearest")
    except ImportError:
        return img  # sin blur si no hay SciPy

def fast_heatmap(x, y, grid_res_m=1, sigma=1.2):
    nx_bins = int(105 // grid_res_m) + 1
    ny_bins = int(68 // grid_res_m) + 1

    H, _, _ = np.histogram2d(x, y, bins=[nx_bins, ny_bins], range=[[0, 105], [0, 68]])
    H = gaussian_blur(H.astype(np.float32), sigma)
    if H.max() > 0:
        H /= H.max()
    return H.T, [0, 105, 0, 68]

# -------------------------------
# Pipeline
# -------------------------------
if __name__ == "__main__":
    player = f"Player{PLAYER_NUMBER}"
    x, y = load_player_tracking(TRACKING_HOME_PATH, player, sample_every=SAMPLE_EVERY)
    x, y = normalize_to_pitch(x, y)

    H, extent = fast_heatmap(x, y, GRID_RES_M, GAUSS_SIGMA)

    # distancia recorrida
    dx, dy = np.diff(x), np.diff(y)
    total_distance = np.sum(np.hypot(dx, dy))
    print(f"Distancia recorrida {player}: {total_distance:.2f} m aprox.")

    # plot
    fig, ax = plt.subplots(figsize=(12, 8))
    draw_pitch(ax)
    im = ax.imshow(H, extent=extent, origin="lower", cmap=CMAP, interpolation="bilinear", alpha=ALPHA)
    plt.colorbar(im, ax=ax, fraction=0.030, pad=0.02).set_label("Intensidad")
    ax.set_title(f"Mapa de calor rápido — {player} (Home)")
    plt.tight_layout()
    plt.show()
