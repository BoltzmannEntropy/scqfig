"""T1 against qubit frequency over 24 hours; five defects show as Lorentzian T1 dips whose frequencies wander, and one switches between two frequencies (synthetic data)."""
import numpy as np

import scqfig as sq
from scqfig.plots.timedomain import tls_t1

TITLE = "TLS spectral diffusion"
FAMILY = "Time domain"


def build():
    rng = np.random.default_rng(37)
    hours = np.linspace(0, 24, 97)
    freq = np.linspace(5.10, 5.30, 201)                  # GHz
    f0 = np.array([5.122, 5.163, 5.191, 5.243, 5.271])   # defect centres (GHz)
    drift = np.cumsum(rng.normal(0, 0.7, (hours.size, f0.size)), axis=0)        # random walk (MHz)
    drift += 3.5 * np.sin(2 * np.pi * hours[:, None] / rng.uniform(8, 20, f0.size) + rng.uniform(0, 6, f0.size))
    state = np.cumsum(rng.random(hours.size) < 0.06) % 2                        # telegraphic switching
    drift[:, 3] += 9.0 * state
    tracks = f0 + 1e-3 * drift                           # shape (time, defect)
    g = np.array([0.09, 0.06, 0.12, 0.07, 0.08])         # coupling g/2pi (MHz)
    gamma = np.array([1.2, 0.8, 1.5, 1.0, 1.2])          # defect decoherence gamma/2pi (MHz)
    T1 = tls_t1(freq[:, None], tracks, g=g, gamma=gamma, t1_bg=80.0)
    T1 = T1 * rng.normal(1, 0.07, T1.shape)
    fig, ax = sq.figure("column", aspect=0.78)
    sq.plots.tls_map(ax, hours, freq, T1, vmin=0, vmax=100)
    ax.set_xticks(range(0, 25, 6))
    ax.set_yticks(np.arange(5.10, 5.301, 0.05))
    return fig


if __name__ == "__main__":
    sq.demo_main(build, __file__)
