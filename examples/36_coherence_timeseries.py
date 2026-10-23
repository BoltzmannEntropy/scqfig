"""T1 and echo T2 measured every 10 minutes for 48 hours, with their histograms; T1 drops whenever a wandering defect drifts onto the qubit frequency (synthetic data)."""
import numpy as np

import scqfig as sq
from scqfig.plots.timedomain import tls_t1

TITLE = "Coherence over time"
FAMILY = "Time domain"


def build():
    rng = np.random.default_rng(36)
    dt = 1 / 6                                           # hours between measurements
    hours = np.arange(0, 48, dt)
    det = np.empty(hours.size)                           # defect detuning (MHz), Ornstein-Uhlenbeck
    det[0], tau, mu, sd = 2.5, 2.0, 2.2, 1.5
    for k in range(1, hours.size):
        det[k] = det[k - 1] + (mu - det[k - 1]) * dt / tau + sd * np.sqrt(2 * dt / tau) * rng.normal()
    fq = 5.200                                           # qubit frequency (GHz)
    T1 = tls_t1(fq, fq + 1e-3 * det[:, None], g=0.04, gamma=0.7, t1_bg=92.0) * rng.normal(1, 0.06, hours.size)
    Tphi = 190.0 * rng.lognormal(0, 0.12, hours.size)
    T2 = 1 / (1 / (2 * T1) + 1 / Tphi) * rng.normal(1, 0.05, hours.size)
    fig, (ax, axh) = sq.figure("column", aspect=0.66, ncols=2, sharey=True, gridspec_kw={"width_ratios": [4.2, 1]})
    sq.plots.coherence_timeseries(ax, axh, hours, T1, T2=T2)
    ax.set_ylim(0, 130)
    ax.set_xticks(range(0, 49, 12))
    return fig


if __name__ == "__main__":
    sq.demo_main(build, __file__)
