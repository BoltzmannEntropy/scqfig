"""Four quantities across six devices on strips that share the device rows, each on its own log axis; filled markers are measured, open markers simulated, and n.r. marks a quantity a source does not report (synthetic numbers)."""
import scqfig as sq

TITLE = "Study strips"
FAMILY = "Comparison"


def build():
    devices = [f"Device {i}" for i in range(1, 7)]
    columns = {
        r"$|2\chi|/2\pi$" "\n(MHz)": {"Device 1": 1.2, "Device 2": 4.6, "Device 4": [3.0, (2.2, "simulated")],
                                      "Device 5": 0.9, "Device 6": 7.5},
        "storage\n" r"$T_1$ (µs)": {"Device 2": 900.0, "Device 3": 45.0, "Device 4": 1.2e4, "Device 6": 300.0},
        "efficiency\n" r"$\eta$": {"Device 1": 0.58, "Device 2": [0.41, (0.62, "simulated")], "Device 3": 0.88,
                                   "Device 5": 0.27, "Device 6": [0.46, 0.43]},
        "background\n" r"$\bar n$": {"Device 1": 3e-4, "Device 3": 1.1e-3, "Device 4": (5e-5, "simulated"),
                                     "Device 5": 7e-3, "Device 6": 2e-4},
    }
    fig, axes = sq.figure("page", aspect=0.3, ncols=4, sharey=True)
    sq.plots.study_strips(axes, devices, columns)
    return fig


if __name__ == "__main__":
    sq.demo_main(build, __file__)
