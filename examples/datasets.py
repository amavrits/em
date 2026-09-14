"""Real datasets for the examples, embedded so the examples run offline.

Both are small, public, and long-standing benchmarks for mixture models, so
carrying the numbers here costs a few kilobytes and removes a download and a
dependency on ``pandas``/``statsmodels`` from every example run.

``OLD_FAITHFUL``
    272 eruptions of the Old Faithful geyser in Yellowstone, each recorded as
    (duration of the eruption, waiting time until the next one), both in
    minutes. From Azzalini and Bowman (1990), "A look at some data on the Old
    Faithful geyser", *Applied Statistics* 39, 357-365; distributed as R's
    ``datasets::faithful``. The two clear clusters -- short eruption followed
    by a short wait, long eruption followed by a long wait -- are correlated
    within each cluster, which is exactly what a full-covariance 2-D mixture
    is for.

``GALAXY_VELOCITIES``
    Radial velocities in km/s of 82 galaxies in a slice of the Corona Borealis
    region. Collected by Postman, Huchra and Geller (1986), *Astronomical
    Journal* 92, 1238; made a standard mixture-model benchmark by Roeder
    (1990), "Density estimation with confidence sets exemplified by superclus-
    ters and voids in the galaxies", *JASA* 85, 617-624; distributed as R's
    ``MASS::galaxies``. Velocity separates galaxies into superclusters split
    by voids, so the number of components is the scientific question, not a
    nuisance parameter -- how many superclusters this is remains disputed,
    with published answers from three to seven. (This copy uses 26690 for the
    78th observation; ``MASS`` long carried 26960 there, a transcription error
    relative to Roeder's table.)

    Note: it is Roeder's paper, not Dempster, Laird and Rubin (1977), that put
    these velocities on the map. The DLR paper predates the data by nine years
    and its own worked examples are different ones.
"""

from __future__ import annotations
import numpy as np

__all__ = ["OLD_FAITHFUL", "GALAXY_VELOCITIES", "old_faithful", "galaxy_velocities"]

# (eruption duration, waiting time until the next eruption), both in minutes.
OLD_FAITHFUL = np.array([
    (3.6, 79), (1.8, 54), (3.333, 74), (2.283, 62), (4.533, 85), (2.883, 55), (4.7, 88),
    (3.6, 85), (1.95, 51), (4.35, 85), (1.833, 54), (3.917, 84), (4.2, 78), (1.75, 47),
    (4.7, 83), (2.167, 52), (1.75, 62), (4.8, 84), (1.6, 52), (4.25, 79), (1.8, 51),
    (1.75, 47), (3.45, 78), (3.067, 69), (4.533, 74), (3.6, 83), (1.967, 55), (4.083, 76),
    (3.85, 78), (4.433, 79), (4.3, 73), (4.467, 77), (3.367, 66), (4.033, 80), (3.833, 74),
    (2.017, 52), (1.867, 48), (4.833, 80), (1.833, 59), (4.783, 90), (4.35, 80), (1.883, 58),
    (4.567, 84), (1.75, 58), (4.533, 73), (3.317, 83), (3.833, 64), (2.1, 53), (4.633, 82),
    (2, 59), (4.8, 75), (4.716, 90), (1.833, 54), (4.833, 80), (1.733, 54), (4.883, 83),
    (3.717, 71), (1.667, 64), (4.567, 77), (4.317, 81), (2.233, 59), (4.5, 84), (1.75, 48),
    (4.8, 82), (1.817, 60), (4.4, 92), (4.167, 78), (4.7, 78), (2.067, 65), (4.7, 73),
    (4.033, 82), (1.967, 56), (4.5, 79), (4, 71), (1.983, 62), (5.067, 76), (2.017, 60),
    (4.567, 78), (3.883, 76), (3.6, 83), (4.133, 75), (4.333, 82), (4.1, 70), (2.633, 65),
    (4.067, 73), (4.933, 88), (3.95, 76), (4.517, 80), (2.167, 48), (4, 86), (2.2, 60),
    (4.333, 90), (1.867, 50), (4.817, 78), (1.833, 63), (4.3, 72), (4.667, 84), (3.75, 75),
    (1.867, 51), (4.9, 82), (2.483, 62), (4.367, 88), (2.1, 49), (4.5, 83), (4.05, 81),
    (1.867, 47), (4.7, 84), (1.783, 52), (4.85, 86), (3.683, 81), (4.733, 75), (2.3, 59),
    (4.9, 89), (4.417, 79), (1.7, 59), (4.633, 81), (2.317, 50), (4.6, 85), (1.817, 59),
    (4.417, 87), (2.617, 53), (4.067, 69), (4.25, 77), (1.967, 56), (4.6, 88), (3.767, 81),
    (1.917, 45), (4.5, 82), (2.267, 55), (4.65, 90), (1.867, 45), (4.167, 83), (2.8, 56),
    (4.333, 89), (1.833, 46), (4.383, 82), (1.883, 51), (4.933, 86), (2.033, 53), (3.733, 79),
    (4.233, 81), (2.233, 60), (4.533, 82), (4.817, 77), (4.333, 76), (1.983, 59), (4.633, 80),
    (2.017, 49), (5.1, 96), (1.8, 53), (5.033, 77), (4, 77), (2.4, 65), (4.6, 81), (3.567, 71),
    (4, 70), (4.5, 81), (4.083, 93), (1.8, 53), (3.967, 89), (2.2, 45), (4.15, 86), (2, 58),
    (3.833, 78), (3.5, 66), (4.583, 76), (2.367, 63), (5, 88), (1.933, 52), (4.617, 93),
    (1.917, 49), (2.083, 57), (4.583, 77), (3.333, 68), (4.167, 81), (4.333, 81), (4.5, 73),
    (2.417, 50), (4, 85), (4.167, 74), (1.883, 55), (4.583, 77), (4.25, 83), (3.767, 83),
    (2.033, 51), (4.433, 78), (4.083, 84), (1.833, 46), (4.417, 83), (2.183, 55), (4.8, 81),
    (1.833, 57), (4.8, 76), (4.1, 84), (3.966, 77), (4.233, 81), (3.5, 87), (4.366, 77),
    (2.25, 51), (4.667, 78), (2.1, 60), (4.35, 82), (4.133, 91), (1.867, 53), (4.6, 78),
    (1.783, 46), (4.367, 77), (3.85, 84), (1.933, 49), (4.5, 83), (2.383, 71), (4.7, 80),
    (1.867, 49), (3.833, 75), (3.417, 64), (4.233, 76), (2.4, 53), (4.8, 94), (2, 55),
    (4.15, 76), (1.867, 50), (4.267, 82), (1.75, 54), (4.483, 75), (4, 78), (4.117, 79),
    (4.083, 78), (4.267, 78), (3.917, 70), (4.55, 79), (4.083, 70), (2.417, 54), (4.183, 86),
    (2.217, 50), (4.45, 90), (1.883, 54), (1.85, 54), (4.283, 77), (3.95, 79), (2.333, 64),
    (4.15, 75), (2.35, 47), (4.933, 86), (2.9, 63), (4.583, 85), (3.833, 82), (2.083, 57),
    (4.367, 82), (2.133, 67), (4.35, 74), (2.2, 54), (4.45, 83), (3.567, 73), (4.5, 73),
    (4.15, 88), (3.817, 80), (3.917, 71), (4.45, 83), (2, 56), (4.283, 79), (4.767, 78),
    (4.533, 84), (1.85, 58), (4.25, 83), (1.983, 43), (2.25, 60), (4.75, 75), (4.117, 81),
    (2.15, 46), (4.417, 90), (1.817, 46), (4.467, 74),
])

# Radial velocity in km/s.
GALAXY_VELOCITIES = np.array([
    9172, 9350, 9483, 9558, 9775, 10227, 10406, 16084, 16170, 18419, 18552, 18600, 18927,
    19052, 19070, 19330, 19343, 19349, 19440, 19473, 19529, 19541, 19547, 19663, 19846, 19856,
    19863, 19914, 19918, 19973, 19989, 20166, 20175, 20179, 20196, 20215, 20221, 20415, 20629,
    20795, 20821, 20846, 20875, 20986, 21137, 21492, 21701, 21814, 21921, 21960, 22185, 22209,
    22242, 22249, 22314, 22374, 22495, 22746, 22747, 22888, 22914, 23206, 23241, 23263, 23484,
    23538, 23542, 23666, 23706, 23711, 24129, 24285, 24289, 24366, 24717, 24990, 25633, 26690,
    26995, 32065, 32789, 34279,
], dtype=float)


def old_faithful() -> np.ndarray:
    """The 272 Old Faithful eruptions, shape ``(272, 2)``.

    Columns are (eruption duration, waiting time), both in minutes. The copy is
    fresh on every call, so a caller can scale or subset it in place.
    """
    return OLD_FAITHFUL.copy()


def galaxy_velocities(scale: float = 1000.0) -> np.ndarray:
    """The 82 galaxy velocities, shape ``(82,)``.

    Args:
        scale: Divisor applied to the km/s values. The default of 1000 gives
            the units this data is conventionally analysed and plotted in,
            which also keeps the fitted variances near 1 rather than near 1e6.
    """
    return GALAXY_VELOCITIES / float(scale)
