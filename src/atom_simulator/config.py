"""Edit these explicit constants to choose the default scene and production cases."""

from dataclasses import dataclass

from atom_simulator.physics import QuantumNumbers

# Default scene: full hydrogen state, natural time tau = t / t_mu.
N = 6
L = 2
M = 1
PARTICLE_COUNT = 450_000
RANDOM_SEED = 7
BEAD_RADIUS = 0.0035
BACKGROUND = "light"
WIDTH = 1920
HEIGHT = 1080
GALLERY_WIDTH = 3840
GALLERY_HEIGHT = 2160
FPS = 30
CLIP_SECONDS = 22.0
TIME_UNITS_PER_SECOND = 300.0
TRACER_COUNT = 96
TRAIL_VIDEO_SECONDS = 0.65


@dataclass(frozen=True)
class FlowCase:
    filename: str
    state: QuantumNumbers
    time_rate: float
    camera_distance: float


# The +m/-m pair deliberately uses the SAME seed, camera and time scale.
FLOW_CASES = (
    FlowCase("01_2p_m_plus1", QuantumNumbers(2, 1, 1), 4.0, 2.2),
    FlowCase("02_2p_m_minus1", QuantumNumbers(2, 1, -1), 4.0, 2.2),
    FlowCase("03_3d_m_plus2", QuantumNumbers(3, 2, 2), 14.0, 2.4),
    FlowCase("04_6d_m_plus1", QuantumNumbers(6, 2, 1), 300.0, 3.0),
)


def default_time_rate(state: QuantumNumbers) -> float:
    if state == QuantumNumbers(N, L, M):
        return TIME_UNITS_PER_SECOND
    for case in FLOW_CASES:
        if case.state == state:
            return case.time_rate
    rates_by_n = (
        0.0,
        1.0,
        4.0,
        14.0,
        50.0,
        120.0,
        300.0,
        450.0,
        800.0,
        1200.0,
        1800.0,
        2500.0,
        3500.0,
    )
    return rates_by_n[state.n]
