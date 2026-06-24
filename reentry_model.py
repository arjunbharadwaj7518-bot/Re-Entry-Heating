"""
reentry_model.py
----------------
Core physics for the re-entry heating simulation.
All equations extracted from the Colab notebook so the
Streamlit app can import them cleanly.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Tuple, List

# ── Physical constants ────────────────────────────────────────────────────────
R_AIR   = 287.05        # Specific gas constant for air  [J/(kg·K)]
GAMMA   = 1.4           # Ratio of specific heats        [-]
SIGMA   = 5.6704e-8     # Stefan-Boltzmann constant      [W/(m²·K⁴)]
G0      = 9.80665       # Standard gravity               [m/s²]
R_EARTH = 6.371e6       # Earth mean radius              [m]

# ── 5-layer exponential atmosphere (1976 US Std Atm, simplified) ──────────────
ATM_LAYERS = [
    {"h_base":     0, "h_top":  11_000, "rho0": 1.2250,   "H": 8_500},
    {"h_base": 11_000, "h_top":  25_000, "rho0": 0.3639,   "H": 6_341},
    {"h_base": 25_000, "h_top":  47_000, "rho0": 0.08803,  "H": 7_922},
    {"h_base": 47_000, "h_top":  86_000, "rho0": 0.001322, "H": 7_257},
    {"h_base": 86_000, "h_top": 200_000, "rho0": 5.46e-6,  "H": 5_877},
]


# ─────────────────────────────────────────────────────────────────────────────
# ATMOSPHERE MODEL
# ─────────────────────────────────────────────────────────────────────────────

def atmosphere(h: float) -> Tuple[float, float, float]:
    """
    5-layer exponential atmosphere.

    Parameters
    ----------
    h : altitude [m]

    Returns
    -------
    rho : density      [kg/m³]
    P   : pressure     [Pa]
    T   : temperature  [K]
    """
    h = max(h, 0.0)
    for layer in ATM_LAYERS:
        if h <= layer["h_top"]:
            rho = layer["rho0"] * np.exp(-(h - layer["h_base"]) / layer["H"])
            T   = max(216.65, 288.15 - 0.0065 * min(h, 11_000.0))
            P   = rho * R_AIR * T
            return rho, P, T
    return 1e-10, 1e-10, 186.0


def atmosphere_profile(h_max_km: float = 120.0, n: int = 500):
    """
    Return arrays of altitude, density, pressure, temperature
    for plotting the atmosphere profile.
    """
    alts = np.linspace(0, h_max_km * 1e3, n)
    rhos, Ps, Ts = zip(*[atmosphere(h) for h in alts])
    return alts / 1e3, np.array(rhos), np.array(Ps), np.array(Ts)


# ─────────────────────────────────────────────────────────────────────────────
# VEHICLE CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Vehicle:
    """
    Geometric and thermal properties of a re-entry vehicle.

    Attributes
    ----------
    name          : label used in plots / tables
    mass          : total vehicle mass                  [kg]
    nose_radius   : effective nose radius (Sutton-Graves) [m]
    ref_area      : aerodynamic reference area          [m²]
    Cd            : drag coefficient                    [-]
    emissivity    : TPS surface emissivity (0–1)        [-]
    wall_temp_ini : initial TPS wall temperature        [K]
    cp_tps        : TPS heat capacity                   [J/(kg·K)]
    mpa_tps       : TPS areal mass density              [kg/m²]
    beta           : ballistic coefficient (auto-computed) [kg/m²]
    """
    name          : str   = "Generic Capsule"
    mass          : float = 8_000.0
    nose_radius   : float = 1.5
    ref_area      : float = 7.07
    Cd            : float = 1.5
    emissivity    : float = 0.85
    wall_temp_ini : float = 300.0
    cp_tps        : float = 900.0
    mpa_tps       : float = 50.0
    beta          : float = field(init=False)

    def __post_init__(self):
        self.beta = self.mass / (self.Cd * self.ref_area)


# ─────────────────────────────────────────────────────────────────────────────
# HEAT TRANSFER EQUATIONS
# ─────────────────────────────────────────────────────────────────────────────

def sutton_graves(rho: float, V: float, Rn: float,
                  k_sg: float = 1.7415e-4) -> float:
    """
    Sutton-Graves stagnation-point convective heat flux.

        q_conv = k_SG * sqrt(rho / Rn) * V³   [W/m²]
    """
    return k_sg * np.sqrt(rho / Rn) * V**3


def radiative_cooling(T_wall: float, emissivity: float) -> float:
    """Stefan-Boltzmann surface radiation: q_rad = ε·σ·T⁴  [W/m²]"""
    return emissivity * SIGMA * T_wall**4


def update_wall_temp(q_conv: float, q_rad: float, T_wall: float,
                     dt: float, cp: float, mpa: float) -> float:
    """
    Forward-Euler wall temperature update from energy balance:
        (m/A)·cp·dT/dt = q_conv − q_rad
    """
    dT_dt = (q_conv - q_rad) / (mpa * cp)
    return T_wall + dt * dT_dt


def dynamic_pressure(rho: float, V: float) -> float:
    """q_dyn = ½·ρ·V²  [Pa]"""
    return 0.5 * rho * V**2


# ─────────────────────────────────────────────────────────────────────────────
# EQUATIONS OF MOTION
# ─────────────────────────────────────────────────────────────────────────────

def eom(h: float, V: float, vehicle: Vehicle,
        gamma: float) -> Tuple[float, float]:
    """
    Ballistic re-entry EOM on a non-rotating flat Earth.

        dh/dt = V·sin(γ)
        dV/dt = −D/m − g·sin(γ)

    gamma < 0 → vehicle descends.
    """
    h = max(h, 0.0)
    V = max(V, 1.0)
    rho, _, _ = atmosphere(h)
    g          = G0 * (R_EARTH / (R_EARTH + h))**2
    drag_accel = dynamic_pressure(rho, V) * vehicle.Cd * vehicle.ref_area / vehicle.mass
    dh_dt      = V * np.sin(gamma)
    dV_dt      = -drag_accel - g * np.sin(gamma)
    return dh_dt, dV_dt


# ─────────────────────────────────────────────────────────────────────────────
# RK4 INTEGRATOR
# ─────────────────────────────────────────────────────────────────────────────

def rk4_step(h: float, V: float, dt: float,
             vehicle: Vehicle, gamma: float) -> Tuple[float, float]:
    """Single 4th-order Runge-Kutta step."""
    def f(hi, Vi):
        return eom(hi, Vi, vehicle, gamma)

    dh1, dV1 = f(h, V)
    dh2, dV2 = f(h + 0.5*dt*dh1, V + 0.5*dt*dV1)
    dh3, dV3 = f(h + 0.5*dt*dh2, V + 0.5*dt*dV2)
    dh4, dV4 = f(h +     dt*dh3, V +     dt*dV3)

    h_new = h + (dt/6.0) * (dh1 + 2*dh2 + 2*dh3 + dh4)
    V_new = V + (dt/6.0) * (dV1 + 2*dV2 + 2*dV3 + dV4)
    return h_new, V_new


# ─────────────────────────────────────────────────────────────────────────────
# SIMULATION ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def run_simulation(vehicle: Vehicle,
                   h0: float        = 120e3,
                   V0: float        = 7_800.0,
                   gamma_deg: float = -6.5,
                   dt: float        = 2.0,
                   h_stop: float    = 5_000.0,
                   t_max: float     = 900.0) -> pd.DataFrame:
    """
    Time-march the re-entry from entry interface to near-ground.

    Parameters
    ----------
    vehicle   : Vehicle instance
    h0        : entry altitude                [m]
    V0        : entry speed                   [m/s]
    gamma_deg : flight-path angle (neg=down)  [deg]
    dt        : time step                     [s]
    h_stop    : halt altitude                 [m]
    t_max     : max simulation time           [s]

    Returns
    -------
    pd.DataFrame  one row per time step, columns:
        time [s], altitude [km], velocity [m/s], mach [-],
        density [kg/m3], q_conv [W/m2], q_rad [W/m2],
        T_wall [K], dyn_pressure [Pa], heat_load [MJ/m2]
    """
    gamma_rad = np.radians(gamma_deg)
    h, V      = float(h0), float(V0)
    T_wall    = vehicle.wall_temp_ini
    t         = 0.0
    records: List[dict] = []
    q_history: List[float] = []

    while h > h_stop and V > 200.0 and t < t_max:
        rho, _, T_atm = atmosphere(h)
        a_sound = np.sqrt(GAMMA * R_AIR * T_atm)
        mach    = V / a_sound

        q_conv  = sutton_graves(rho, V, vehicle.nose_radius)
        q_rad   = radiative_cooling(T_wall, vehicle.emissivity)
        q_dyn   = dynamic_pressure(rho, V)

        T_wall  = min(
            update_wall_temp(q_conv, q_rad, T_wall, dt,
                             vehicle.cp_tps, vehicle.mpa_tps),
            4_000.0
        )
        T_wall  = max(T_wall, 200.0)

        q_history.append(q_conv)
        Q_total = float(np.trapezoid(q_history) * dt / 1e6)

        records.append({
            "time [s]"          : round(t, 1),
            "altitude [km]"     : h / 1_000.0,
            "velocity [m/s]"    : V,
            "mach [-]"          : mach,
            "density [kg/m3]"   : rho,
            "q_conv [W/m2]"     : q_conv,
            "q_rad [W/m2]"      : q_rad,
            "T_wall [K]"        : T_wall,
            "dyn_pressure [Pa]" : q_dyn,
            "heat_load [MJ/m2]" : Q_total,
        })

        h, V = rk4_step(h, V, dt, vehicle, gamma_rad)
        t   += dt

    return pd.DataFrame(records)


def mission_summary(df: pd.DataFrame, label: str) -> dict:
    """Extract scalar peak/total metrics from a simulation DataFrame."""
    return {
        "Vehicle"                : label,
        "Entry Duration [s]"     : round(df["time [s]"].iloc[-1], 1),
        "Peak q_conv [MW/m²]"    : round(df["q_conv [W/m2]"].max()   / 1e6, 4),
        "Peak T_wall [K]"        : round(df["T_wall [K]"].max(), 1),
        "Peak Mach [-]"          : round(df["mach [-]"].max(), 2),
        "Peak DynPres [kPa]"     : round(df["dyn_pressure [Pa]"].max() / 1e3, 3),
        "Total Heat Load [MJ/m²]": round(df["heat_load [MJ/m2]"].iloc[-1], 3),
        "Final Altitude [km]"    : round(df["altitude [km]"].iloc[-1], 2),
    }


def run_parametric(vehicle: Vehicle,
                   angles: List[float],
                   h0: float = 120e3,
                   V0: float = 7_800.0,
                   dt: float = 2.0) -> Tuple[list, list]:
    """
    Run one simulation per entry angle.

    Returns
    -------
    summaries : list of summary dicts
    dataframes: list of DataFrames (same order as angles)
    """
    summaries, dataframes = [], []
    for ang in angles:
        df = run_simulation(vehicle, h0=h0, V0=V0, gamma_deg=ang,
                            dt=dt, t_max=900.0)
        summaries.append(mission_summary(df, f"{ang}°"))
        dataframes.append(df)
    return summaries, dataframes
