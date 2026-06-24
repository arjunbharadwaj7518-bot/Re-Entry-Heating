# 🚀 Re-Entry Heating Simulator

An interactive Streamlit web app that estimates **atmospheric re-entry heating** for a user-defined spacecraft, using the Sutton-Graves stagnation-point correlation, a simplified radiative equilibrium TPS model, and a 4th-order Runge-Kutta trajectory integrator.

---

## Live Demo

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-url.streamlit.app)

> Replace the badge URL after you deploy to [Streamlit Community Cloud](https://streamlit.io/cloud).

---

## Features

| Feature | Details |
|---|---|
| **Vehicle inputs** | Mass, nose radius, reference area, drag coefficient, TPS emissivity, heat capacity, areal mass |
| **Entry conditions** | Altitude (80–150 km), velocity (4 000–12 000 m/s), flight-path angle, time step |
| **Trajectory** | RK4 integration of ballistic equations of motion on flat Earth |
| **Heating** | Sutton-Graves convective flux + Stefan-Boltzmann radiative cooling |
| **Parametric study** | Entry-angle sweep with comparison plots and summary table |
| **Plots** | 6-panel interactive Plotly figure (altitude, velocity, heat flux, wall temp, heat load, Mach) |
| **Atmosphere** | 5-layer exponential atmosphere (0–200 km) |
| **Data export** | Download full simulation DataFrame as CSV |

---

## Physics Summary

### Sutton-Graves Heat Flux (stagnation point)
```
q_conv = k_SG * sqrt(ρ / Rn) * V³     [W/m²]

k_SG = 1.7415×10⁻⁴  (Earth air)
ρ    = freestream density   [kg/m³]
Rn   = nose radius          [m]
V    = vehicle speed        [m/s]
```

### TPS Wall Temperature (energy balance)
```
(m/A) · cp · dT/dt = q_conv − q_rad
q_rad = ε · σ · T_wall⁴
```

### Equations of Motion (flat-Earth ballistic)
```
dh/dt = V · sin(γ)
dV/dt = −D/m − g · sin(γ)

γ  = flight-path angle (negative = descending)
D/m = ½ρV²·Cd·A / m
```

---

## Repository Structure

```
reentry_app/
├── app.py                  # Streamlit application (UI + plots)
├── reentry_model.py        # Physics engine (importable module)
├── requirements.txt        # Python dependencies
├── .streamlit/
│   └── config.toml         # App theme (dark aerospace)
├── .gitignore
└── README.md
```

---

## Running Locally

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/reentry-heating-simulator.git
cd reentry-heating-simulator

# 2. Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

The app opens at **http://localhost:8501** in your browser.

---

## Deploying to Streamlit Community Cloud

1. Push this repository to GitHub (public repo).
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **New app** → select your repo → set **Main file path** to `app.py`.
4. Click **Deploy**. The app is live in ~60 seconds.
5. Copy the URL and update the badge at the top of this README.

---

## Input Parameters Reference

### Vehicle Configuration

| Parameter | Unit | Typical range | Effect |
|---|---|---|---|
| Mass | kg | 100 – 20 000 | Higher mass → higher ballistic coefficient → deeper penetration |
| Nose radius Rn | m | 0.05 – 8.0 | Larger Rn → lower peak heat flux (∝ 1/√Rn) |
| Reference area | m² | 0.01 – 100 | Sets aerodynamic drag |
| Drag coefficient Cd | — | 0.5 – 2.0 | Blunt bodies ≈ 1.5; slender ≈ 0.5 |
| Emissivity ε | — | 0.1 – 1.0 | Higher → more radiative cooling → lower T_wall |
| Initial wall temp | K | 200 – 500 | Starting TPS temperature |
| TPS heat capacity cp | J/(kg·K) | 400 – 2000 | AVCOAT ≈ 900; carbon-carbon ≈ 750 |
| TPS areal mass | kg/m² | 5 – 150 | Thermal inertia of the heat shield |

### Entry Conditions

| Parameter | Typical value | Notes |
|---|---|---|
| Entry altitude | 120 km | Standard entry interface |
| Entry velocity | 7 800 m/s | Low-Earth orbit return |
| Flight-path angle | −6.5° | Apollo used −6.5°; steeper = hotter |
| Time step | 2.0 s | Reduce to 0.5 s for steep entries |

---

## Limitations

- No ablation mass loss
- No real-gas / chemical non-equilibrium effects (important above Mach 15)
- Constant flight-path angle (no lift)
- Flat-Earth, non-rotating approximation
- 1D wall temperature (no through-thickness conduction)

---

## References

- Sutton, K. & Graves, R.A. (1971). *A general stagnation-point convective-heating equation for arbitrary gas mixtures.* NASA TR R-376.
- Anderson, J.D. (2006). *Hypersonic and High Temperature Gas Dynamics*, 2nd ed. AIAA.
- COESA (1976). *U.S. Standard Atmosphere 1976.* NOAA-S/T 76-1562.

---

## License

MIT — free to use, modify, and redistribute with attribution.
