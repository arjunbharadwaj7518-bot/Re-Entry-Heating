"""
app.py — Re-Entry Heating Simulator (Streamlit)
================================================
Run locally:   streamlit run app.py
Deploy:        push repo to GitHub, connect to Streamlit Community Cloud
"""

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

from reentry_model import (
    Vehicle, run_simulation, mission_summary,
    run_parametric, atmosphere_profile,
)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Re-Entry Heating Simulator",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS  (minimal — keeps it readable)
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    /* metric card accent */
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, #0d1b2a 0%, #1a3a5c 100%);
        border-radius: 10px;
        padding: 12px 16px;
        border: 1px solid #2e6da4;
    }
    [data-testid="stMetricLabel"]  { color: #8ab4d4 !important; font-size: 0.78rem; }
    [data-testid="stMetricValue"]  { color: #e8f4fd !important; font-size: 1.35rem; }
    [data-testid="stMetricDelta"]  { color: #56b4e9 !important; }

    /* sidebar header */
    section[data-testid="stSidebar"] .block-container { padding-top: 1rem; }

    /* tab strip */
    .stTabs [data-baseweb="tab-list"] { gap: 6px; }
    .stTabs [data-baseweb="tab"]      { border-radius: 6px 6px 0 0; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PLOTLY THEME  (dark aerospace look)
# ─────────────────────────────────────────────────────────────────────────────

PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="#0d1b2a",
    plot_bgcolor="#0d1b2a",
    font=dict(family="Inter, Arial", color="#c8d8e8", size=12),
    margin=dict(l=50, r=20, t=40, b=50),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#2e6da4", borderwidth=1),
    xaxis=dict(gridcolor="#1e3a5a", zerolinecolor="#2e6da4"),
    yaxis=dict(gridcolor="#1e3a5a", zerolinecolor="#2e6da4"),
)

COLORS = {
    "primary"   : "#4a9eda",
    "secondary" : "#e06c75",
    "tertiary"  : "#98c379",
    "warning"   : "#e5c07b",
    "danger"    : "#be5046",
    "parametric": ["#313695","#4575b4","#74add1","#f46d43","#d73027"],
}


def apply_theme(fig: go.Figure, title: str = "") -> go.Figure:
    fig.update_layout(**PLOTLY_LAYOUT, title=dict(text=title, font_size=14))
    fig.update_xaxes(gridcolor="#1e3a5a", zerolinecolor="#2e6da4")
    fig.update_yaxes(gridcolor="#1e3a5a", zerolinecolor="#2e6da4")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR — USER INPUTS
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🛸 Re-Entry Simulator")
    st.markdown("---")

    # ── Vehicle parameters ──────────────────────────────────────────────────
    st.markdown("### 🔩 Vehicle Configuration")

    vehicle_name = st.text_input("Vehicle Name", value="My Spacecraft")

    col1, col2 = st.columns(2)
    with col1:
        mass         = st.number_input("Mass (kg)",         min_value=10.0,
                                       max_value=50_000.0,  value=5_800.0,  step=100.0)
        nose_radius  = st.number_input("Nose Radius Rn (m)", min_value=0.01,
                                       max_value=10.0,       value=4.7,      step=0.1)
        ref_area     = st.number_input("Ref. Area (m²)",    min_value=0.01,
                                       max_value=200.0,      value=12.0,     step=0.5)
        Cd           = st.number_input("Drag Coeff. Cd",    min_value=0.1,
                                       max_value=3.0,        value=1.5,      step=0.05)
    with col2:
        emissivity   = st.number_input("Emissivity ε",      min_value=0.01,
                                       max_value=1.0,        value=0.85,     step=0.01)
        wall_temp_ini= st.number_input("Init. Wall T (K)",  min_value=200.0,
                                       max_value=1000.0,     value=300.0,    step=10.0)
        cp_tps       = st.number_input("TPS cp (J/kg·K)",   min_value=100.0,
                                       max_value=3000.0,     value=900.0,    step=50.0)
        mpa_tps      = st.number_input("TPS Areal Mass (kg/m²)", min_value=1.0,
                                       max_value=200.0,      value=50.0,     step=1.0)

    st.markdown("---")

    # ── Entry conditions ────────────────────────────────────────────────────
    st.markdown("### 🌍 Entry Conditions")

    h0        = st.slider("Entry Altitude (km)",     min_value=80,   max_value=150,
                          value=120,  step=5) * 1_000.0
    V0        = st.slider("Entry Velocity (m/s)",    min_value=4_000, max_value=12_000,
                          value=7_800, step=100)
    gamma_deg = st.slider("Flight-Path Angle (°)",   min_value=-30,  max_value=-1,
                          value=-7,   step=1)
    dt        = st.selectbox("Time Step (s)", options=[0.5, 1.0, 2.0, 5.0], index=2)

    st.markdown("---")

    # ── Parametric study options ────────────────────────────────────────────
    st.markdown("### 📊 Parametric Study")
    run_param   = st.checkbox("Run entry-angle sweep", value=True)
    param_range = st.slider("Angle range (°)",
                            min_value=-30, max_value=-1,
                            value=(-20, -2), step=1)
    param_steps = st.slider("Number of angles", min_value=3, max_value=8, value=5)

    st.markdown("---")
    run_btn = st.button("▶  Run Simulation", type="primary", use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
# 🚀 Re-Entry Heating Simulator
**Simplified stagnation-point heating model using Sutton-Graves correlation + RK4 trajectory integration**
""")
st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# RUN SIMULATION (cached so re-renders don't re-run)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def cached_sim(name, mass, nose_radius, ref_area, Cd, emissivity,
               wall_temp_ini, cp_tps, mpa_tps,
               h0, V0, gamma_deg, dt):
    v = Vehicle(name=name, mass=mass, nose_radius=nose_radius,
                ref_area=ref_area, Cd=Cd, emissivity=emissivity,
                wall_temp_ini=wall_temp_ini, cp_tps=cp_tps, mpa_tps=mpa_tps)
    df = run_simulation(v, h0=h0, V0=V0, gamma_deg=gamma_deg, dt=dt)
    return v, df


@st.cache_data(show_spinner=False)
def cached_param(name, mass, nose_radius, ref_area, Cd, emissivity,
                 wall_temp_ini, cp_tps, mpa_tps,
                 h0, V0, angles_tuple, dt):
    v = Vehicle(name=name, mass=mass, nose_radius=nose_radius,
                ref_area=ref_area, Cd=Cd, emissivity=emissivity,
                wall_temp_ini=wall_temp_ini, cp_tps=cp_tps, mpa_tps=mpa_tps)
    summaries, dfs = run_parametric(v, list(angles_tuple), h0=h0, V0=V0, dt=dt)
    return summaries, dfs


# Trigger on button press (or first load)
if run_btn or "df_main" not in st.session_state:
    with st.spinner("Running trajectory + heat transfer simulation …"):
        vehicle, df_main = cached_sim(
            vehicle_name, mass, nose_radius, ref_area, Cd, emissivity,
            wall_temp_ini, cp_tps, mpa_tps,
            h0, V0, gamma_deg, dt
        )
        st.session_state["df_main"]  = df_main
        st.session_state["vehicle"]  = vehicle

        if run_param:
            angles = list(np.linspace(param_range[0], param_range[1],
                                      param_steps).round(1))
            param_summaries, param_dfs = cached_param(
                vehicle_name, mass, nose_radius, ref_area, Cd, emissivity,
                wall_temp_ini, cp_tps, mpa_tps,
                h0, V0, tuple(angles), dt
            )
            st.session_state["param_summaries"] = param_summaries
            st.session_state["param_dfs"]        = param_dfs
            st.session_state["param_angles"]     = angles

df      = st.session_state.get("df_main")
vehicle = st.session_state.get("vehicle")

if df is None or df.empty:
    st.warning("No simulation data yet — click ▶ Run Simulation in the sidebar.")
    st.stop()

summary = mission_summary(df, vehicle.name)

# ─────────────────────────────────────────────────────────────────────────────
# METRIC CARDS
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("### 📋 Mission Summary")
c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Peak Heat Flux",     f"{summary['Peak q_conv [MW/m²]']:.3f} MW/m²")
c2.metric("Peak Wall Temp",     f"{summary['Peak T_wall [K]']:.0f} K")
c3.metric("Peak Mach",          f"{summary['Peak Mach [-]']:.1f}")
c4.metric("Peak Dyn. Pressure", f"{summary['Peak DynPres [kPa]']:.2f} kPa")
c5.metric("Total Heat Load",    f"{summary['Total Heat Load [MJ/m²]']:.3f} MJ/m²")
c6.metric("Entry Duration",     f"{summary['Entry Duration [s]']:.0f} s")

# Ballistic coefficient callout
beta = vehicle.beta
st.info(
    f"**Ballistic Coefficient β = {beta:.1f} kg/m²**  "
    f"(mass={mass:.0f} kg, Cd={Cd}, A={ref_area} m²)  ·  "
    f"Nose radius Rn = {nose_radius} m  ·  "
    f"Entry angle γ = {gamma_deg}°"
)

st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Trajectory & Heating",
    "🌡️ Atmosphere Profile",
    "🔄 Parametric Study",
    "🗂️ Raw Data",
])


# ── TAB 1 : Main 6-panel figure ──────────────────────────────────────────────
with tab1:
    st.markdown(f"#### Simulation results for **{vehicle.name}**")

    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            "Altitude vs Time",
            "Velocity – Altitude (Phase Space)",
            "Stagnation Heat Flux",
            "TPS Wall Temperature",
            "Cumulative Heat Load",
            "Mach Number vs Altitude",
        ),
        vertical_spacing=0.12,
        horizontal_spacing=0.10,
    )

    t   = df["time [s]"]
    alt = df["altitude [km]"]

    # ── row 1 ──
    fig.add_trace(go.Scatter(x=t, y=alt, mode="lines",
                             line=dict(color=COLORS["primary"], width=2.5),
                             name="Altitude", showlegend=False), row=1, col=1)

    fig.add_trace(go.Scatter(x=df["velocity [m/s]"]/1000, y=alt, mode="lines",
                             line=dict(color=COLORS["secondary"], width=2.5),
                             name="Velocity", showlegend=False), row=1, col=2)

    # ── row 2 ──
    fig.add_trace(go.Scatter(x=t, y=df["q_conv [W/m2]"]/1e6, mode="lines",
                             line=dict(color=COLORS["warning"], width=2.5),
                             name="q_conv", showlegend=False), row=2, col=1)

    fig.add_trace(go.Scatter(x=t, y=df["T_wall [K]"], mode="lines",
                             line=dict(color=COLORS["danger"], width=2.5),
                             name="T_wall", showlegend=False), row=2, col=2)
    # Material limit lines on wall-temp panel
    fig.add_hline(y=1800, line_dash="dash", line_color="orange",
                  annotation_text="TUFI limit 1800 K",
                  annotation_font_color="orange", row=2, col=2)
    fig.add_hline(y=3000, line_dash="dash", line_color="red",
                  annotation_text="Carbon limit 3000 K",
                  annotation_font_color="red", row=2, col=2)

    # ── row 3 ──
    fig.add_trace(go.Scatter(x=t, y=df["heat_load [MJ/m2]"], mode="lines",
                             line=dict(color=COLORS["tertiary"], width=2.5),
                             name="Heat Load", showlegend=False), row=3, col=1)

    fig.add_trace(go.Scatter(x=df["mach [-]"], y=alt, mode="lines",
                             line=dict(color=COLORS["primary"], width=2.5),
                             name="Mach", showlegend=False), row=3, col=2)
    fig.add_vline(x=1.0, line_dash="dot", line_color="gray",
                  annotation_text="Mach 1", annotation_font_color="gray",
                  row=3, col=2)
    fig.add_vline(x=5.0, line_dash="dot", line_color="orange",
                  annotation_text="Mach 5", annotation_font_color="orange",
                  row=3, col=2)

    # Axis labels
    axis_labels = {
        "xaxis":  "Time [s]",          "yaxis":  "Altitude [km]",
        "xaxis2": "Velocity [km/s]",   "yaxis2": "Altitude [km]",
        "xaxis3": "Time [s]",          "yaxis3": "Heat Flux [MW/m²]",
        "xaxis4": "Time [s]",          "yaxis4": "Wall Temp [K]",
        "xaxis5": "Time [s]",          "yaxis5": "Heat Load [MJ/m²]",
        "xaxis6": "Mach [-]",          "yaxis6": "Altitude [km]",
    }
    for axis, label in axis_labels.items():
        fig.update_layout(**{f"{axis}_title_text": label})

    apply_theme(fig)
    fig.update_layout(height=850)
    st.plotly_chart(fig, use_container_width=True)


# ── TAB 2 : Atmosphere profile ───────────────────────────────────────────────
with tab2:
    st.markdown("#### Atmosphere Model Profile (0 – 120 km)")

    alts_km, rhos, Ps, Ts = atmosphere_profile(h_max_km=120, n=500)

    fig_atm = make_subplots(rows=1, cols=3,
                             subplot_titles=("Density [kg/m³]",
                                             "Pressure [Pa]",
                                             "Temperature [K]"))

    fig_atm.add_trace(go.Scatter(x=rhos, y=alts_km, mode="lines",
                                  line=dict(color=COLORS["primary"], width=2.5),
                                  showlegend=False), row=1, col=1)
    fig_atm.add_trace(go.Scatter(x=Ps,   y=alts_km, mode="lines",
                                  line=dict(color=COLORS["tertiary"], width=2.5),
                                  showlegend=False), row=1, col=2)
    fig_atm.add_trace(go.Scatter(x=Ts,   y=alts_km, mode="lines",
                                  line=dict(color=COLORS["warning"], width=2.5),
                                  showlegend=False), row=1, col=3)

    # Log-scale for density and pressure
    fig_atm.update_xaxes(type="log", row=1, col=1,
                          title_text="Density [kg/m³]")
    fig_atm.update_xaxes(type="log", row=1, col=2,
                          title_text="Pressure [Pa]")
    fig_atm.update_xaxes(title_text="Temperature [K]", row=1, col=3)
    for col in [1, 2, 3]:
        fig_atm.update_yaxes(title_text="Altitude [km]", row=1, col=col)

    # Mark the entry altitude
    entry_alt_km = h0 / 1_000
    for col in [1, 2, 3]:
        fig_atm.add_hline(y=entry_alt_km, line_dash="dash",
                           line_color="#4a9eda", opacity=0.7,
                           annotation_text=f"Entry ({entry_alt_km:.0f} km)",
                           annotation_font_color="#4a9eda",
                           row=1, col=col)

    apply_theme(fig_atm)
    fig_atm.update_layout(height=500)
    st.plotly_chart(fig_atm, use_container_width=True)


# ── TAB 3 : Parametric study ─────────────────────────────────────────────────
with tab3:
    if "param_dfs" not in st.session_state:
        st.info("Enable **Run entry-angle sweep** in the sidebar and click ▶ Run Simulation.")
    else:
        param_dfs    = st.session_state["param_dfs"]
        param_angles = st.session_state["param_angles"]
        param_sums   = st.session_state["param_summaries"]

        st.markdown("#### Entry Angle Sweep — Summary Table")
        param_df_display = pd.DataFrame([
            {k: v for k, v in s.items() if k != "Vehicle"} for s in param_sums
        ], index=[f"{a}°" for a in param_angles])

        # ── Colour-coded table via Plotly (no matplotlib dependency) ──────────
        numeric_cols = param_df_display.select_dtypes(include="number").columns.tolist()

        # Build per-column colour scaling: low=blue, high=red
        fill_colors = ["#1a3a5c"] * len(param_df_display)  # header row colour
        cell_fills  = []
        for col in param_df_display.columns:
            if col in numeric_cols:
                vals    = param_df_display[col].astype(float)
                lo, hi  = vals.min(), vals.max()
                rng     = hi - lo if hi != lo else 1.0
                normed  = (vals - lo) / rng          # 0 → 1
                # interpolate blue (#313695) → red (#d73027)
                hex_cols = []
                for n in normed:
                    r = int(49  + n * (215 - 49))
                    g = int(54  + n * (48  - 54))
                    b = int(149 + n * (39  - 149))
                    hex_cols.append(f"rgb({r},{g},{b})")
                cell_fills.append(hex_cols)
            else:
                cell_fills.append(["#0d1b2a"] * len(param_df_display))

        fig_tbl = go.Figure(go.Table(
            header=dict(
                values=["<b>Angle</b>"] + [f"<b>{c}</b>" for c in param_df_display.columns],
                fill_color="#1a3a5c",
                font=dict(color="white", size=11),
                align="center",
                line_color="#2e6da4",
            ),
            cells=dict(
                values=[param_df_display.index.tolist()] +
                       [param_df_display[c].round(3).tolist()
                        for c in param_df_display.columns],
                fill_color=["#0d2a45"] + cell_fills,
                font=dict(color="white", size=11),
                align="center",
                line_color="#1e3a5a",
                height=28,
            ),
        ))
        fig_tbl.update_layout(
            **PLOTLY_LAYOUT,
            height=60 + 30 * len(param_df_display),
            margin=dict(l=0, r=0, t=10, b=0),
        )
        st.plotly_chart(fig_tbl, use_container_width=True)

        st.markdown("#### Entry Angle Sweep — Plots")

        fig_p = make_subplots(rows=1, cols=3,
                               subplot_titles=("Heat Flux vs Time",
                                               "Wall Temp vs Time",
                                               "Heat Load vs Time"))
        for i, (ang, df_p) in enumerate(zip(param_angles, param_dfs)):
            col_hex = COLORS["parametric"][i % len(COLORS["parametric"])]
            label   = f"{ang}°"
            t_p     = df_p["time [s]"]
            kw      = dict(mode="lines", name=label,
                           line=dict(color=col_hex, width=2),
                           legendgroup=label)

            fig_p.add_trace(go.Scatter(x=t_p, y=df_p["q_conv [W/m2]"]/1e6,
                                        **kw), row=1, col=1)
            kw2 = {**kw, "showlegend": False}
            fig_p.add_trace(go.Scatter(x=t_p, y=df_p["T_wall [K]"],
                                        **kw2), row=1, col=2)
            fig_p.add_trace(go.Scatter(x=t_p, y=df_p["heat_load [MJ/m2]"],
                                        **kw2), row=1, col=3)

        fig_p.update_xaxes(title_text="Time [s]")
        fig_p.update_yaxes(title_text="Heat Flux [MW/m²]", row=1, col=1)
        fig_p.update_yaxes(title_text="Wall Temp [K]",      row=1, col=2)
        fig_p.update_yaxes(title_text="Heat Load [MJ/m²]",  row=1, col=3)

        apply_theme(fig_p)
        fig_p.update_layout(height=480)
        st.plotly_chart(fig_p, use_container_width=True)


# ── TAB 4 : Raw data ─────────────────────────────────────────────────────────
with tab4:
    st.markdown(f"#### Full Simulation DataFrame — {len(df)} rows")

    # Column selector
    all_cols = df.columns.tolist()
    sel_cols = st.multiselect("Columns to display", all_cols,
                               default=all_cols)
    st.dataframe(df[sel_cols], use_container_width=True, height=450)

    # Download buttons
    csv_bytes = df.to_csv(index=False).encode()
    st.download_button(
        label="⬇️ Download CSV",
        data=csv_bytes,
        file_name=f"{vehicle_name.replace(' ','_')}_reentry_data.csv",
        mime="text/csv",
    )

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown(
    "<small style='color:#4a6fa5'>Re-Entry Heating Simulator · "
    "Sutton-Graves (1971) correlation · RK4 trajectory integrator · "
    "5-layer exponential atmosphere · Built with Streamlit</small>",
    unsafe_allow_html=True,
)
