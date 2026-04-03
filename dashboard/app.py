"""
SemiYield Intelligence Platform - Streamlit Dashboard.

Navigation pages:
  1. Simulation       - Oxidation, implantation, etching, deposition
  2. Data Generator   - Synthetic fab data with drift and aging
  3. SPC Dashboard    - Control charts, Western Electric rules, Cpk
  4. Yield Prediction - Ensemble model training and SHAP explanations
  5. Process Optimizer- Bayesian process window optimization
  6. SPICE Export     - BSIM model card generation
"""

from __future__ import annotations

import io
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from semiyield.simulation import (
    DealGroveModel,
    IonImplantationModel,
    LangmuirHinshelwoodModel,
    CVDModel,
)
from semiyield.datagen import FabDataGenerator
from semiyield.spc import ControlChart, western_electric_violations, process_capability
from semiyield.models import YieldEnsemble, SHAPExplainer
from semiyield.doe import ProcessWindowOptimizer
from semiyield.spice import SPICEExporter

# ------------------------------------------------------------------ #
# Page configuration                                                   #
# ------------------------------------------------------------------ #

st.set_page_config(
    page_title="SemiYield Intelligence Platform",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------------ #
# Sidebar navigation                                                   #
# ------------------------------------------------------------------ #

PAGES = [
    "Simulation",
    "Data Generator",
    "SPC Dashboard",
    "Yield Prediction",
    "Process Optimizer",
    "SPICE Export",
]

with st.sidebar:
    st.title("SemiYield")
    st.markdown("AI-driven semiconductor process optimization")
    st.markdown("---")
    page = st.radio("Navigation", PAGES)


# ================================================================== #
# PAGE: Simulation                                                     #
# ================================================================== #

def page_simulation() -> None:
    st.title("Process Simulation")

    tab_ox, tab_impl, tab_etch, tab_dep = st.tabs(
        ["Oxidation", "Implantation", "Etching", "Deposition"]
    )

    # ---- Oxidation ----
    with tab_ox:
        st.subheader("Deal-Grove Thermal Oxidation")
        col1, col2 = st.columns(2)
        with col1:
            temp_ox = st.slider("Temperature (C)", 800, 1200, 1000, 10)
            atm_ox = st.selectbox("Atmosphere", ["dry", "wet"])
            t_max = st.slider("Max time (min)", 10, 300, 120, 10)
            init_thick = st.slider("Initial oxide thickness (nm)", 0.0, 5.0, 0.0, 0.1)
        with col2:
            st.markdown("**Deal-Grove model:**")
            st.latex(r"x^2 + Ax = B(t + \tau)")
            st.markdown("Solves for oxide thickness x(t) with Arrhenius rate constants.")

        model_ox = DealGroveModel()
        times = np.linspace(0, t_max, 200)
        thicknesses = model_ox.growth_curve(times, temp_ox, atm_ox, init_thick)
        rates = np.array([model_ox.rate(t, temp_ox, atm_ox, init_thick) for t in times])

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=times, y=thicknesses, name="Thickness (nm)", line=dict(color="#1f77b4", width=2)))
        fig.update_layout(
            title=f"Oxide Growth: {temp_ox}C, {atm_ox}",
            xaxis_title="Time (min)",
            yaxis_title="Oxide Thickness (nm)",
            height=350,
        )
        st.plotly_chart(fig, use_container_width=True)

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=times[1:], y=rates[1:], name="Growth rate", line=dict(color="#ff7f0e", width=2)))
        fig2.update_layout(
            title="Growth Rate vs Time",
            xaxis_title="Time (min)",
            yaxis_title="Rate (nm/min)",
            height=300,
        )
        st.plotly_chart(fig2, use_container_width=True)

        final_thickness = model_ox.grow(t_max, temp_ox, atm_ox, init_thick)
        st.metric("Final Thickness", f"{final_thickness:.2f} nm")

    # ---- Implantation ----
    with tab_impl:
        st.subheader("Ion Implantation Dopant Profile")
        col1, col2 = st.columns(2)
        with col1:
            species = st.selectbox("Ion species", ["boron", "phosphorus", "arsenic", "antimony"])
            energy = st.slider("Energy (keV)", 10, 300, 50, 5)
            dose = st.number_input("Dose (cm^-2)", value=1e13, format="%.2e", min_value=1e10, max_value=1e16)
            background = st.number_input("Background doping (cm^-3)", value=1e16, format="%.2e")
            dist_type = st.selectbox("Distribution model", ["gaussian", "pearsoniv"])
        with col2:
            st.markdown("**Gaussian profile:**")
            st.latex(r"N(x) = \frac{\Phi}{\sqrt{2\pi}\,\Delta R_p} \exp\!\left(-\frac{(x-R_p)^2}{2\Delta R_p^2}\right)")

        model_impl = IonImplantationModel(distribution=dist_type)
        depths = np.linspace(0, 600, 500)
        conc = model_impl.profile(depths, dose, energy, species)
        jd = model_impl.junction_depth(dose, energy, species, background)

        fig = go.Figure()
        mask = conc > 0
        fig.add_trace(go.Scatter(x=depths[mask], y=conc[mask], name=f"{species} profile", line=dict(color="#2ca02c", width=2)))
        if background > 0:
            fig.add_hline(y=background, line_dash="dash", line_color="red", annotation_text="Background")
        if jd > 0:
            fig.add_vline(x=jd, line_dash="dot", line_color="orange", annotation_text=f"xj={jd:.1f} nm")
        fig.update_layout(
            title=f"{species.capitalize()} implant at {energy} keV",
            xaxis_title="Depth (nm)",
            yaxis_title="Concentration (cm^-3)",
            yaxis_type="log",
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)
        st.metric("Junction Depth", f"{jd:.1f} nm" if jd > 0 else "N/A (dose below background)")

    # ---- Etching ----
    with tab_etch:
        st.subheader("Langmuir-Hinshelwood Etch Rate")
        col1, col2 = st.columns(2)
        with col1:
            etch_mode = st.selectbox("Etch mode", ["single", "two_reactant"])
            etch_material = st.selectbox("Material", ["SiO2", "Si", "Si3N4"])
            etch_pressure = st.slider("Pressure (mTorr)", 10, 500, 100, 10)
            etch_temp = st.slider("Temperature (C)", 20, 400, 50, 10)
            if etch_mode == "two_reactant":
                pressure_b = st.slider("O2 pressure (mTorr)", 5, 200, 50, 5)
            else:
                pressure_b = None
        with col2:
            st.markdown("**L-H kinetics:**")
            st.latex(r"R = k_s \frac{KP}{1+KP}")

        model_etch = LangmuirHinshelwoodModel(mode=etch_mode)

        temps_etch = np.linspace(20, 400, 100)
        rates_etch = [model_etch.rate(etch_pressure, t, etch_material, pressure_b) for t in temps_etch]

        pressures = np.linspace(10, 500, 100)
        rates_vs_p = [model_etch.rate(p, etch_temp, etch_material, pressure_b) for p in pressures]

        fig_e1 = go.Figure()
        fig_e1.add_trace(go.Scatter(x=temps_etch, y=rates_etch, name="Etch rate vs T", line=dict(color="#9467bd", width=2)))
        fig_e1.update_layout(xaxis_title="Temperature (C)", yaxis_title="Etch Rate (nm/min)", height=300,
                             title=f"{etch_material} etch rate vs temperature")
        st.plotly_chart(fig_e1, use_container_width=True)

        fig_e2 = go.Figure()
        fig_e2.add_trace(go.Scatter(x=pressures, y=rates_vs_p, name="Rate vs pressure", line=dict(color="#8c564b", width=2)))
        fig_e2.update_layout(xaxis_title="Pressure (mTorr)", yaxis_title="Etch Rate (nm/min)", height=300,
                             title=f"{etch_material} etch rate vs pressure at {etch_temp}C")
        st.plotly_chart(fig_e2, use_container_width=True)

        r_sio2 = model_etch.rate(etch_pressure, etch_temp, "SiO2", pressure_b)
        r_si = model_etch.rate(etch_pressure, etch_temp, "Si", pressure_b)
        sel = r_sio2 / r_si if r_si > 0 else float("inf")
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("SiO2 etch rate", f"{r_sio2:.1f} nm/min")
        col_b.metric("Si etch rate", f"{r_si:.1f} nm/min")
        col_c.metric("SiO2/Si selectivity", f"{sel:.2f}")

    # ---- Deposition ----
    with tab_dep:
        st.subheader("CVD/PVD Deposition")
        col1, col2 = st.columns(2)
        with col1:
            proc_type = st.selectbox("Process type", ["LPCVD", "PECVD", "ALD", "PVD"])
            mat_map = {"LPCVD": ["SiO2", "Si3N4", "poly_Si"],
                       "PECVD": ["SiO2", "Si3N4", "a_Si"],
                       "ALD":   ["HfO2", "Al2O3", "TiN"],
                       "PVD":   ["Al", "TiN", "W"]}
            dep_material = st.selectbox("Material", mat_map[proc_type])
            dep_temp = st.slider("Temperature (C)", 100, 900, 600, 25)
            dep_pressure = st.slider("Pressure (Torr)", 0.001, 1.0, 0.1, 0.001, format="%.3f")
            dep_time = st.slider("Time (min or cycles for ALD)", 1, 200, 30, 1)
            dep_ar = st.slider("Feature aspect ratio", 0.0, 10.0, 2.0, 0.5)
            dep_dt = st.slider("Temperature delta for stress (K)", -500, 500, -200, 25)
        with col2:
            st.markdown("**CVD growth rate:**")
            st.latex(r"R = C \cdot P \cdot \exp\!\left(-\frac{E_a}{k_B T}\right)")

        model_dep = CVDModel(process_type=proc_type, material=dep_material)
        thickness = model_dep.deposit(dep_time, dep_temp, dep_pressure)
        step_cov = model_dep.step_coverage(dep_ar)
        unif = model_dep.uniformity(300.0, dep_ar)

        try:
            stress = model_dep.stress(dep_dt, dep_material)
            stress_str = f"{stress:.0f} MPa"
        except ValueError:
            stress_str = "N/A"

        temps_dep = np.linspace(300, 900, 150)
        rates_dep = model_dep.rate_vs_temperature(temps_dep, dep_pressure)

        fig_d = go.Figure()
        fig_d.add_trace(go.Scatter(x=temps_dep, y=rates_dep, line=dict(color="#17becf", width=2)))
        fig_d.update_layout(
            title=f"{proc_type} {dep_material} growth rate vs temperature",
            xaxis_title="Temperature (C)", yaxis_title="Rate (nm/min or nm/cycle)",
            yaxis_type="log", height=350,
        )
        st.plotly_chart(fig_d, use_container_width=True)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Deposited thickness", f"{thickness:.2f} nm")
        c2.metric("Step coverage", f"{step_cov * 100:.1f}%")
        c3.metric("Non-uniformity (1s)", f"{unif:.2f}%")
        c4.metric("Film stress", stress_str)


# ================================================================== #
# PAGE: Data Generator                                                 #
# ================================================================== #

def page_datagen() -> None:
    st.title("Synthetic Fab Data Generator")

    col1, col2 = st.columns([1, 2])
    with col1:
        n_lots = st.slider("Number of lots", 10, 500, 100, 10)
        wafers_per_lot = st.slider("Wafers per lot", 5, 25, 25, 5)
        drift_rate = st.slider("Drift rate", 0.0, 0.3, 0.05, 0.01)
        aging_factor = st.slider("Aging factor", 0.0, 0.01, 0.002, 0.0005, format="%.4f")
        seed = st.number_input("Random seed", value=42, step=1)
        generate_btn = st.button("Generate Dataset", type="primary")

    if "fab_df" not in st.session_state:
        st.session_state.fab_df = None

    if generate_btn:
        with st.spinner("Generating synthetic fab data..."):
            gen = FabDataGenerator(
                seed=int(seed),
                drift_rate=drift_rate,
                aging_factor=aging_factor,
            )
            df = gen.generate(n_lots=n_lots, wafers_per_lot=wafers_per_lot)
            st.session_state.fab_df = df
            st.session_state.fab_gen = gen

    df = st.session_state.fab_df
    if df is None:
        st.info("Configure parameters and click Generate Dataset.")
        return

    with col2:
        st.success(f"Generated {len(df)} wafers across {df['lot_id'].nunique()} lots.")
        st.dataframe(df.drop(columns=["wafer_map"], errors="ignore").head(20), use_container_width=True)

    # Download button
    csv_buf = io.StringIO()
    df.drop(columns=["wafer_map"], errors="ignore").to_csv(csv_buf, index=False)
    st.download_button(
        "Download CSV",
        data=csv_buf.getvalue().encode(),
        file_name="fab_data.csv",
        mime="text/csv",
    )

    # Trend plots
    st.subheader("Process Parameter Trends")
    param_choice = st.selectbox("Parameter", [
        "gate_oxide_thickness", "poly_cd", "implant_dose",
        "anneal_temp", "metal_resistance", "contact_resistance", "yield"
    ])

    lot_means = df.groupby("lot_sequence")[param_choice].mean().reset_index()
    fig_trend = px.line(lot_means, x="lot_sequence", y=param_choice,
                        title=f"{param_choice} lot mean over time")
    st.plotly_chart(fig_trend, use_container_width=True)

    # Wafer map
    st.subheader("Wafer Map")
    wafer_ids = df["wafer_id"].tolist()
    selected_wafer = st.selectbox("Select wafer", wafer_ids[:50])
    if "fab_gen" in st.session_state:
        wmap = st.session_state.fab_gen.generate_wafer_map(selected_wafer)
        wmap_display = np.where(np.isnan(wmap), 0.0, wmap)
        fig_wmap = px.imshow(
            wmap_display,
            color_continuous_scale="RdYlGn",
            zmin=0, zmax=1,
            title=f"Local yield map: {selected_wafer}",
        )
        st.plotly_chart(fig_wmap, use_container_width=True)


# ================================================================== #
# PAGE: SPC Dashboard                                                  #
# ================================================================== #

def page_spc() -> None:
    st.title("SPC Dashboard")

    col1, col2 = st.columns([1, 3])
    with col1:
        data_source = st.radio("Data source", ["Use generated data", "Upload CSV"])

    df_spc: pd.DataFrame | None = None

    if data_source == "Upload CSV":
        uploaded = st.file_uploader("Upload process data CSV", type=["csv"])
        if uploaded:
            df_spc = pd.read_csv(uploaded)
    else:
        if st.session_state.get("fab_df") is not None:
            df_spc = st.session_state.fab_df
        else:
            st.warning("No generated data found. Go to the Data Generator page first.")
            return

    if df_spc is None:
        st.info("Provide data to continue.")
        return

    numeric_cols = df_spc.select_dtypes(include=[np.number]).columns.tolist()
    param_spc = st.selectbox("Parameter to chart", numeric_cols)

    data_arr = df_spc[param_spc].dropna().values
    if len(data_arr) < 10:
        st.warning("Insufficient data for SPC analysis (need at least 10 points).")
        return

    usl_default = float(np.percentile(data_arr, 99.5))
    lsl_default = float(np.percentile(data_arr, 0.5))

    col_usl, col_lsl = st.columns(2)
    usl = col_usl.number_input("USL", value=round(usl_default, 4))
    lsl = col_lsl.number_input("LSL", value=round(lsl_default, 4))

    # Fit I-MR chart
    chart = ControlChart(chart_type="IMR")
    chart.fit(data_arr)
    cd = chart.chart_data()

    violations = western_electric_violations(data_arr, cd["ucl"], cd["lcl"], cd["cl"])
    viol_idx = set(v[0] for v in violations)

    # Build I chart
    x_vals = list(range(len(data_arr)))
    colors = ["red" if i in viol_idx else "#1f77b4" for i in x_vals]

    fig_ic = go.Figure()
    fig_ic.add_trace(go.Scatter(
        x=x_vals, y=data_arr.tolist(),
        mode="lines+markers",
        marker=dict(color=colors, size=5),
        line=dict(color="#1f77b4"),
        name=param_spc,
    ))
    fig_ic.add_hline(y=cd["cl"], line_dash="solid", line_color="green", annotation_text="CL")
    fig_ic.add_hline(y=cd["ucl"], line_dash="dash", line_color="red", annotation_text="UCL")
    fig_ic.add_hline(y=cd["lcl"], line_dash="dash", line_color="red", annotation_text="LCL")
    fig_ic.update_layout(
        title=f"I-MR Chart: {param_spc}",
        xaxis_title="Observation",
        yaxis_title=param_spc,
        height=400,
    )
    st.plotly_chart(fig_ic, use_container_width=True)

    # Capability
    cap = process_capability(data_arr, usl, lsl)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Cp", f"{cap['Cp']:.3f}")
    c2.metric("Cpk", f"{cap['Cpk']:.3f}")
    c3.metric("Pp", f"{cap['Pp']:.3f}")
    c4.metric("Ppk", f"{cap['Ppk']:.3f}")
    st.metric("Sigma level", f"{cap['sigma_level']:.2f}")

    # Violations table
    if violations:
        st.subheader(f"Out-of-Control Signals ({len(violations)} found)")
        viol_df = pd.DataFrame(violations, columns=["Index", "Rule", "Description"])
        st.dataframe(viol_df, use_container_width=True)
    else:
        st.success("No Western Electric rule violations detected.")


# ================================================================== #
# PAGE: Yield Prediction                                               #
# ================================================================== #

def page_yield_prediction() -> None:
    st.title("Yield Prediction")

    if st.session_state.get("fab_df") is None:
        st.warning("Generate fab data first using the Data Generator page.")
        return

    df = st.session_state.fab_df
    feature_cols = [
        "gate_oxide_thickness", "poly_cd", "implant_dose",
        "anneal_temp", "metal_resistance", "contact_resistance",
        "etch_rate", "deposition_unif", "defect_density",
    ]
    available_features = [c for c in feature_cols if c in df.columns]
    target_col = "yield"

    if target_col not in df.columns:
        st.error("No 'yield' column found in generated data.")
        return

    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Training Configuration")
        test_frac = st.slider("Test set fraction", 0.1, 0.4, 0.2, 0.05)
        n_estimators = st.slider("Number of trees (RF/XGB)", 50, 500, 200, 50)
        train_btn = st.button("Train Ensemble Model", type="primary")

    if "yield_model" not in st.session_state:
        st.session_state.yield_model = None
        st.session_state.yield_metrics = None
        st.session_state.shap_vals = None

    if train_btn:
        with st.spinner("Training ensemble model..."):
            clean = df[available_features + [target_col]].dropna()
            X = clean[available_features].values
            y = clean[target_col].values

            n_test = max(10, int(len(X) * test_frac))
            X_train, X_test = X[:-n_test], X[-n_test:]
            y_train, y_test = y[:-n_test], y[-n_test:]

            val_split = max(5, int(len(X_train) * 0.15))
            X_tr, X_val = X_train[:-val_split], X_train[-val_split:]
            y_tr, y_val = y_train[:-val_split], y_train[-val_split:]

            model = YieldEnsemble(n_estimators=n_estimators, lstm_epochs=30, random_state=42)
            model.fit(X_tr, y_tr, X_val, y_val)
            metrics = model.score(X_test, y_test)
            st.session_state.yield_model = model
            st.session_state.yield_metrics = metrics
            st.session_state.yield_features = available_features
            st.session_state.yield_X_test = X_test

            explainer = SHAPExplainer()
            try:
                shap_vals = explainer.explain(model.rf, X_test, available_features)
                st.session_state.shap_vals = shap_vals
                st.session_state.top_features = explainer.top_features(
                    model.rf, X_test, available_features, n=len(available_features)
                )
            except Exception:
                st.session_state.shap_vals = None

    if st.session_state.yield_model is None:
        st.info("Configure parameters and click Train Ensemble Model.")
        return

    metrics = st.session_state.yield_metrics
    with col2:
        c1, c2, c3 = st.columns(3)
        c1.metric("R2", f"{metrics['R2']:.4f}")
        c2.metric("RMSE", f"{metrics['RMSE']:.4f}")
        c3.metric("MAE", f"{metrics['MAE']:.4f}")

    # SHAP importance chart
    if st.session_state.get("top_features"):
        st.subheader("Feature Importance (SHAP)")
        feat_names = [f[0] for f in st.session_state.top_features]
        feat_vals = [f[1] for f in st.session_state.top_features]
        fig_shap = go.Figure(go.Bar(
            x=feat_vals[::-1],
            y=feat_names[::-1],
            orientation="h",
            marker_color="#1f77b4",
        ))
        fig_shap.update_layout(
            xaxis_title="Mean |SHAP value|",
            title="Feature importance for yield prediction",
            height=350,
        )
        st.plotly_chart(fig_shap, use_container_width=True)

    # Manual prediction
    st.subheader("Predict Yield for Custom Process Parameters")
    input_vals = {}
    cols_pred = st.columns(3)
    features = st.session_state.yield_features
    df_clean = df[features].dropna()
    for i, feat in enumerate(features):
        col_i = cols_pred[i % 3]
        mn = float(df_clean[feat].min())
        mx = float(df_clean[feat].max())
        med = float(df_clean[feat].median())
        input_vals[feat] = col_i.number_input(feat, value=med, format="%.5g")

    X_pred = np.array([[input_vals[f] for f in features]])
    model = st.session_state.yield_model
    mean_pred, std_pred = model.predict_proba(X_pred)
    st.metric(
        "Predicted Yield",
        f"{float(mean_pred[0]):.4f}",
        delta=f"±{float(std_pred[0]):.4f} uncertainty"
    )


# ================================================================== #
# PAGE: Process Optimizer                                              #
# ================================================================== #

def page_optimizer() -> None:
    st.title("Bayesian Process Window Optimizer")

    st.markdown(
        "Define parameter bounds, then run Bayesian optimization to find the "
        "process conditions that maximise predicted yield."
    )

    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Parameter Bounds")
        ox_lo = st.number_input("Gate oxide time min (s)", value=50.0)
        ox_hi = st.number_input("Gate oxide time max (s)", value=200.0)
        ann_lo = st.number_input("Anneal temp min (C)", value=900.0)
        ann_hi = st.number_input("Anneal temp max (C)", value=1100.0)
        n_iter = st.slider("Optimization iterations", 10, 50, 20, 5)
        run_opt = st.button("Run Optimization", type="primary")

    if "opt_result" not in st.session_state:
        st.session_state.opt_result = None

    if run_opt:
        with st.spinner("Running Bayesian optimization..."):
            optimizer = ProcessWindowOptimizer(seed=42)
            optimizer.define_space({
                "gate_oxide_time": (ox_lo, ox_hi),
                "anneal_temp": (ann_lo, ann_hi),
            })

            # Objective: quadratic yield model (synthetic)
            # Maximum at (gate_oxide_time ~ 125, anneal_temp ~ 1000)
            def objective(x: np.ndarray) -> float:
                t, T = x[0], x[1]
                t_opt, T_opt = (ox_lo + ox_hi) / 2, (ann_lo + ann_hi) / 2
                t_range = (ox_hi - ox_lo) / 2
                T_range = (ann_hi - ann_lo) / 2
                val = 0.95 - 0.5 * ((t - t_opt) / t_range) ** 2 - 0.3 * ((T - T_opt) / T_range) ** 2
                return float(np.clip(val + np.random.normal(0, 0.02), 0, 1))

            result = optimizer.optimize(objective, n_iter=n_iter, n_init=5)
            result["optimizer"] = optimizer
            st.session_state.opt_result = result

    if st.session_state.opt_result is None:
        st.info("Configure bounds and click Run Optimization.")
        return

    result = st.session_state.opt_result
    optimizer = result["optimizer"]

    with col2:
        st.subheader("Optimization Results")
        bp = result["best_params"]
        c1, c2 = st.columns(2)
        for i, (k, v) in enumerate(bp.items()):
            (c1 if i % 2 == 0 else c2).metric(k, f"{v:.2f}")
        st.metric("Best Yield", f"{result['best_value']:.4f}")

    # Convergence plot
    st.subheader("Optimization Convergence")
    history_y = result["history_y"]
    best_so_far = np.maximum.accumulate(history_y)
    fig_conv = go.Figure()
    fig_conv.add_trace(go.Scatter(y=history_y, mode="markers", name="Observations", marker=dict(color="#aec7e8")))
    fig_conv.add_trace(go.Scatter(y=best_so_far, mode="lines", name="Best so far", line=dict(color="#1f77b4", width=2)))
    fig_conv.update_layout(
        xaxis_title="Iteration", yaxis_title="Yield",
        title="Bayesian optimization convergence",
        height=350,
    )
    st.plotly_chart(fig_conv, use_container_width=True)

    # Process window
    try:
        window = optimizer.process_window(confidence=0.95)
        st.subheader("Process Window (95% of optimum)")
        window_df = pd.DataFrame(
            [(k, f"{v[0]:.3g}", f"{v[1]:.3g}") for k, v in window.items()],
            columns=["Parameter", "Lower bound", "Upper bound"],
        )
        st.dataframe(window_df, use_container_width=True)
    except Exception as exc:
        st.warning(f"Process window computation unavailable: {exc}")


# ================================================================== #
# PAGE: SPICE Export                                                   #
# ================================================================== #

def page_spice() -> None:
    st.title("SPICE Model Card Export")

    st.markdown(
        "Convert process simulation parameters to BSIM3v3/BSIM4 SPICE model cards "
        "compatible with ngspice and LTspice."
    )

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Process Parameters")
        tox = st.slider("Gate oxide thickness (nm)", 1.0, 20.0, 8.5, 0.1)
        Lg = st.slider("Channel length (nm)", 28.0, 500.0, 90.0, 1.0)
        NA_exp = st.slider("Substrate doping (log10 cm^-3)", 14.0, 18.0, 17.0, 0.1)
        NA = 10 ** NA_exp
        xj = st.slider("Junction depth (nm)", 10.0, 200.0, 50.0, 1.0)
        model_level = st.selectbox("Model level", ["bsim3", "bsim4"])
        model_name = st.text_input("Model name", value="nmos_90nm")

    process_params = {
        "oxide_thickness_nm": tox,
        "channel_length_nm": Lg,
        "doping_concentration": NA,
        "junction_depth_nm": xj,
    }

    exporter = SPICEExporter(model_level=model_level)
    spice_params = exporter.process_to_spice(process_params, model_name)

    with col2:
        st.subheader("Computed SPICE Parameters")
        key_params = {
            "VTH0": spice_params["VTH0"],
            "TOX (m)": spice_params["TOXE"],
            "U0 (cm2/Vs)": spice_params["U0"],
            "K1": spice_params["K1"],
            "CDSC": spice_params["CDSC"],
            "RDSW": spice_params["RDSW"],
        }
        param_df = pd.DataFrame(
            [(k, f"{v:.4g}") for k, v in key_params.items()],
            columns=["Parameter", "Value"],
        )
        st.dataframe(param_df, use_container_width=True)

    # Full parameter table
    with st.expander("Show all BSIM parameters"):
        full_df = pd.DataFrame(
            [(k, f"{v:.6g}") for k, v in spice_params.items()],
            columns=["Parameter", "Value"],
        )
        st.dataframe(full_df, use_container_width=True)

    # File generation and download
    st.subheader("Download SPICE Files")
    col_a, col_b, col_c = st.columns(3)

    with col_a:
        with tempfile.NamedTemporaryFile(suffix=".subckt", delete=False) as tmp:
            exporter.write_subckt(process_params, model_name, tmp.name)
            content = Path(tmp.name).read_text()
        st.download_button("Download .subckt", content.encode(), f"{model_name}.subckt", "text/plain")
        st.code(content, language="spice")

    with col_b:
        with tempfile.NamedTemporaryFile(suffix=".lib", delete=False) as tmp:
            exporter.write_model_card(process_params, model_name, tmp.name)
            content = Path(tmp.name).read_text()
        st.download_button("Download .lib", content.encode(), f"{model_name}.lib", "text/plain")
        st.code(content, language="spice")

    with col_c:
        with tempfile.NamedTemporaryFile(suffix=".sp", delete=False) as tmp_tb:
            with tempfile.NamedTemporaryFile(suffix=".lib", delete=False) as tmp_mc:
                exporter.write_model_card(process_params, model_name, tmp_mc.name)
                exporter.write_testbench(model_name, tmp_tb.name)
            content = Path(tmp_tb.name).read_text()
        st.download_button("Download testbench", content.encode(), f"{model_name}_tb.sp", "text/plain")
        st.code(content, language="spice")


# ================================================================== #
# Router                                                               #
# ================================================================== #

if page == "Simulation":
    page_simulation()
elif page == "Data Generator":
    page_datagen()
elif page == "SPC Dashboard":
    page_spc()
elif page == "Yield Prediction":
    page_yield_prediction()
elif page == "Process Optimizer":
    page_optimizer()
elif page == "SPICE Export":
    page_spice()
