# SemiYield Intelligence Platform

[![CI](https://github.com/OutBlade/semiyield/actions/workflows/ci.yml/badge.svg)](https://github.com/OutBlade/semiyield/actions)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Live Demo](https://img.shields.io/badge/Live%20Demo-Cloudflare-orange)](https://semiyield.trycloudflare.com)

**Live Demo:** https://semiyield.trycloudflare.com *(available when host machine is running — served via Cloudflare Tunnel)*

SemiYield is an open-source Python platform that combines first-principles semiconductor process simulation, machine learning yield prediction, statistical process control, Bayesian design-of-experiments optimization, and SPICE model export into a single integrated toolkit. It targets process engineers, researchers, and students working on silicon device fabrication.


## Architecture

The platform is organized into six modules that reflect the natural flow of a semiconductor process integration workflow:

```
                        SemiYield Intelligence Platform
                        ================================

    Process Physics           Data & Models               Output
    ---------------           -------------               ------

    simulation/               datagen/                    spice/
    +------------+            +------------+              +----------+
    | DealGrove  |            | FabData    |              | SPICE    |
    | Implant    |  ------>   | Generator  |  ------>     | Exporter |
    | Etching    |            +------------+              +----------+
    | CVD/PVD    |                  |
    +------------+                  v
                              models/
                              +------------+
                              | Yield      |
                              | Ensemble   |
                              | (RF+XGB    |
                              |  +LSTM)    |
                              +------------+
                                    |
                         +----------+----------+
                         |                     |
                       spc/                  doe/
                  +----------+         +----------+
                  | Control  |         | Bayesian |
                  | Charts   |         | Optimizer|
                  | Cpk/Ppk  |         | GP + EI  |
                  +----------+         +----------+
                         |                     |
                         +---------+-----------+
                                   |
                             dashboard/
                          +-------------+
                          | Streamlit   |
                          | Web UI      |
                          +-------------+
```


## Installation

### From source (recommended for development)

```bash
git clone https://github.com/semiyield/semiyield.git
cd semiyield
pip install -e ".[dev]"
```

### Using pip

```bash
pip install semiyield
```

### Using Docker

```bash
docker compose up
```

The dashboard will be available at http://localhost:8501.


## Quick start

### Simulation

```python
from semiyield.simulation import DealGroveModel, IonImplantationModel

# Thermal oxidation: 1 hour dry O2 at 1000 C
model = DealGroveModel()
thickness_nm = model.grow(time=60, temperature=1000, atmosphere="dry")
print(f"Oxide thickness: {thickness_nm:.1f} nm")

# Boron implantation profile
import numpy as np
impl = IonImplantationModel(distribution="gaussian")
depths = np.linspace(0, 400, 500)  # nm
conc = impl.profile(depths, dose=1e13, energy=80, species="boron")
xj = impl.junction_depth(dose=1e13, energy=80, species="boron", background=1e16)
print(f"Junction depth: {xj:.1f} nm")
```

### Synthetic fab data generation

```python
from semiyield.datagen import FabDataGenerator

gen = FabDataGenerator(seed=42, drift_rate=0.05, aging_factor=0.002)
df = gen.generate(n_lots=100, wafers_per_lot=25)
print(df[["lot_id", "wafer_id", "yield", "defect_density"]].head())

wmap = gen.generate_wafer_map("LOT0001_W01")  # 2D numpy array
```

### Yield prediction

```python
from semiyield.models import YieldEnsemble, SHAPExplainer
import numpy as np

# Prepare features from fab data
feature_cols = ["gate_oxide_thickness", "poly_cd", "implant_dose",
                "anneal_temp", "metal_resistance", "contact_resistance"]
X = df[feature_cols].values
y = df["yield"].values

# Train ensemble
model = YieldEnsemble(n_estimators=200, lstm_epochs=50)
model.fit(X[:800], y[:800], X[800:900], y[800:900])

# Predict with uncertainty
mean, std = model.predict_proba(X[900:])
metrics = model.score(X[900:], y[900:])
print(f"R2={metrics['R2']:.4f}, RMSE={metrics['RMSE']:.4f}")

# SHAP feature importance
explainer = SHAPExplainer()
top = explainer.top_features(model.rf, X[900:], feature_cols, n=5)
for name, importance in top:
    print(f"  {name}: {importance:.4f}")
```

### Statistical Process Control

```python
from semiyield.spc import ControlChart, western_electric_violations, process_capability

# Fit an I-MR chart on historical data
chart = ControlChart(chart_type="IMR")
chart.fit(df["gate_oxide_thickness"].values)

# Check a new measurement
in_control = chart.update(8.4)  # True or False

# Detect Western Electric rule violations
data = df["poly_cd"].values
cd = chart.chart_data()
violations = western_electric_violations(data, cd["ucl"], cd["lcl"], cd["cl"])
for idx, rule, desc in violations:
    print(f"Violation at index {idx}: {desc}")

# Process capability
cap = process_capability(data, usl=100.0, lsl=80.0)
print(f"Cpk={cap['Cpk']:.3f}, Ppk={cap['Ppk']:.3f}")
```

### Bayesian process window optimization

```python
from semiyield.doe import ProcessWindowOptimizer

optimizer = ProcessWindowOptimizer(seed=0)
optimizer.define_space({
    "gate_oxide_time": (50, 200),    # seconds
    "anneal_temp": (900, 1100),      # degrees C
})

def my_yield_fn(x):
    # Replace with real measurement or model prediction
    return float(some_yield_model.predict(x.reshape(1, -1))[0])

result = optimizer.optimize(my_yield_fn, n_iter=50)
print("Best params:", result["best_params"])
print("Best yield:", result["best_value"])

window = optimizer.process_window(confidence=0.95)
for param, (lo, hi) in window.items():
    print(f"  {param}: [{lo:.1f}, {hi:.1f}]")
```

### SPICE model export

```python
from semiyield.spice import SPICEExporter

exporter = SPICEExporter(model_level="bsim3")
process = {
    "oxide_thickness_nm": 8.5,
    "channel_length_nm": 90.0,
    "doping_concentration": 1e17,
    "junction_depth_nm": 50.0,
}

params = exporter.process_to_spice(process, "nmos_90nm")
print(f"VTH0 = {params['VTH0']:.4f} V")
print(f"U0   = {params['U0']:.1f} cm2/Vs")

exporter.write_model_card(process, "nmos_90nm", "nmos_90nm.lib")
exporter.write_testbench("nmos_90nm", "testbench.sp")
```


## Module documentation

### semiyield.simulation

First-principles process simulation models.

`DealGroveModel` implements the Deal-Grove linear-parabolic oxidation equation with Arrhenius temperature-dependent rate constants for dry O2 and steam (H2O) ambients.

`IonImplantationModel` computes dopant concentration profiles using Gaussian and Pearson IV distributions with LSS-theory range parameters for boron, phosphorus, arsenic, and antimony in silicon.

`LangmuirHinshelwoodModel` models plasma etch rates via surface adsorption kinetics in both single-reactant (CF4) and two-reactant competitive (CHF3/O2) modes with Arrhenius temperature dependence.

`CVDModel` covers LPCVD, PECVD, ALD, and PVD deposition with growth rate, step coverage, non-uniformity, and biaxial film stress calculations.

### semiyield.datagen

`FabDataGenerator` produces synthetic lot-level and wafer-level process data with realistic lot-to-lot drift (random walk), equipment aging, wafer spatial non-uniformity, and Murphy's yield model.

### semiyield.models

`YieldEnsemble` stacks a RandomForestRegressor, XGBRegressor, and an LSTM neural network, with ensemble weights learned via Ridge regression. It provides point predictions and uncertainty estimates via RF variance and Monte-Carlo dropout.

`SHAPExplainer` wraps SHAP TreeExplainer and KernelExplainer, providing feature importance ranking and process sensitivity analysis.

### semiyield.spc

`ControlChart` supports Xbar-R, Xbar-S, I-MR, EWMA, and CUSUM charts with Phase I limit estimation and real-time Phase II updating.

`western_electric_violations` implements all eight Western Electric Handbook rules for run pattern detection.

`process_capability` computes Cp, Cpk, Pp, Ppk, and sigma level following the AIAG SPC Manual.

### semiyield.doe

`ProcessWindowOptimizer` uses a BoTorch SingleTaskGP surrogate with Expected Improvement acquisition for sample-efficient process parameter optimization, with a scipy fallback when BoTorch is unavailable.

### semiyield.spice

`SPICEExporter` converts physical process parameters to BSIM3v3 / BSIM4 SPICE model cards (threshold voltage from depletion approximation, mobility from Caughey-Thomas model, short-channel parameters), writing files compatible with ngspice and LTspice.


## Running the dashboard

```bash
streamlit run dashboard/app.py
```

Or with Docker:

```bash
docker compose up
```

Navigate to http://localhost:8501.


## Running tests

```bash
pytest
```

With coverage:

```bash
pytest --cov=semiyield --cov-report=html
```


## Technical background

### Deal-Grove oxidation model

The Deal-Grove model (1965) describes thermal silicon dioxide growth via a linear-parabolic equation derived from Fick's law of diffusion through the growing oxide layer combined with a surface reaction. The parabolic term dominates at large thickness (diffusion-limited); the linear term dominates at early times (reaction-limited). Rate constants B and B/A follow Arrhenius temperature dependence, and wet (steam) oxidation proceeds roughly 5-10x faster than dry oxidation due to the higher solubility and diffusivity of water in SiO2.

### Langmuir-Hinshelwood etch kinetics

The Langmuir-Hinshelwood mechanism models heterogeneous catalysis and surface reactions where reactant species first adsorb onto surface sites before reacting. The fractional surface coverage follows theta = KP/(1+KP). In plasma etching, the two-reactant competitive form captures the interplay between fluorocarbon polymer deposition (CHF3) and polymer removal (O2), which determines oxide-to-silicon selectivity.

### Murphy's yield model

Murphy's yield model (1964) gives the statistical yield of integrated circuits as a function of die area and random defect density. The model assumes a triangular distribution of defect densities across wafers and integrates over this distribution to obtain Y = ((1-exp(-AD))/(AD))^2. This predicts that yield decreases rapidly for large die or high defect density, motivating both defect reduction and die-size minimization.

### Bayesian optimization

Bayesian optimization replaces exhaustive design-of-experiments grids with a probabilistic surrogate model (Gaussian Process) that is cheap to evaluate. At each iteration, an acquisition function (Expected Improvement) balances exploration of uncertain regions and exploitation of known good regions, finding process optima in far fewer experiments than grid or random search. This is particularly valuable for semiconductor processes where each wafer run is expensive.


## Contributing

Contributions are welcome. Please open an issue to discuss proposed changes, fork the repository, create a feature branch, write tests for new functionality, and open a pull request.

Code style: black (line length 100), ruff linting. All physics formulas should include inline citations.


## License

MIT License. See LICENSE file for details.
