# Astro Sentinel 🛰️
### Automated Test Equipment (ATE) Burn-In Screening & Fault Diagnostics Console

A mission-assurance QA platform engineered for space-grade semiconductor qualification. **Astro Sentinel** combines population anomaly detection (**AEC-Q001 DPAT**) with physical wear kinetics (**Arrhenius Acceleration & HistGradientBoosting**) to detect latent chip defects early, saving up to **144 chamber hours per aborted unit** under **MIL-STD-883 Method 1015** protocols.

---

## ⚡ Core Features

- **Module A (Population Screening):** Identifies lot mavericks across wafer sensor vectors (`secom.data`) via robust Median Absolute Deviation (**MAD**) and `IsolationForest`.
- **Module B (Degradation Kinetics):** Ingests empirical NASA SMU sweeps (`Turn On.csv`, `Breakdown.csv`, `LeakageIV.csv`) to project 168-hour wear from 24-hour readouts.
- **Automated 24h Early Abort:** Cuts power to runaway devices at 24 hours to prevent chamber damage and save facility capacity.
- **Explainable QA Audit:** Outlines root-cause physical and statistical drivers for every `PASS`, `FLAGGED`, or `EARLY REJECT` disposition.
- **Flight Certificate Export:** Generates an auditable, timestamped CSV qualification certificate.
- **Interactive UI:** Glassmorphism dashboard featuring real-time telemetry replay and dynamic stress parameter sliders[cite: 1].

---

## 🏗️ Architecture
Raw ATE Telemetry
                       │
   ┌───────────────────┴───────────────────┐
   ▼                                       ▼
Module A (AEC-Q001)                    Module B (Physics)
Wafer Maverick Screening               Thermal Wear Extrapolation
(secom.data + IsolationForest)         (NASA Sweeps + HistGradBoost)
│                                       │
└───────────────────┬───────────────────┘
▼
Tri-State Gate Logic
│
┌──────────────────┼──────────────────┐
▼                  ▼                  ▼
[ PASS ]          [ FLAGGED ]       [ EARLY REJECT ]
Flight-Ready       Lot Maverick        Aborted at 24h
Payload Qualified  Manual QA Review     Saves 144 Hours


---

## 📊 Ingested Datasets

| Source | File | Purpose |
| :--- | :--- | :--- |
| **Wafer Lot Telemetry** | `secom.data` | 1,567 IC vectors for population baseline modeling[cite: 1]. |
| **NASA Breakdown Sweep** | `Breakdown.csv` | Empirical junction breakdown ceiling ($V_{br} = 114.80\text{ V}$)[cite: 1]. |
| **NASA Leakage Sweep** | `LeakageIV.csv` | Sub-nanoamp quiescent baseline ($I_{leak} = 4.792\text{ nA}$)[cite: 1]. |
| **NASA Conduction Sweep** | `Turn On.csv` | Threshold voltage calibration knee ($V_{th} = 2.814\text{ V}$)[cite: 1]. |

---

## 🚀 Quick Start

```bash
# 1. Clone the repository
git clone [https://github.com/BhuvaneswariK8/Astro-Sentinel.git](https://github.com/BhuvaneswariK8/Astro-Sentinel.git)
cd Astro-Sentinel

# 2. Activate virtual environment
# Windows (PowerShell):
.\env\Scripts\Activate.ps1
# Linux/macOS:
source env/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch console
streamlit run astro.py
🎛️ Live Demo Guide
Adjust Stress Sliders: Modulate chamber temperature (85 
∘
 C→150 
∘
 C) to see Arrhenius kinetics shift failure rates in real time[cite: 1].

Replay Chamber Telemetry: Click ▶ Trigger 24h Stress Run to watch automated shutdown trigger on a failing chip[cite: 1].

Inspect Causal Drivers: Open the Engineer's Diagnostics Console to inspect physical root causes for flagged or rejected units.

Export Flight Certificate: Click 📄 Download Official Flight Acceptance Certificate (CSV) to download the qualification log[cite: 1].

📜 Standards & Methodologies
MIL-STD-883 Method 1015: Burn-in procedures, test durations (168h), and thermal equivalence[cite: 1].

AEC-Q001: Statistical outlier identification via Median Absolute Deviation (MAD) Part Average Testing (DPAT)[cite: 1].

Arrhenius Kinetics: Thermal acceleration formulation (E 
a
​
 =0.7 eV) modeling dielectric breakdown[cite: 1].
