import io
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
import streamlit as st

# 1. Page Configuration
st.set_page_config(
    page_title="ASTRO SENTINEL | Space-Grade Semiconductor QA",
    page_icon="🛰️",
    layout="wide"
)

st.title("🛰️ Astro Sentinel | Mission Assurance QA Console")
st.caption("MIL-STD-883 Method 1015 • AEC-Q001 DPAT • NASA Degradation Extrapolation")

# 2. Sidebar Parameters
st.sidebar.header("🎛️ Chamber Controls")
chamber_temp = st.sidebar.slider("Chamber Temperature (°C)", 85.0, 150.0, 125.0, 5.0)
sensor_noise = st.sidebar.slider("Sensor Noise (σ)", 0.05, 1.0, 0.25, 0.05)
lot_size = st.sidebar.slider("Batch Size (DUTs)", 50, 200, 120, 10)
spec_ceiling = st.sidebar.number_input("Datasheet Ceiling (µA)", value=50.0, step=5.0)


# 3. Safe Empirical File Loader
@st.cache_data
def load_hardware_sweeps():
    v_brk, i_leak, v_th = 114.80, 4.792, 2.814
    for f in os.listdir("."):
        fl = f.lower()
        try:
            if "breakdown" in fl and fl.endswith(".csv"):
                df_b = pd.read_csv(f, header=None)
                if not df_b.empty:
                    v_brk = float(pd.to_numeric(df_b.iloc[:, 0], errors="coerce").dropna().max())
            elif "leakage" in fl and fl.endswith(".csv"):
                df_l = pd.read_csv(f, header=None)
                target_col = 1 if df_l.shape[1] > 1 else 0
                series = pd.to_numeric(df_l.iloc[:, target_col], errors="coerce").dropna()
                if not series.empty:
                    i_leak = float(np.abs(series).max() * 1e9)
            elif "turn" in fl and fl.endswith(".csv"):
                df_t = pd.read_csv(f, header=None)
                if df_t.shape[1] >= 2:
                    val_col = pd.to_numeric(df_t.iloc[:, 1], errors="coerce")
                    volt_col = pd.to_numeric(df_t.iloc[:, 0], errors="coerce")
                    knee = volt_col[val_col > 1e-6]
                    v_th = float(knee.iloc[0]) if len(knee) > 0 else float(volt_col.max())
                elif not df_t.empty:
                    v_th = float(pd.to_numeric(df_t.iloc[:, 0], errors="coerce").dropna().max())
        except Exception:
            pass
    return v_brk, i_leak, v_th


v_breakdown, i_leakage_na, v_threshold = load_hardware_sweeps()


# 4. Safe Telemetry Ingestion & Prior Historical Baseline Training
@st.cache_data
def load_lot_data(n_units, temp_c, noise_std):
    base_0h = None
    for f in os.listdir("."):
        if "secom" in f.lower():
            try:
                raw = pd.read_csv(f, sep=r"\s+", header=None, nrows=n_units, usecols=range(6))
                imp = SimpleImputer(strategy="median")
                clean = imp.fit_transform(raw)
                scaler = MinMaxScaler(feature_range=(8.5, 42.0))
                base_0h = scaler.fit_transform(clean[:, [0]]).flatten()
                break
            except Exception:
                pass

    if base_0h is None or len(base_0h) < n_units:
        np.random.seed(42)
        base_h = np.random.normal(10.2, 0.8, int(n_units * 0.90))
        base_m = np.random.normal(38.2, 2.0, int(n_units * 0.05))
        base_d = np.random.normal(10.5, 0.6, n_units - len(base_h) - len(base_m))
        base_0h = np.concatenate([base_h, base_m, base_d])

    t_k = temp_c + 273.15
    af = np.exp((0.7 / 8.617e-5) * ((1 / 298.15) - (1 / t_k))) / 10000.0

    val_24 = np.zeros(n_units)
    val_168 = np.zeros(n_units)
    cats = []

    np.random.seed(42)
    for i in range(n_units):
        is_mav = base_0h[i] > 32.0
        is_drifter = (i % 20 == 0) and not is_mav

        if is_drifter:
            k = np.random.uniform(1.2, 1.8)
            val_24[i] = base_0h[i] + (k * (24**0.6) * (af * 0.1)) + np.random.normal(0, noise_std)
            val_168[i] = base_0h[i] + (k * (168**0.6) * (af * 0.1)) + np.random.normal(0, noise_std)
            cats.append("Latent Drifter")
        elif is_mav:
            val_24[i] = base_0h[i] + np.random.normal(0.3, 0.1) + np.random.normal(0, noise_std)
            val_168[i] = base_0h[i] + np.random.normal(0.8, 0.15) + np.random.normal(0, noise_std)
            cats.append("Lot Maverick")
        else:
            val_24[i] = base_0h[i] + (0.18 * (24**0.3) * af * 0.05) + np.random.normal(0, noise_std)
            val_168[i] = base_0h[i] + (0.18 * (168**0.3) * af * 0.05) + np.random.normal(0, noise_std)
            cats.append("Flight Nominal")

    return pd.DataFrame({
        "Part_ID": [f"DUT_{i+1:04d}" for i in range(n_units)],
        "Value_0h": np.round(base_0h, 3),
        "Value_24h": np.round(val_24, 3),
        "Value_168h_True": np.round(val_168, 3),
        "Category": cats
    })


@st.cache_resource
def get_pretrained_degradation_model():
    """Trains an offline reference model on historical lots to prevent test-set data leakage."""
    np.random.seed(1337)
    n_hist = 2000
    h_0h = np.random.uniform(8.0, 40.0, n_hist)
    h_noise = np.random.normal(0, 0.2, n_hist)
    h_k = np.random.choice([0.18, 1.5], size=n_hist, p=[0.9, 0.1])
    h_24 = h_0h + (h_k * (24**0.5) * 0.08) + h_noise
    h_168 = h_0h + (h_k * (168**0.5) * 0.08) + h_noise

    X_train = pd.DataFrame({
        "Value_0h": h_0h,
        "Value_24h": h_24,
        "Delta_24_0": h_24 - h_0h
    })
    y_train = h_168

    model = HistGradientBoostingRegressor(random_state=42)
    model.fit(X_train, y_train)
    return model


df_active = load_lot_data(lot_size, chamber_temp, sensor_noise)
reg_model = get_pretrained_degradation_model()

# 5. Dual-Engine Screening Pipeline
med_0h = df_active["Value_0h"].median()
mad_0h = 1.4826 * np.median(np.abs(df_active["Value_0h"] - med_0h))
dpat_upper = med_0h + (3.5 * mad_0h)

features = df_active[["Value_0h", "Value_24h"]].copy()
features["Delta_24_0"] = features["Value_24h"] - features["Value_0h"]

iso_model = IsolationForest(contamination=0.08, random_state=42)
iso_flags = iso_model.fit_predict(features)
df_active["Anomaly_Score"] = np.round(iso_model.decision_function(features), 4)

df_active["Predicted_168h"] = np.round(reg_model.predict(features), 3)


def evaluate_gate(row, flag):
    if row["Value_0h"] > spec_ceiling or row["Value_24h"] > spec_ceiling or row["Predicted_168h"] > spec_ceiling:
        return "EARLY REJECT"
    if row["Value_0h"] > dpat_upper or flag == -1:
        return "FLAGGED"
    return "PASS"


df_active["Decision"] = [evaluate_gate(row, iso_flags[i]) for i, row in df_active.iterrows()]

total_dut = len(df_active)
n_pass = int((df_active["Decision"] == "PASS").sum())
n_flag = int((df_active["Decision"] == "FLAGGED").sum())
n_reject = int((df_active["Decision"] == "EARLY REJECT").sum())
yield_pct = int(round((n_pass / total_dut) * 100))
hours_saved = n_reject * 144

# 6. Native Metric Cards
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Total Tested", f"{total_dut} DUTs")
m2.metric("Flight Pass", f"{n_pass}", delta=f"{yield_pct}% Yield")
m3.metric("Mavericks (Flagged)", f"{n_flag}")
m4.metric("Early Aborts (24h)", f"{n_reject}")
m5.metric("Preserved Time", f"{hours_saved} hrs")

st.divider()

# 7. SMU Reference
s1, s2, s3, s4 = st.columns(4)
s1.info(f"**Breakdown V_br:** {v_breakdown:.2f} V")
s2.info(f"**Quiescent I_leak:** {i_leakage_na:.3f} nA")
s3.info(f"**Turn-on V_th:** {v_threshold:.3f} V")
s4.warning(f"**DPAT Upper:** {dpat_upper:.2f} µA")

# 8. Screening Ledger & Inspector
c_left, c_right = st.columns([2, 1])

with c_left:
    st.subheader("📋 Component Screening Ledger")
    st.dataframe(
        df_active[["Part_ID", "Decision", "Value_0h", "Value_24h", "Predicted_168h", "Category"]],
        height=320
    )

    buf = io.StringIO()
    df_active.to_csv(buf, index=False)
    st.download_button(
        "📄 Download Qualification Certificate (CSV)",
        data=buf.getvalue(),
        file_name="Flight_Acceptance_Certificate.csv",
        mime="text/csv"
    )

with c_right:
    st.subheader("🔬 Diagnostics Inspector")
    selected_dut = st.selectbox("Choose Component:", df_active["Part_ID"])
    row = df_active[df_active["Part_ID"] == selected_dut].iloc[0]

    if row["Decision"] == "EARLY REJECT":
        st.error(f"""
        **Status: EARLY REJECT (Aborted at 24h)**
        - **Initial (0h):** {row['Value_0h']:.2f} µA
        - **At 24h:** {row['Value_24h']:.2f} µA
        - **Predicted 168h:** {row['Predicted_168h']:.2f} µA (Exceeds {spec_ceiling:.1f} µA spec)
        - **Saved:** 144 facility burn-in hours
        """)
    elif row["Decision"] == "FLAGGED":
        st.warning(f"""
        **Status: FLAGGED (Maverick Part)**
        - **Initial (0h):** {row['Value_0h']:.2f} µA (Exceeds DPAT limit of {dpat_upper:.2f} µA)
        - **Action:** Shunted to manual screening
        """)
    else:
        st.success("""
        **Status: PASS (Flight Certified)**
        - Nominal wear profile.
        - Certified for satellite subsystem integration.
        """)

# 9. Plots
st.divider()
p1, p2 = st.columns(2)

with p1:
    st.subheader("Lot Distribution & DPAT Boundary")
    fig1, ax1 = plt.subplots(figsize=(6, 3))
    ax1.hist(df_active["Value_0h"], bins=20, color="#2563eb", alpha=0.7, edgecolor="black")
    ax1.axvline(dpat_upper, color="orange", linestyle="--", label=f"DPAT ({dpat_upper:.1f} µA)")
    ax1.axvline(spec_ceiling, color="red", linestyle="-", label=f"Datasheet ({spec_ceiling:.1f} µA)")
    ax1.set_xlabel("0h Current (µA)")
    ax1.set_ylabel("DUT Count")
    ax1.legend(loc="upper right")
    ax1.grid(alpha=0.3)
    st.pyplot(fig1)
    plt.close(fig1)

with p2:
    st.subheader("Degradation Trajectories (0h → 24h → 168h)")
    fig2, ax2 = plt.subplots(figsize=(6, 3))

    pass_df = df_active[df_active["Decision"] == "PASS"].head(6)
    reject_df = df_active[df_active["Decision"] == "EARLY REJECT"].head(6)

    for i, (_, r) in enumerate(pass_df.iterrows()):
        lbl = "Flight Pass (Nominal)" if i == 0 else None
        ax2.plot([0, 24, 168], [r["Value_0h"], r["Value_24h"], r["Predicted_168h"]], color="green", alpha=0.4, label=lbl)

    for i, (_, r) in enumerate(reject_df.iterrows()):
        lbl = "Early Reject (Aborted)" if i == 0 else None
        ax2.plot([0, 24, 168], [r["Value_0h"], r["Value_24h"], r["Predicted_168h"]], color="red", linestyle="--", marker="o", label=lbl)

    ax2.axhline(spec_ceiling, color="red", linestyle=":", label=f"Spec ({spec_ceiling:.1f} µA)")
    ax2.set_xlabel("Chamber Hours")
    ax2.set_ylabel("Leakage (µA)")
    ax2.legend(loc="upper left")
    ax2.grid(alpha=0.3)
    st.pyplot(fig2)
    plt.close(fig2)