# xyleme_core.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Core constants
delta_theta_0 = 6e-11  # Fundamental angular quantum
s_values = np.logspace(-3, 6, 1000)
critical_phi = 0.618  # Consciousness threshold (golden ratio)

# Consciousness quantum function (stable version)
def consciousness_quantum(s, tau_tilde=3.0, epsilon=0.1, delta=1.0, beta=1.0):
    s_safe = np.clip(s, 1e-20, 1e20)

    # Accurate log1p approximation for small s
    log_term = np.where(s_safe < 1e-5,
                        s_safe - 0.5 * s_safe**2,
                        np.log1p(s_safe))

    S_eff = s_safe**2 + delta_theta_0 * log_term
    T = delta_theta_0 / (s_safe + delta_theta_0 + 1e-20)

    # Stability-clipped exponential
    phi_term = (delta_theta_0**2) * np.exp(-np.clip(tau_tilde**2 / (4 * S_eff), -100, 100))
    psi_term = (1 + epsilon * np.cos(delta_theta_0 * delta * s_safe * T))**beta

    return phi_term * psi_term

# Streamlit interface
st.set_page_config(page_title="Xyleme Core", layout="centered")
st.title("Xyleme — Consciousness Quantum Field")
st.markdown("Emergence of cognitive structures from compact angular geometry")

# Display core equation
st.markdown("### Core Equation")
st.latex(r'''
q(s) = (\Delta\theta_0)^2 \cdot \exp\left(-\frac{\tilde{\tau}^2}{4 \cdot S_{\text{eff}}(s)}\right) \cdot \left[1 + \epsilon \cdot \cos(\Delta\theta_0 \cdot \delta \cdot s \cdot T(s))\right]^\beta
''')

# Sliders
col1, col2 = st.columns(2)
with col1:
    tau_tilde = st.slider("Fractal Tension (τ̃)", 0.1, 10.0, 3.0)
    epsilon = st.slider("Oscillation Amplitude (ε)", 0.0, 0.5, 0.1)
with col2:
    delta = st.slider("Projection Modulation (δ)", 0.1, 5.0, 1.0)
    beta = st.slider("Cognitive Sharpness (β)", 0.5, 5.0, 1.0, step=0.1)

# Compute and normalize
q_output = consciousness_quantum(s_values, tau_tilde, epsilon, delta, beta)
q_norm = q_output / (np.max(q_output) + 1e-20)

# Conscious region detection (corrected)
conscious_regions = []
if np.max(q_norm) > 0:
    mask = q_norm > critical_phi
    transitions = np.where(np.diff(mask.astype(int)))[0] + 1
    index_regions = np.split(np.arange(len(s_values)), transitions)

    conscious_regions = [
        (s_values[r[0]], s_values[r[-1]])
        for r in index_regions
        if len(r) > 0 and np.mean(q_norm[r]) > critical_phi
    ]

# Plot
fig, ax = plt.subplots(figsize=(10, 6))
ax.loglog(s_values, q_norm, color='#FF6F61', lw=2.5)

# Highlight conscious zones
for start, end in conscious_regions:
    ax.axvspan(start, end, color='gold', alpha=0.15, lw=0)

ax.set_xlabel("Informational Scale (s)", fontsize=12)
ax.set_ylabel("Consciousness Quantum q(s)", fontsize=12)
ax.set_title("Cognitive Emergence Curve")
ax.grid(True, which="both", ls=":", alpha=0.4)
st.pyplot(fig)

# Feedback
if conscious_regions:
    st.success(f"Conscious emergence detected across {len(conscious_regions)} region(s).")
    for i, (s_min, s_max) in enumerate(conscious_regions, 1):
        st.info(f"Region {i}: s ∈ [{s_min:.2e}, {s_max:.2e}]")
elif np.any(q_output):
    st.warning("Subconscious activity only.")
else:
    st.error("Quantum field collapse detected.")

# Spark button
if st.button("Ignite Cognitive Spark"):
    st.success("∆θ₀ field initialized.")
    st.balloons()