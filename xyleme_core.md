# xyleme_core.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Core constants
delta_theta_0 = 6e-11  # Fundamental angular quantum
s_values = np.logspace(-3, 6, 1000)
critical_phi = 0.618  # Consciousness threshold (golden ratio)

# Consciousness quantum function (validated C∆GE core)
def consciousness_quantum(s, tau_tilde=3.0, epsilon=0.1, delta=1.0, beta=1.0):
    s_safe = np.clip(s, 1e-20, 1e20)
    log_term = np.where(s_safe < 1e-5,
                        s_safe - 0.5 * s_safe**2,
                        np.log1p(s_safe))
    S_eff = s_safe**2 + delta_theta_0 * log_term
    T = delta_theta_0 / (s_safe + delta_theta_0 + 1e-20)
    phi_term = (delta_theta_0**2) * np.exp(-np.clip(tau_tilde**2 / (4 * S_eff), -100, 100))
    psi_term = (1 + epsilon * np.cos(delta_theta_0 * delta * s_safe * T))**beta
    return phi_term * psi_term

# Streamlit UI
st.set_page_config(page_title="Xyleme Core", layout="centered")
st.title("Xyleme — Consciousness Quantum Field")
st.markdown("Emergence of cognitive structures from compact angular geometry")

# Display equation
st.markdown("### Core Equation")
st.latex(r'''
q(s) = (\Delta\theta_0)^2 \cdot \exp\left(-\frac{\tilde{\tau}^2}{4 \cdot S_{\text{eff}}(s)}\right) \cdot \left[1 + \epsilon \cdot \cos(\Delta\theta_0 \cdot \delta \cdot s \cdot T(s))\right]^\beta
''')

# Input field
st.markdown("### Perceptual Input")
user_input = st.text_area("Enter a perception or thought")

# Store input when "Send" is clicked
if st.button("Send") and user_input.strip():
    st.session_state.input_text = user_input.strip()

# Get stored input or fallback
perception = st.session_state.get("input_text", "...")

# Sliders for parameters
col1, col2 = st.columns(2)
with col1:
    tau_tilde = st.slider("Fractal Tension (τ̃)", 0.1, 10.0, 3.0)
    epsilon = st.slider("Oscillation Amplitude (ε)", 0.0, 0.5, 0.1)
with col2:
    delta = st.slider("Projection Modulation (δ)", 0.1, 5.0, 1.0)
    beta = st.slider("Cognitive Sharpness (β)", 0.5, 5.0, 1.0, step=0.1)

# Compute and normalize securely
q_output = consciousness_quantum(s_values, tau_tilde, epsilon, delta, beta)
max_q = np.max(q_output) if np.any(q_output) else 1.0
q_norm = q_output / (max_q + 1e-20)

# Conscious region detection (robust)
conscious_regions = []
if np.max(q_norm) > 0:
    mask = q_norm > critical_phi
    transitions = np.where(np.diff(mask.astype(int)))[0] + 1
    index_regions = np.split(np.arange(len(s_values)), transitions)

    for r in index_regions:
        if len(r) == 0:
            continue
        start = s_values[min(r[0], len(s_values) - 1)]
        end = s_values[min(r[-1], len(s_values) - 1)]
        if np.mean(q_norm[r]) > critical_phi:
            conscious_regions.append((start, end))

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