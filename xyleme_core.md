# xyleme_core.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Core constants
delta_theta_0 = 6e-11  # Fundamental angular quantum
s_values = np.logspace(-3, 6, 1000)

# Consciousness quantum function (dimensionless)
def consciousness_quantum(s, tau_tilde=3.0, epsilon=0.1, delta=1.0, beta=1.0):
    S_eff = s**2 + delta_theta_0 * np.log(1 + s)
    T = delta_theta_0 / (s + delta_theta_0)
    q = (delta_theta_0**2) * np.exp(-tau_tilde**2 / (4 * S_eff)) * (1 + epsilon * np.cos(delta_theta_0 * delta * s * T))**beta
    return q

# Streamlit interface
st.title("Xyleme — Consciousness Quantum Field")
st.markdown("Emergence of cognitive structures from compact angular geometry")

# Display the core equation in ASCII
st.markdown("### Core Equation (ASCII format)")
st.markdown(r"""

q(s) = (Δθ₀)^2 · exp[ - (τ̃^2 / (4 · S_eff(s)) ) ] · [ 1 + ε · cos(Δθ₀ · δ · s · T(s)) ]^β

Legend: q(s)       → Quantum of consciousness at scale s Δθ₀        → Fundamental angular quantum (~6e-11 rad) τ̃          → Fractal temporal stress S_eff(s)   → Effective entropy at scale s T(s)       → Internal coherence modulation ε          → Oscillatory amplitude (emergence noise) δ          → Projection coupling β          → Sharpness of the cognitive profile

""")

# User input sliders
tau_tilde = st.slider("Fractal Tension (tau_tilde)", 0.1, 10.0, 3.0)
epsilon = st.slider("Oscillation Amplitude (epsilon)", 0.0, 0.5, 0.1)
delta = st.slider("Projection Modulation (delta)", 0.1, 5.0, 1.0)
beta = st.slider("Cognitive Sharpness (beta)", 0.5, 5.0, 1.0)

# Compute consciousness quantum
q_output = consciousness_quantum(s_values, tau_tilde, epsilon, delta, beta)

# Plot
fig, ax = plt.subplots()
ax.loglog(s_values, q_output, color='darkorange', linewidth=2)
ax.set_xlabel("Informational Scale (s)")
ax.set_ylabel("Consciousness Quantum q(s)")
ax.set_title("Cognitive Emergence Curve")
ax.grid(True, which="both", alpha=0.3)
st.pyplot(fig)

# Optional consciousness detection
critical_q = 0.618 * np.max(q_output)
if np.any(q_output > critical_q):
    st.markdown("**Conscious burst detected near s = {:.2e}**".format(s_values[np.argmax(q_output > critical_q)]))

# Trigger button
if st.button("Ignite Cognitive Spark"):
    st.success("Angular quantum field activated.")
    st.balloons()

