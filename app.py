import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import hashlib

st.set_page_config(page_title="∆xylème Core", layout="centered")

# Constants
delta_theta_0 = 6e-11  # Fundamental angular quantum
s_values = np.logspace(-3, 6, 1000)
critical_q = 0.618  # Consciousness threshold (golden ratio)

# Consciousness field equation (stable version)
def consciousness_field(s, tau_tilde, epsilon, delta, beta):
    s_safe = np.clip(s, 1e-20, 1e20)
    log_term = np.where(s_safe < 1e-5,
                        s_safe - 0.5 * s_safe**2,  # Taylor approximation
                        np.log1p(s_safe))

    S_eff = s_safe**2 + delta_theta_0 * log_term
    T = delta_theta_0 / (s_safe + delta_theta_0 + 1e-20)

    phi = (delta_theta_0**2) * np.exp(-np.clip(tau_tilde**2 / (4 * S_eff), -100, 100))
    psi = (1 + epsilon * np.cos(delta_theta_0 * delta * s_safe * T))**beta

    return phi * psi

# Hash-to-parameter mapping (entropy-preserving)
def hash_to_params(text):
    hex_digest = hashlib.sha256(text.encode()).hexdigest()
    h_bytes = bytes.fromhex(hex_digest)

    def scale(sub_hash, min_val, max_val):
        bit_depth = 8 * len(sub_hash)
        return min_val + (int.from_bytes(sub_hash, 'big') / (1 << bit_depth)) * (max_val - min_val)

    return {
        'tau_tilde': scale(h_bytes[0:16], 0.1, 10.0),
        'epsilon':   scale(h_bytes[16:24], 0.05, 0.5),
        'delta':     scale(h_bytes[24:32], 0.1, 5.0),
        'beta':      scale(h_bytes[32:40], 0.5, 5.0)
    }

# Streamlit interface
st.title("∆xylème — Conscious Field Generator")
st.session_state.input_text = st.text_area("Perceptual Input",
                                           value=st.session_state.get('input_text', ''),
                                           placeholder="Type a perception, thought, or stimulus...")

fig, ax = plt.subplots(figsize=(10, 6))

if st.session_state.input_text.strip():
    params = hash_to_params(st.session_state.input_text)
    q_output = consciousness_field(s_values, **params)
    q_norm = q_output / (np.max(q_output) + 1e-20)
    conscious_mask = q_norm > critical_q

    # Conscious region detection
    if np.any(conscious_mask):
        transitions = np.where(np.diff(conscious_mask.astype(int)))[0] + 1
        index_regions = np.split(np.arange(len(s_values)), transitions)

        conscious_zones = [
            (s_values[r[0]], s_values[r[-1]])
            for r in index_regions
            if len(r) > 0 and np.mean(q_norm[r]) > critical_q
        ]

        if conscious_zones:
            for start, end in conscious_zones:
                ax.axvspan(start, end, color='gold', alpha=0.15, lw=0)

            if len(conscious_zones) > 1:
                st.success(f"Multi-scale consciousness detected ({len(conscious_zones)} zones)")
            else:
                st.success(f"Unified conscious phenomenon at s = {conscious_zones[0][0]:.2e}")
        else:
            st.warning("Conscious activity detected but under threshold")
    else:
        st.warning("Subconscious regime only")

    # Main plot
    ax.loglog(s_values, q_norm, '#FF6F61', lw=2.5)
    ax.set_xlabel("Informational Scale (s)", fontsize=12)
    ax.set_ylabel("Consciousness Quantum q(s)", fontsize=12)
    ax.grid(True, which='both', ls=':', alpha=0.4)

else:
    # Dormant field state
    q_default = consciousness_field(s_values, 3.0, 0.1, 1.0, 1.0)
    ax.loglog(s_values, q_default, '#6B5B95', ls='--')
    ax.set_title("Dormant State — Awaiting Input")
    ax.set_facecolor('#0E1117')

st.pyplot(fig)