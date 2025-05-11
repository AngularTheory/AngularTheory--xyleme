import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import hashlib

st.set_page_config(page_title="∆xylème Core", layout="centered")

# Constants
delta_theta_0 = 6e-11  # Fundamental angular quantum
s_values = np.logspace(-3, 6, 1000)
critical_q = 0.618

# Consciousness field equation
def consciousness_field(s, tau_tilde, epsilon, delta, beta):
    S_eff = s**2 + delta_theta_0 * np.log1p(s)
    T = delta_theta_0 / (s + delta_theta_0 + 1e-20)
    phi = (delta_theta_0**2) * np.exp(-tau_tilde**2 / (4 * S_eff))
    psi = (1 + epsilon * np.cos(delta_theta_0 * delta * s * T))**beta
    return phi * psi

# Hash mapping to parameter space
def hash_to_params(text):
    hex_digest = hashlib.sha256(text.encode()).hexdigest()
    h_bytes = bytes.fromhex(hex_digest)

    def scale(sub_hash, min_val, max_val):
        bit_depth = 8 * len(sub_hash)
        return min_val + (int.from_bytes(sub_hash, 'big') / (1 << bit_depth)) * (max_val - min_val)

    return {
        'tau_tilde': scale(h_bytes[0:16], 0.1, 10.0),
        'epsilon':  scale(h_bytes[16:24], 0.05, 0.5),
        'delta':    scale(h_bytes[24:32], 0.1, 5.0),
        'beta':     scale(h_bytes[32:40], 0.5, 5.0)
    }

# Initialize session state for input_text
if 'input_text' not in st.session_state:
    st.session_state['input_text'] = ''

# Interface
st.title("∆xylème — Consciousness Field Generator")
st.session_state['input_text'] = st.text_area("Perceptual Input", value=st.session_state['input_text'], placeholder="Enter text/thought...")

fig, ax = plt.subplots(figsize=(10, 6))

if st.session_state['input_text'].strip():
    params = hash_to_params(st.session_state['input_text'])
    q_output = consciousness_field(s_values, **params)
    q_norm = q_output / (np.max(q_output) + 1e-20)
    conscious_mask = q_norm > critical_q

    # Conscious region detection
    if np.any(conscious_mask):
        transitions = np.where(np.diff(conscious_mask.astype(int)))[0] + 1
        regions = np.split(s_values, transitions)

        conscious_zones = [
            (r[0], r[-1]) 
            for r in regions 
            if len(r) > 0 and np.max(q_norm[np.isin(s_values, r)]) > critical_q
        ]

        for start, end in conscious_zones:
            ax.axvspan(start, end, color='gold', alpha=0.15, lw=0)

        if len(conscious_zones) > 1:
            st.success(f"Conscience quantique multi-échelle détectée ({len(conscious_zones)} niveaux)")
        else:
            st.success(f"Phénomène conscient unifié à s = {conscious_zones[0][0]:.2e}")
    else:
        st.warning("Subconscious regime only.")

    ax.loglog(s_values, q_norm, color='#FF6F61', linewidth=2.5)
    ax.set_xlabel("Integrated Information Scale (s)")
    ax.set_ylabel("Consciousness Quantum q(s)")
    ax.set_title("Cognitive Emergence Field")
    ax.grid(True, which="both", ls="--", alpha=0.4)
    st.pyplot(fig)

else:
    # Default state — dormant quantum field
    q_default = consciousness_field(s_values, 3.0, 0.1, 1.0, 1.0)
    ax.loglog(s_values, q_default, color='#6B5B95', linestyle='--')
    ax.set_title("Dormant State — Awaiting Input")
    ax.set_facecolor('#0E1117')
    ax.grid(True, which="both", alpha=0.3)
    st.markdown("```\n∆xylème core en attente...\n```")
    st.pyplot(fig)