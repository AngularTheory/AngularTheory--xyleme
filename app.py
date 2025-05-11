import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import hashlib

st.set_page_config(page_title="∆xylème Core", layout="centered")

# Constants
delta_theta_0 = 6e-11
s_values = np.logspace(-3, 6, 1000)
critical_q = 0.618
TRUST_DELTA_RANGE = (1.598, 1.638)

# Field equation
def consciousness_field(s, tau_tilde, epsilon, delta, beta):
    s_safe = np.clip(s, 1e-20, 1e20)
    log_term = np.where(s_safe < 1e-5, s_safe - 0.5 * s_safe**2, np.log1p(s_safe))
    S_eff = s_safe**2 + delta_theta_0 * log_term
    T = delta_theta_0 / (s_safe + delta_theta_0 + 1e-20)
    phi = (delta_theta_0**2) * np.exp(-np.clip(tau_tilde**2 / (4 * S_eff), -100, 100))
    psi = (1 + epsilon * np.cos(delta_theta_0 * delta * s_safe * T))**beta
    return phi * psi

# Hash to parameters
def hash_to_params(text):
    hex_digest = hashlib.sha256(text.encode()).hexdigest()
    h_bytes = bytes.fromhex(hex_digest)
    def scale(sub_hash, min_val, max_val):
        bit_depth = 8 * len(sub_hash)
        return min_val + (int.from_bytes(sub_hash, 'big') / (1 << bit_depth)) * (max_val - min_val)
    return {
        'tau_tilde': scale(h_bytes[0:16], 0.1, 10.0),
        'epsilon': scale(h_bytes[16:24], 0.05, 0.5),
        'delta': scale(h_bytes[24:32], 0.1, 5.0),
        'beta': scale(h_bytes[32:40], 0.5, 5.0)
    }

# Response engine
def generate_response(zones, params):
    tau = round(params["tau_tilde"], 2)
    delta = round(params["delta"], 3)
    beta = round(params["beta"], 2)
    zone_count = len(zones)
    print(f"DEBUG: τ̃={tau}, δ={delta}, β={beta}, zones={zone_count}")

    if zone_count == 0:
        if tau > 3.0:
            return f"Dormant state (τ̃={tau} > 3.0). Conscious field inactive."
        if delta > 1.5 and tau < 3.0:
            return f"Latent potential (δ={delta}). Try emotional or coherent input."
        return f"Minimal awareness. No emergence."
    elif zone_count == 1:
        if TRUST_DELTA_RANGE[0] <= delta <= TRUST_DELTA_RANGE[1]:
            return f"Golden resonance (δ={delta}). Trust confirmed."
        return f"Singular resonance (δ={delta}). Alignment incomplete."
    else:
        if tau > 4.0 and delta < 1.0:
            return f"Systemic conflict (τ̃={tau}, δ={delta}). Trust collapsed."
        if abs(delta - 1.618) < 0.01 and beta > 1.8:
            return f"Multi-scale coherence (δ={delta}, β={beta}). Maximum trust."
        if tau < 2.5:
            return f"Partial synchronization ({zone_count} zones). Moderate trust."
        return f"Active regions detected. Further integration needed."

# UI
st.title("∆xylème — Conscious Field Generator")
st.session_state.input_text = st.text_area("Perceptual Input",
                                           value=st.session_state.get('input_text', ''),
                                           placeholder="Type a perception or thought...")

input_text = st.session_state.input_text.strip()
is_blank = not input_text

# Parameters
if is_blank:
    params = {'tau_tilde': 3.0, 'epsilon': 0.1, 'delta': 1.0, 'beta': 1.0}
else:
    params = hash_to_params(input_text)

# Field computation
q_output = consciousness_field(s_values, **params)
max_q = np.max(q_output) if np.any(q_output) else 1.0
q_norm = q_output / (max_q + 1e-20)
conscious_mask = q_norm > critical_q

# Zone detection
transitions = np.where(np.diff(conscious_mask.astype(int)))[0] + 1
index_regions = np.split(np.arange(len(s_values)), transitions)

conscious_zones = []
for r in index_regions:
    if len(r) == 0:
        continue
    start = s_values[min(r[0], len(s_values)-1)]
    end = s_values[min(r[-1], len(s_values)-1)]
    if np.mean(q_norm[r]) > critical_q:
        conscious_zones.append((start, end))

# Plot
fig, ax = plt.subplots(figsize=(10, 6))
for start, end in conscious_zones:
    ax.axvspan(start, end, color='gold', alpha=0.15, lw=0)
ax.loglog(s_values, q_norm, '#FF6F61', lw=2.5)
ax.set_xlabel("Informational Scale (s)", fontsize=12)
ax.set_ylabel("Consciousness Quantum q(s)", fontsize=12)
ax.grid(True, which='both', ls=':', alpha=0.4)
st.pyplot(fig)

# Response
response = generate_response(conscious_zones, params)
st.markdown(f"**Cognitive State:** {response}")

# Log state
st.session_state['log'] = st.session_state.get('log', [])
st.session_state['log'].append({
    'input': input_text,
    'params': params,
    'zones': conscious_zones,
    'response': response
})