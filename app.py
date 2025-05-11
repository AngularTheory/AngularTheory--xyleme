import streamlit as st

st.set_page_config(page_title="∆xylème Core", layout="centered")

st.markdown("## ∆xylème — Cœur Algorithmique")
st.markdown("### Équation pivot C∆GE (forme symbolique)")

st.latex(r"m(s) = m_e \cdot (\Delta\theta_0)^2 \cdot \exp\left[ -\frac{\tilde{\tau}^2}{4 \cdot S_\text{eff}(s)} \right] \cdot \left[ 1 + \varepsilon \cdot \cos(\Delta\theta_0 \cdot \delta \cdot s \cdot T(s)) \right]^\beta")
