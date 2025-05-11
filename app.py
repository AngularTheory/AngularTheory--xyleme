# Optional user input for cognitive stimulation
st.markdown("---")
st.markdown("### Stimulate the Xyleme Core with a perception (textual or emotional input)")

user_input = st.text_area("Symbolic Input", placeholder="Type something meaningful...")

if st.button("Stimulate ∆xylème"):
    if user_input.strip():
        st.markdown(f"**∆τ response**: {np.abs(hash(user_input)) % 1000:.3f}")
        st.markdown(f"**Decision**: `EXPLORE`")
        st.markdown(f"**Internal Epoch**: `0`")
    else:
        st.warning("Input is empty. Please enter a perception or stimulus.")