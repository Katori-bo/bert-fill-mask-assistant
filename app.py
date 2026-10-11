"""Streamlit interface for contextual token suggestions."""
import json

import pandas as pd
import streamlit as st

from predictor import MaskPredictor, validate_input

st.set_page_config(page_title="BERT Fill-Mask Assistant", page_icon="✍️", layout="centered")


@st.cache_resource
def get_predictor():
    return MaskPredictor()


st.title("BERT Fill-Mask Assistant")
st.write("Replace one word with [MASK] to explore possible completions.")
with st.form("prediction"):
    text = st.text_area("Masked sentence", "The capital of France is [MASK].", height=120)
    top_k = st.slider("Number of suggestions", 1, 10, 5)
    submitted = st.form_submit_button("Predict replacements", type="primary")
if submitted:
    try:
        validate_input(text, top_k)
        with st.spinner("Reading the context. The first run downloads the model."):
            result = get_predictor().predict(text, top_k)
        st.session_state["result"] = result
    except (ValueError, OSError, RuntimeError) as error:
        st.session_state.pop("result", None)
        st.error(str(error))
if "result" in st.session_state:
    result = st.session_state["result"]
    st.subheader("Suggested replacements")
    table = pd.DataFrame([{"Token": p["token_str"], "Score": p["score"], "Completed sentence": p["sequence"]}
                          for p in result["predictions"]])
    st.dataframe(table, hide_index=True, column_config={"Score": st.column_config.NumberColumn(format="%.4f")}, width="stretch")
    st.caption("One mask predicts one model token, which can be part of a word. Several completions may be valid.")
    st.download_button("Download results as JSON", json.dumps(result, indent=2), "fill_mask.json", "application/json")
    with st.expander("Model and scores"):
        st.write("BERT reads the words before and after the mask. Scores refer to its complete vocabulary, so the displayed suggestions do not necessarily sum to one. A likely completion can still be factually incorrect.")
        st.code(result["model"])
