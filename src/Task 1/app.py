from wandb import Api
import wandb
import os
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

wandb.login(key=os.environ.get("WANDB_API_KEY"))

api = Api()
run = api.run(os.environ.get("ARTIFACTS_URI"))

for f in run.files():
    print(f.Name)
    f.download(replace=True)

@st.cache_resource
def load_model(model_path=None):
    # load tokenizer + model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()
    return tokenizer, model

st.title("AG News Headline Classifier")

local_folder = "~/bert_ag_news"

if st.button("Load Model"):
    with st.spinner("Loading model..."):
        tokenizer, model = load_model(local_folder)
    if tokenizer is not None:
        st.success("Model loaded successfully!")

    # Class label names (AG News typically has 4)
    labels = ["World", "Sports", "Business", "Sci/Tech"]

    # Input headlines
    headlines = st.text_area("Enter headlines (one per line)")

    if st.button("Predict"):
        if not headlines.strip():
            st.warning("Please enter at least one headline.")
        else:
            texts = headlines.strip().split("\n")
            # Tokenize
            inputs = tokenizer(
                texts,
                padding=True,
                truncation=True,
                return_tensors="pt"
            )

            # Predict
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                preds = torch.argmax(logits, dim=-1)

            for text, pred in zip(texts, preds):
                st.write(f"**Headline:** {text}")
                st.write(f"**Predicted label:** {labels[pred]}")
                st.write("---")