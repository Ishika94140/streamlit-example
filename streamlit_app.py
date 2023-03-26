import streamlit as st
import pandas as pd
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch

# Charger le modèle et tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)

# Désactiver le calcul sur le GPU
device = torch.device('cpu')
model.to(device)

# Charger le fichier CSV contenant les données à prédire
df = pd.read_csv("https://github.com/Ishika94140/streamlit-example/blob/37706a6a58203ebfc25b49dd210eeb2c39b0e804/training_file_1.csv", encoding='latin1')

# Prétraiter les données
abstracts = df['Abstract'].tolist()
encoded_abstracts = tokenizer(abstracts, padding=True, truncation=True, return_tensors='pt')
input_ids = encoded_abstracts['input_ids']
attention_masks = encoded_abstracts['attention_mask']

# Définir l'application Streamlit
st.title("Détection de plagiat dans les abstracts")

# Récupérer l'abstract déposé par l'utilisateur
user_abstract = st.text_input("Entrez l'abstract : ")

# Faire les prédictions si l'abstract de l'utilisateur est différent des abstracts dans le jeu de données
if user_abstract not in abstracts:
    encoded_user_abstract = tokenizer(user_abstract, padding=True, truncation=True, return_tensors='pt')
    user_input_ids = encoded_user_abstract['input_ids']
    user_attention_mask = encoded_user_abstract['attention_mask']

    with torch.no_grad():
        model.eval()
        user_input_ids = user_input_ids.to(device)
        user_attention_mask = user_attention_mask.to(device)
        output = model(user_input_ids, attention_mask=user_attention_mask)
        pred = output[0].argmax(dim=1).tolist()[0]

    if pred == 1:
        st.write("Plagiat détecté !")
    else:
        st.write("Aucun plagiat détecté.")
else:
    st.write("L'abstract est déjà présent dans le jeu de données, plagiat détecté.")
