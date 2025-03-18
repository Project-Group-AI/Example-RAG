import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import gradio as gr

# Charger les données
df = pd.read_csv("C:\\Users\\titou\\Desktop\\I-LLM\\acceslibre-with-web-url(1).csv", low_memory=False)

# Fonction description
def creer_description(row):
    descriptions = [
        str(row["name"]) if pd.notnull(row["name"]) else "",
        f"{str(row['numero'])} {str(row['voie'])}" if pd.notnull(row['numero']) else str(row['voie']),
        str(row["commune"]) if pd.notnull(row["commune"]) else "",
        f"Activité : {str(row['activite'])}" if pd.notnull(row["activite"]) else "",
    ]

    moteur = []
    if row.get("entree_pmr"):
        moteur.append("entrée accessible PMR")
    if row.get("sanitaires_adaptes"):
        moteur.append("sanitaires adaptés PMR")
    if row.get("stationnement_pmr"):
        moteur.append("stationnement PMR disponible")
    if moteur:
        descriptions.append(f"Accessibilité moteur : {', '.join(moteur)}")

    visuel = []
    if row.get("cheminement_ext_bande_guidage"):
        visuel.append("bande de guidage extérieure")
    if row.get("entree_vitree_vitrophanie"):
        visuel.append("vitrophanie à l'entrée")
    if row.get("entree_balise_sonore"):
        visuel.append("balise sonore disponible")
    if visuel:
        descriptions.append(f"Accessibilité visuelle : {', '.join(visuel)}")

    auditif = []
    if row.get("accueil_equipements_malentendants_presence"):
        auditif.append("équipements pour malentendants disponibles")
    if auditif:
        descriptions.append(f"Accessibilité auditive : {', '.join(auditif)}")

    cognitif = []
    if row.get("accueil_personnels"):
        cognitif.append("personnel formé à l'accueil spécifique")
    if cognitif:
        descriptions.append(f"Accessibilité cognitive : {', '.join(cognitif)}")

    if pd.notnull(row['site_internet']):
        descriptions.append(f"Site web : {row['site_internet']}")

    descriptions = [desc for desc in descriptions if desc]

    return "; ".join(descriptions)

df["description"] = df.apply(creer_description, axis=1)

# Embeddings & FAISS
embeddings = np.load("embeddings.npy")
index = faiss.read_index("faiss_hnsw.index")

model = SentenceTransformer('all-MiniLM-L6-v2')

# Gemini configuration
genai.configure(api_key="AIzaSyCRYsdwXIdwn21a-R8zw01z7xAopksNOVU")
model_gemini = genai.GenerativeModel('gemini-2.0-flash')

# Fonction retrieval
def recuperer_infos(query, k=3):
    query_embedding = model.encode([query])
    faiss.normalize_L2(query_embedding)
    D, I = index.search(query_embedding, k=k)
    return df.iloc[I[0]]

# Fonction Gemini response
def generer_reponse(query):
    resultats = recuperer_infos(query)
    prompt = f"L'utilisateur demande : '{query}'. Voici les établissements correspondants :\n"
    for _, row in resultats.iterrows():
        prompt += f"- {row['description']}\n"
    prompt += "\nPrésente ces établissements clairement en mettant en avant les adaptations spécifiques pour le handicap mentionné."
    response = model_gemini.generate_content(prompt)
    return response.text

# Gradio UI
def chatbot_interface(user_input, history):
    reponse = generer_reponse(user_input)
    return reponse

chat_interface = gr.ChatInterface(fn=chatbot_interface,
                                  title="🤖 Gemini Accessibilité RAG",
                                  description="Pose tes questions sur l'accessibilité des lieux selon différents handicaps.")

chat_interface.launch()
