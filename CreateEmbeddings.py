# Exécute ces cellules séparément dans Colab :

# Cellule 1 : Installation des librairies
!pip install sentence-transformers faiss-cpu pandas numpy

# Cellule 2 : Import des bibliothèques
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

# Active le GPU pour Sentence Transformers
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device utilisé : {device}")

# Cellule 3 : Charger le CSV sur Colab

from google.colab import drive
drive.mount('/content/drive')


# Remplace ci-dessous par le nom exact de ton fichier après upload
# Exemple pour CSV
df = pd.read_csv('/content/drive/MyDrive/acceslibre-with-web-url(1).csv', low_memory=False)

# Cellule 4 : Fonction de création des descriptions
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

# Cellule 5 : Génération rapide des embeddings (GPU)
model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
embeddings = model.encode(df["description"].tolist(), batch_size=1024, show_progress_bar=True, convert_to_numpy=True)

# Sauvegarder les embeddings générés sur Colab
np.save("embeddings.npy", embeddings)

# Cellule 6 : Créer et sauvegarder l'index FAISS HNSW
dimension = embeddings.shape[1]
index = faiss.IndexHNSWFlat(dimension, 32)
faiss.normalize_L2(embeddings)
index.add(embeddings)
faiss.write_index(index, "faiss_hnsw.index")

# Cellule 7 : Téléchargement local des embeddings et index
from google.colab import files
files.download("embeddings.npy")
files.download("faiss_hnsw.index")
