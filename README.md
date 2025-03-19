# 📌 Projet Gemini Accessibilité RAG

Ce projet permet de trouver facilement des établissements (restaurants, hôtels, coiffeurs, etc.) accessibles aux personnes en situation de handicap (PMR, visuel, auditif, cognitif), en combinant une approche RAG (Retrieval-Augmented Generation) avec Gemini.

## 🔧 Comment ça marche ?

### 1️⃣ Génération des Embeddings (Google Colab)

Le fichier CreateEmbeddings.py génère les embeddings pour chaque établissement à partir d'une description détaillée. 

Ces embeddings (vecteurs) sont ensuite sauvegardés (embeddings.npy) et un index FAISS (faiss_hnsw.index) est créé pour une recherche rapide.

Il faut faire tourner ce code sur google colab car cela dure environ 20 minutes avec le GPU tandis que en local il peux mettre des heures.

### 2️⃣ RAG.py avec Gemini et Gradio

Le fichier RAG.py est à executer en local. 
Il utilise les fichier d'embeddings créé sur colab. 

Chargement : Les embeddings et l'index pré-calculés sont chargés localement.

Recherche (RAG) : À chaque question posée, le système utilise une recherche sémantique pour trouver les établissements les plus pertinents en comparant les vecteurs des embeddings et le vecteur de la question.

Génération de réponses : Gemini (gemini-2.0-flash) crée des réponses naturelles à partir des résultats obtenus.

Interface Gradio : Une interface web interactive permet aux utilisateurs de poser facilement leurs questions.

### ⚠️ Limites actuelles du modèle

Le modèle actuel présente certaines limites, notamment dans la précision des résultats :

Les vecteurs proches ne correspondent pas toujours exactement aux réponses attendues.

Parfois, des résultats non pertinents peuvent être affichés, notamment quand l'activité ou la localisation spécifiée n'est pas parfaitement prise en compte par la recherche sémantique.

### 🚀 Lancement du projet

Exécute d'abord le notebook Colab pour générer les fichiers embeddings et FAISS.

Lance ensuite le script final en local :

RAG.py

L'interface web Gradio se lance automatiquement et permet d'utiliser facilement le modèle.





