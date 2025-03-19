# 📌 Projet Gemini Accessibilité RAG

Ce projet permet de trouver facilement des établissements (restaurants, hôtels, coiffeurs, etc.) accessibles aux personnes en situation de handicap (PMR, visuel, auditif, cognitif), en combinant une approche RAG (Retrieval-Augmented Generation) avec Gemini.

## 🔧 Comment ça marche ?

### 1️⃣ Génération des Embeddings (Google Colab)

Le fichier Google Colab génère les embeddings pour chaque établissement à partir d'une description détaillée.

Ces embeddings (vecteurs) sont ensuite sauvegardés (embeddings.npy) et un index FAISS (faiss_hnsw.index) est créé pour une recherche rapide.

### 2️⃣ Script final avec Gemini et Gradio

Chargement : Les embeddings et l'index pré-calculés sont chargés localement.

Recherche (RAG) : À chaque question posée, le système utilise une recherche sémantique pour trouver les établissements les plus pertinents.

Génération de réponses : Gemini (gemini-2.0-flash) crée des réponses naturelles à partir des résultats obtenus.

Interface Gradio : Une interface web interactive permet aux utilisateurs de poser facilement leurs questions.

### 🚀 Lancement du projet

Exécute d'abord le notebook Colab pour générer les fichiers embeddings et FAISS.

Lance ensuite le script final en local :

python ton_script_final.py

L'interface web Gradio se lance automatiquement et permet d'utiliser facilement le modèle.

✨ Profite de ton outil interactif !




