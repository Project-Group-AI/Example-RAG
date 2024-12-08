# Example-RAG

## Explanation
- Step 1: Data encoding
The knowledge base (texts, documents, etc.) is converted into embedding vectors using an encoding model (often a sentence-transformers type model).
- Step 2: Encoding the User Query
The user query is also converted to a vector using the same encoding pattern.
- Step 3: Search for relevant documents
We use a similarity measure (for example, cosine similarity) to compare the query vector with those of the base and retrieve the most relevant documents.
- Step 4: Generation with context
The retrieved documents serve as context for an LLM (like GPT or Mistral) to generate a relevant answer.

### Result without good context with RAG
![CleanShot 2024-12-08 at 18 29 43@2x](https://github.com/user-attachments/assets/e83e9cc8-b999-490b-a67e-fc4f82ed3d62)

### Result with good context with RAG
![CleanShot 2024-12-08 at 18 30 28@2x](https://github.com/user-attachments/assets/9bc72859-294f-4905-aa3c-022ee62de09a)

### RAG method allows you to sort relevant documents according to the question
for example, if I ask a question about chess, the documents chosen to answer are those related to chess and not the others
![CleanShot 2024-12-08 at 18 54 05@2x](https://github.com/user-attachments/assets/85df87ab-a987-4a00-8ff4-ad82b37d31a1) ![CleanShot 2024-12-08 at 18 55 55@2x](https://github.com/user-attachments/assets/6fc1b64b-2191-47d2-b7d1-27ed6afcfa3b)

### Even without a good context the LLM can answer a question to which it can
![CleanShot 2024-12-08 at 18 35 01@2x](https://github.com/user-attachments/assets/92c6c6a4-4be1-420f-80d3-34e41e85f120)


