from sentence_transformers import SentenceTransformer
import FaqLista

# 1. Load a pretrained Sentence Transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

# The sentences to encode
sentences = FaqLista.faq_str

# 2. Calculate embeddings by calling model.encode()
embeddings = model.encode(sentences)
#print(embeddings.shape)


question = "Quais os Suportes Técnicos?"
question_embenddings = model.encode(question)

# 3. Calculate the embedding similarities
similarities = model.similarity(embeddings, question_embenddings)
#print(similarities)

# 4. Get the index of the most similar FAQ
most_similar = similarities.argmax()
similar_question = FaqLista.faq[most_similar]
print(f'P: {similar_question["P"]}\nR: {similar_question["R"]}')