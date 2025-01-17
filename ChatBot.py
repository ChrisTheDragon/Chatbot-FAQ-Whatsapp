import chromadb
from FaqLista import faq_str, faq
from ollama import chat
from ollama import ChatResponse
from chromadb.config import Settings

# Configuração do ChromaDB
client = chromadb.Client(Settings(
    persist_directory="./chroma",  # Diretório para persistência dos dados
    chroma_db_impl="duckdb+parquet",     # Backend padrão
))

# Criação da coleção para o FAQ
collection = client.get_or_create_collection(name="faq_collection")


chat_history = []

def initialize_model(model="all-MiniLM-L6-v2"):
    return model
    
model = initialize_model()

def store_faq_in_chromadb(faq, model, collection):
    # Gere embeddings das perguntas
    embeddings = model.encode([item['Q'] for item in faq])
    
    # Adicione cada item à coleção
    for i, item in enumerate(faq):
        collection.add(
            documents=[item['Q']],  # Texto da pergunta
            metadatas=[{'answer': item['R']}],  # Resposta associada
            ids=[f"faq_{i}"],  # IDs únicos
            embeddings=[embeddings[i]]  # Embedding gerado
        )

# Exemplo de uso
store_faq_in_chromadb(faq, model, collection)

def query_chromadb(question, model, collection, top_k=1):
    # Gere o embedding da pergunta
    query_embedding = model.encode([question])
    
    # Execute a consulta no ChromaDB
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k  # Número de resultados mais próximos
    )
    
    # Retorna a resposta mais relevante
    if results['documents']:
        return results['metadatas'][0]['answer']
    else:
        return "Desculpe, não encontrei uma resposta no FAQ."



print("\n============iniciando================")
print("ChatBot -> Olá, sou um ChatBot e estou aqui para te ajudar. Digite 'exit' para sair.\n")

while True:
    question = input("Você -> ")
    if question == "exit":
        break
    
    # Busca no ChromaDB
    similar_response = query_chromadb(question, model, collection)
    
    # Adiciona ao histórico do chat
    chat_history.append({'role': 'user', 'content': question})
    chat_history.append({'role': 'assistant', 'content': f"FAQ -> {similar_response}"})
    
    # Gera resposta usando Ollama
    response = chat(model='llama3.2:1b', messages=chat_history + [
        {
            'role': 'user',
            'content': 'Faça uma resposta clara e objetiva usando de base o texto fornecido.',
        },
    ], options={"Temperature": 0.2})
    
    final_response = response.message.content
    print(f"ChatBot -> {final_response}\n")
