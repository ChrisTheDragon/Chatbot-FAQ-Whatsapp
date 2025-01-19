import chromadb
from ollama import chat
from ollama import ChatResponse
from chromadb.config import Settings
from DataBase import loadDocument, storageInChroma




chat_history = []

def initialize_model(model="all-MiniLM-L6-v2"):
    return model
    
model = initialize_model()





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
