from langchain_ollama.llms import OllamaLLM
from DataBase import loadDocument, storageInChroma, loadAndStoreDocument, initializeChromaDB

vectorstore = initializeChromaDB()

question = "quais documentos eu preciso para comprovar a renda?"
retrive = vectorstore.as_retriever(
    search_type = "similarity",
    search_kwargs = {"k":3}
)

retrive_doc = retrive.invoke(question)

context = ' '.join([doc.page_content for doc in retrive_doc])
print(context)

llm = OllamaLLM(model = "llama3.2:1b")



response = llm.invoke(f"""Responda a pergunta de acordo com o contexto dado previamente: Pergunta {question} Contexto: {context}""")

print(response)

chat_history = []








# print("\n============iniciando================")
# print("ChatBot -> Olá, sou um ChatBot e estou aqui para te ajudar. Digite 'exit' para sair.\n")

# while True:
#     question = input("Você -> ")
#     if question == "exit":
#         break
    
#     # Busca no ChromaDB
#     similar_response = query_chromadb(question, model, collection)
    
#     # Adiciona ao histórico do chat
#     chat_history.append({'role': 'user', 'content': question})
#     chat_history.append({'role': 'assistant', 'content': f"FAQ -> {similar_response}"})
    
#     # Gera resposta usando Ollama
#     response = chat(model='llama3.2:1b', messages=chat_history + [
#         {
#             'role': 'user',
#             'content': 'Faça uma resposta clara e objetiva usando de base o texto fornecido.',
#         },
#     ], options={"Temperature": 0.2})
    
#     final_response = response.message.content
#     print(f"ChatBot -> {final_response}\n")
