from langchain_ollama.llms import OllamaLLM
from DataBase import loadDocument, storageInChroma, loadAndStoreDocument, initializeChromaDB

vectorstore = initializeChromaDB()

chat_history = []

print("\n============iniciando================")
print("ChatBot -> Olá, sou um ChatBot e estou aqui para te ajudar. Digite 'exit' para sair.\n")

while True:
    question = input("Você -> ")
    if question == "exit":
        break

    retrive = vectorstore.as_retriever(
        search_type = "similarity",
        search_kwargs = {"k":3}
    )

    retrive_doc = retrive.invoke(question)
    context = ' '.join([doc.page_content for doc in retrive_doc])
    #print(context)

    llm = OllamaLLM(model = "llama3.2:1b")

    response = llm.invoke(f"""Você é um chatbot de ajuda a responder sobre Editais de Processo seletivo para ingresso na UFPA e vai responder a pergunta de acordo com o contexto dado previamente: Pergunta {question} Contexto: {context}""")

    print(f'Chatbot -> {response}')
