from sentence_transformers import SentenceTransformer
from FaqLista import faq_str, faq
from ollama import chat
from ollama import ChatResponse

chat_history = []

def initialize_model(model="all-MiniLM-L6-v2"):
    model_init = SentenceTransformer(model)
    return model_init
    
    
model = initialize_model()
embedding_faq = model.encode(faq_str)


print("\n============iniciando================")
print("ChatBot -> Olá, sou um ChatBot e estou aqui para te ajudar. Digite 'exit' para sair.\n")

while True:
    question = input("Voce -> ")
    if question == "exit":
        break
    
    answer = model.encode(question)
    similarities = model.similarity(embedding_faq, answer)

    most_similar = similarities.argmax()
    similar_question = faq[most_similar]
    
    chat_history.append({'role': 'user','content': question})
    chat_history.append({'role': 'assistant','content': f"FAQ -> {similar_question['R']}"})
    
    response: ChatResponse = chat(model='llama3.2:1b', messages=chat_history+[
        {
            'role': 'user',
            'content': 'Faça uma resposta clara e objetiva usando de base o texto fornecido',
        },
    ], options={"Temperature": 0.2})

    final_response = response.message.content
    print(f"ChatBot -> {final_response}\n")
