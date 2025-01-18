from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_ollama.llms import OllamaLLM

loader = PyPDFLoader(r"C:\Users\Christian\OneDrive - Universidade Federal do Pará - UFPA\TRABALHOS\8 Periodo\Engenharia de Software\Chatbot-FAQ-Whatsapp\Editais\Edital_05_2024_COPERPS_PS2025_retificado_v3.pdf")

doc = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size= 1200,
    chunk_overlap = 100,
    add_start_index = True
)

all_splitts = text_splitter.split_documents(doc)

loocal_embeddings = OllamaEmbeddings(model="all-minilm:33m")

vectorstore = Chroma.from_documents(documents=all_splitts, embedding=loocal_embeddings)

question = "caso eu não tenha familiar com conta bancária, o que eu devo fazer?"
retrive = vectorstore.as_retriever(
    search_type = "similarity",
    search_kwargs = {"k":3}
)

retrive_doc = retrive.invoke(question)

context = ' '.join([doc.page_content for doc in retrive_doc])

llm = OllamaLLM(model = "llama3.2:1b")

response = llm.invoke(f"""Responda a pergunta de acordo com o contexto dado previamente: Pergunta {question} Contexto: {context}""")

print(response)