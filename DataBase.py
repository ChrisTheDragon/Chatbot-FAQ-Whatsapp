from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

default_persist_directory = "./chroma_langchain_db"

vector_store = Chroma(
    collection_name="Editais_UFPA",
    embedding_function=OllamaEmbeddings(model="all-minilm:33m"),
    persist_directory=default_persist_directory,  # Where to save data locally, remove if not necessary
)

def loadDocument(pdf_path):
    """
    Carrega um documento PDF e o divide em partes menores para processamento.
    
    Args:
        pdf_path (str): Caminho para o arquivo PDF.
    
    Returns:
        list: Uma lista de documentos divididos em partes.
    """
    try:
        loader = PyPDFLoader(pdf_path)
        doc = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200,
            chunk_overlap=100,
            add_start_index=True
        )
        all_splitts = text_splitter.split_documents(doc)
        return all_splitts
    except Exception as e:
        print(f"Erro ao carregar o documento: {e}")
        return None


def storageInChroma(documents, persist_directory=default_persist_directory, vector_db=vector_store):
    """
    Armazena os documentos em um banco de dados vetorial usando ChromaDB com persistência.
    
    Args:
        documents (list): Lista de documentos processados.
        persist_directory (str): Diretório para persistir os dados do ChromaDB.
        vector_db (Chroma): O banco de dados vetorial para armazenar os documentos.
    
    Returns:
        Chroma: O banco de dados vetorial com os documentos armazenados.
    """
    try:        
        # Configurar o banco de dados Chroma
        vector_db.add_documents(documents=documents)
        
        print(f"Documentos armazenados no diretório: {persist_directory}")
        return vector_store
    except Exception as e:
        print(f"Erro ao armazenar os documentos no ChromaDB: {e}")
        return None


def loadAndStoreDocument(persist_directory=default_persist_directory):
    """
    Solicita o caminho de um documento ao usuário, carrega o conteúdo e o armazena no ChromaDB.
    
    Args:
        persist_directory (str): Diretório para persistir os dados do ChromaDB.
    
    Returns:
        Chroma: O banco de dados vetorial com os documentos armazenados.
    """
    try:
        # Solicitar o caminho do documento ao usuário
        pdf_path = input("Digite o caminho completo do arquivo PDF: ").strip()
        
        # Carregar o documento
        print("Carregando documento...")
        documents = loadDocument(pdf_path)
        
        if not documents:
            print("Falha ao carregar o documento. Verifique o caminho e tente novamente.")
            return None
        
        # Armazenar no ChromaDB
        print("Armazenando documentos no ChromaDB...")
        vectorstore = storageInChroma(
            documents, 
            persist_directory=persist_directory
        )
        
        if vectorstore:
            print(f"Documento armazenado com sucesso no ChromaDB! Dados persistidos em: {persist_directory}")
        else:
            print("Falha ao armazenar os documentos no ChromaDB.")
        
        return vectorstore
    except Exception as e:
        print(f"Ocorreu um erro: {e}")
        return None


if __name__ == "__main__":

    # Processar e armazenar o documento
    #vectorstore = loadAndStoreDocument()
    print(vector_store.similarity_search("caso eu não tenha familiar com conta bancária, o que eu devo fazer?", k=3))
    
    



# '''loader = PyPDFLoader(r"C:\Users\Christian\OneDrive - Universidade Federal do Pará - UFPA\TRABALHOS\8 Periodo\Engenharia de Software\Chatbot-FAQ-Whatsapp\Editais\Edital_05_2024_COPERPS_PS2025_retificado_v3.pdf")

# doc = loader.load()

# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size= 1200,
#     chunk_overlap = 100,
#     add_start_index = True
# )

# all_splitts = text_splitter.split_documents(doc)

# loocal_embeddings = OllamaEmbeddings(model="all-minilm:33m")

# vectorstore = Chroma.from_documents(documents=all_splitts, embedding=loocal_embeddings)'''

# '''question = "caso eu não tenha familiar com conta bancária, o que eu devo fazer?"
# retrive = vectorstore.as_retriever(
#     search_type = "similarity",
#     search_kwargs = {"k":3}
# )

# retrive_doc = retrive.invoke(question)

# context = ' '.join([doc.page_content for doc in retrive_doc])

# llm = OllamaLLM(model = "llama3.2:1b")

# response = llm.invoke(f"""Responda a pergunta de acordo com o contexto dado previamente: Pergunta {question} Contexto: {context}""")

# print(response)'''