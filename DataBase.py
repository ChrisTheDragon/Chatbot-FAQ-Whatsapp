from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama.llms import OllamaLLM

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


from langchain.embeddings import OllamaEmbeddings
from langchain.vectorstores import Chroma

def storageInChroma(documents, embedding_model="all-minilm:33m", persist_directory="/chroma"):
    """
    Armazena os documentos em um banco de dados vetorial usando ChromaDB com persistência.
    
    Args:
        documents (list): Lista de documentos processados.
        embedding_model (str): Modelo de embeddings para usar na representação vetorial.
        persist_directory (str): Diretório para persistir os dados do ChromaDB.
    
    Returns:
        Chroma: O banco de dados vetorial com os documentos armazenados.
    """
    try:
        # Carregar o modelo de embeddings
        embeddings = OllamaEmbeddings(model=embedding_model)
        
        # Configurar o banco de dados Chroma com persistência
        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=persist_directory
        )
        
        # Salvar os dados no disco
        vectorstore.persist()
        
        print(f"Documentos armazenados com persistência no diretório: {persist_directory}")
        return vectorstore
    except Exception as e:
        print(f"Erro ao armazenar os documentos no ChromaDB: {e}")
        return None


def processAndStoreDocument(persist_directory="chroma_storage", embedding_model="all-minilm:33m"):
    """
    Solicita o caminho de um documento ao usuário, carrega o conteúdo e o armazena no ChromaDB.
    
    Args:
        persist_directory (str): Diretório para persistir os dados do ChromaDB.
        embedding_model (str): Modelo de embeddings para usar na representação vetorial.
    
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
        
        # Armazenar no ChromaDB com persistência
        print("Armazenando documentos no ChromaDB...")
        vectorstore = storageInChroma(
            documents, 
            embedding_model=embedding_model, 
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
    # Diretório de persistência para o ChromaDB
    persist_directory = "chroma"
    
    # Processar e armazenar o documento
    vectorstore = processAndStoreDocument(persist_directory=persist_directory)


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