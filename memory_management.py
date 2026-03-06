from dotenv import load_dotenv

def InitializeMemoryManagers():
    from memory_retrieval import Retrieval
    from memory_storage import Storage
    from vector_store import VectorStore
    vs = VectorStore()
    importantObsList = []
    retriever = Retrieval(vs, importantObsList)
    storer = Storage(vs, importantObsList)
    return storer, retriever

load_dotenv()
storer, retriever = InitializeMemoryManagers()