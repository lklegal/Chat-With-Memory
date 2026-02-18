from langchain_openai import OpenAIEmbeddings
import vector_store
import numpy as np

class Retrieval:
    def __init__(self):        
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.embeddings_dim = 1536
        self.dbConnection, self.dbCursor = vector_store.InitializeDB()
        self.dbCursor.execute("SELECT vector FROM memories WHERE category = 'relevance_scorer';")
        relevanceScorerQueryResult = self.dbCursor.fetchall()
        if relevanceScorerQueryResult == []:
            relevanceScoringText = "This is the MOST RELEVANT THING EVER abour a user, \
                 literally their DEFINING CHARACTERISTIC, the core of their very soul."
            self.relevanceScoringVector = self.embeddings.embed_query(relevanceScoringText)
            self.dbCursor.execute("INSERT INTO memories (vector, plain_text, category) \
                                VALUES (vec_f32(?), ?, ?);", (str(self.relevanceScoringVector), \
                                    relevanceScoringText, 'relevance_scorer'))
            self.dbConnection.commit()
        else:
            blob = relevanceScorerQueryResult[0][0]
            self.relevanceScoringVector = (np.frombuffer(blob, dtype=np.float32)).tolist()