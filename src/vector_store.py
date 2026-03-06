import sqlite3
import sqlite_vec
from typing import Literal

class VectorStore:
    def __init__(self):
        self.conn = sqlite3.connect("memory.db")
        self.conn.enable_load_extension(True)
        sqlite_vec.load(self.conn)
        self.cursor = self.conn.cursor()
        self.cursor.execute("CREATE VIRTUAL TABLE IF NOT EXISTS memories USING vec0(\
            vector FLOAT[1536],\
            plain_text TEXT,\
            category TEXT)\
        ")
        self.conn.commit()

    def GetImportantMemoriesTexts(self) -> list:
        self.cursor.execute("SELECT rowid, plain_text FROM memories WHERE category = 'important';")
        return self.cursor.fetchall()
    
    def InsertAchor(self, anchorText: str, anchorVectorStr: str, anchorName: str) -> str:
        self.cursor.execute(("INSERT INTO memories (vector, plain_text, category) "
        "VALUES (vec_f32(?), ?, ?);"), (anchorVectorStr, anchorText, anchorName))
        self.conn.commit()

    def GetAnchor(self, anchorName: str) -> tuple:
        self.cursor.execute("SELECT vector FROM memories WHERE category = ?;", (anchorName,))
        return self.cursor.fetchone()
    
    def CountLessRelevantMemories(self) -> tuple:
        self.cursor.execute(("SELECT COUNT(1) FROM memories WHERE category = 'less_relevant';"))
        return self.cursor.fetchone()
    
    def GetTopK_LessImportantObservations(self, humanMessageVectorStr: str, maxLessImportantObservations: int) -> list:
        self.cursor.execute(("SELECT plain_text FROM memories WHERE category = 'less_relevant' "
                             "ORDER BY vec_distance_cosine(vector, vec_f32(?)) ASC LIMIT ?;"),
                               (humanMessageVectorStr, maxLessImportantObservations))
        return self.cursor.fetchall()
    
    def GetAllLessImportantObservations(self) -> list:
        self.cursor.execute("SELECT plain_text FROM memories WHERE category = 'less_relevant';")
        return self.cursor.fetchall()
    
    def GetPossibleDuplicates(self, observationVectorStr: str) -> list:
        self.cursor.execute(("SELECT plain_text FROM memories WHERE category IN ('important', 'less_relevant') ORDER BY " 
        "vec_distance_cosine(vec_f32(?), vector) ASC LIMIT 10;"), (observationVectorStr,))
        return self.cursor.fetchall()
    
    def GetSimilarityToAnchors(self, stabilityAnchorVectorStr, futureUsefulnessAnchorVectorStr,
                               nonImportanceAnchorVectorStr, observationVectorStr):
        self.cursor.execute(("SELECT vec_distance_cosine(vec_f32(?), vec_f32(?)), "
        "vec_distance_cosine(vec_f32(?), vec_f32(?)), vec_distance_cosine(vec_f32(?), vec_f32(?));"), 
        (stabilityAnchorVectorStr, observationVectorStr, futureUsefulnessAnchorVectorStr, observationVectorStr, 
         nonImportanceAnchorVectorStr, observationVectorStr))
        return self.cursor.fetchone()
    
    def GetImportantMemoriesVectors(self) -> list:
        self.cursor.execute("SELECT rowid, vector FROM memories WHERE category = 'important';")
        return self.cursor.fetchall()
    
    def InsertObservation(self, observationText: str, observationVectorStr: str, category: Literal["important", "less_relevant"]):
        self.cursor.execute("INSERT INTO memories (vector, plain_text, category) VALUES (vec_f32(?), ?, ?);", 
                            (observationVectorStr, observationText, category))
        self.conn.commit()

    def GetObservationID(self, plainTextObservation: str) -> int:
        self.cursor.execute("SELECT rowid FROM memories WHERE plain_text = ?", (plainTextObservation,))
        return self.cursor.fetchall()
    
    def SetMemoryAsLessRelevant(self, obsId: int):
        self.cursor.execute("UPDATE memories SET category = 'less_relevant' WHERE rowid = ?", (obsId,))
        self.conn.commit()

    def Close(self):
        self.conn.close()