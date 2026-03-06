from langchain_openai import OpenAIEmbeddings
import json

class Retrieval:
    def __FetchImportantObservationsWithId(self):
        data = self.vectorStore.GetImportantMemoriesTexts()
        for memory in data:
            memDict = {}
            memDict["id"] = memory[0]
            memDict["plain_text"] = memory[1]
            self.importantObsList.append(memDict)
    
    def __init__(self, vectorStore, importantObsList):        
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.vectorStore = vectorStore
        self.importantObsList = importantObsList
        self.__FetchImportantObservationsWithId()
    
    def FetchLessImportantObservations(self, humanMessage: str, maxLessImportantObservations: int) -> str:
        observationStrings = []
        data = []
        numberOfLessRelevantMemories = self.vectorStore.CountLessRelevantMemories()[0]
        if numberOfLessRelevantMemories > maxLessImportantObservations:
            humanMessageVector = self.embeddings.embed_query(humanMessage)
            data = self.vectorStore.GetTopK_LessImportantObservations(json.dumps(humanMessageVector), 
                                                                    maxLessImportantObservations)
        elif numberOfLessRelevantMemories > 0:
            data = self.vectorStore.GetAllLessImportantObservations()
        for row in data:
            observationStrings.append(row[0])
        return observationStrings
    
    def FetchImportantObservationsStrings(self) -> list[str]:
        observationsStrs = []
        for obs in self.importantObsList:
            observationsStrs.append(obs["plain_text"])
        return observationsStrs