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
        thereExistsSomeLessRelevantMemory = bool(self.vectorStore.CheckIfThereIsAnyLessRelevantMemory()[0])
        if thereExistsSomeLessRelevantMemory:
            humanMessageVector = self.embeddings.embed_query(humanMessage)
            data = self.vectorStore.GetLessImportantObservations(json.dumps(humanMessageVector), maxLessImportantObservations)
            for row in data:
                observationStrings.append(row[0])
        return observationStrings
    
    def FetchImportantObservationsStrings(self) -> list[str]:
        observationsStrs = []
        for obs in self.importantObsList:
            observationsStrs.append(obs["plain_text"])
        return observationsStrs