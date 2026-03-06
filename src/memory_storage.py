from langchain_openai import OpenAIEmbeddings
import numpy as np
import json
from output_schemas import DeduplicationModelOutput
from langchain.chat_models import init_chat_model

class Storage:
    def __init__(self, vectorStore, importantObsList):
        self.vectorStore = vectorStore
        self.importantObsList = importantObsList
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        deduplicationModel = init_chat_model("gpt-4o-mini")
        self.deduplicationModelWithStructure = deduplicationModel.with_structured_output(DeduplicationModelOutput)
        stabilityAnchorQueryResult = self.vectorStore.GetAnchor('stability_anchor')
        if stabilityAnchorQueryResult == None:
            stabilityAnchor = ("A stable, long-term characteristic, preference, goal, "
            "constraint, or behavioral tendency of the user. This includes enduring response "
            "style preferences, recurring values, consistent habits, long-term projects, "
            "professional focus, technical stack, tool usage rules, or repeated patterns "
            "across conversations. It represents something that would likely remain true "
            "across many future interactions. It does NOT include temporary moods, one-time "
            "events, or situational emotional states.")
            self.stabilityAnchorVector = self.embeddings.embed_query(stabilityAnchor)
            self.vectorStore.InsertAchor(stabilityAnchor, json.dumps(self.stabilityAnchorVector), 'stability_anchor')
        else:
            blob = stabilityAnchorQueryResult[0]
            self.stabilityAnchorVector = (np.frombuffer(blob, dtype=np.float32)).tolist()
        
        futureUsefulnessAnchorQueryResult = self.vectorStore.GetAnchor('future_usefulness_anchor')
        if futureUsefulnessAnchorQueryResult == None:
            futureUsefulnessAnchor = ("Information about the user that would meaningfully "
            "improve the assistant’s ability to provide better answers in future "
            "conversations. This includes preferences about response length, tone, "
            "structure, technical depth, programming languages, tools, constraints, "
            "recurring goals, or communication style. It is the kind of information that, "
            "if remembered, would reduce friction or increase relevance in many future "
            "interactions. It excludes trivial personal trivia or facts unlikely to affect "
            "future responses.")
            self.futureUsefulnessAnchorVector = self.embeddings.embed_query(futureUsefulnessAnchor)
            self.vectorStore.InsertAchor(futureUsefulnessAnchor, json.dumps(self.futureUsefulnessAnchorVector), 'future_usefulness_anchor')
        else:
            blob = futureUsefulnessAnchorQueryResult[0]
            self.futureUsefulnessAnchorVector = (np.frombuffer(blob, dtype=np.float32)).tolist()
        
        nonImportanceAnchorQueryResult = self.vectorStore.GetAnchor('non_importance_anchor')
        if nonImportanceAnchorQueryResult == None:    
            nonImportanceAnchor = ("A temporary mood, fleeting emotion, one-time event, "
            "situational update, random fact, or trivial personal detail that is unlikely "
            "to influence future conversations in a meaningful way. This includes transient "
            "feelings, daily activities, casual preferences with no impact on interaction "
            "style, or information that would not help the assistant adapt its behavior over "
            "time.")
            self.nonImportanceAnchorVector = self.embeddings.embed_query(nonImportanceAnchor)
            self.vectorStore.InsertAchor(nonImportanceAnchor, json.dumps(self.nonImportanceAnchorVector), 'non_importance_anchor')
        else:
            blob = nonImportanceAnchorQueryResult[0]
            self.nonImportanceAnchorVector = (np.frombuffer(blob, dtype=np.float32)).tolist()
    
    def __FindPossibleDuplicate(self, observation: str, observationVector: list[float]) -> bool:
        duplicateCandidates = self.vectorStore.GetPossibleDuplicates(json.dumps(observationVector))
        duplicateCandidatesStr = ""
        for index, candidate in enumerate(duplicateCandidates, 1):
            duplicateCandidatesStr += f"{index}- '{candidate[0]}';\n"
        duplicateCandidatesStr = duplicateCandidatesStr[:-1]
        deduplicationPrompt = ("Take a look at this list of observations about a chatbot user:"
        f"\n\n{duplicateCandidatesStr} \n\nPlease, analyze if this new observation about the same user "
        f"is a duplicate of any of the previous observations: '{observation}'\n\nRemember, a "
        "duplicate observation can be near identical, yes, but also a paraphrase, or something "
        "that was already alluded to before (even if fragmented in multiple observations), or "
        "something that just has the same general meaning as previous observations, even if phrased "
        "very differently.")
        modelOutput = self.deduplicationModelWithStructure.invoke(deduplicationPrompt)
        #print(f"########\nDEDUPLICATION FUNCTION MODEL INPUT: {deduplicationPrompt}\nOUTPUT: {modelOutput}\n########\n\n")
        return modelOutput.newObservationIsDuplicate
        
    def __ComputeObservationRelevance(self, observationVector: list[float]) -> float:
        data = self.vectorStore.GetSimilarityToAnchors(json.dumps(self.stabilityAnchorVector), 
        json.dumps(self.futureUsefulnessAnchorVector), json.dumps(self.nonImportanceAnchorVector), 
        json.dumps(observationVector))
        stabilityScore = data[0]
        futureUsefulnessScore = data[1]
        nonImportanceScore = data[2]
        relevanceScore = (stabilityScore + futureUsefulnessScore + (1-nonImportanceScore))/3
        return relevanceScore

    def __FindLeastRelevantButStillRelevantObservation(self) -> list:
        data = self.vectorStore.GetImportantMemoriesVectors()
        leastRelevant = {}
        smallestRelevanceScore = -1
        for row in data:
            id = row[0]
            vector = (np.frombuffer(row[1], dtype=np.float32)).tolist()
            relevanceScore = self.__ComputeObservationRelevance(vector)
            # confusingly, the higher the number, the smaller the relevance
            if relevanceScore > smallestRelevanceScore:
                smallestRelevanceScore = relevanceScore
                leastRelevant = {"id": id, "relevanceScore": relevanceScore}
        return leastRelevant

    def __GetObservationID(self, plainTextObservation: str) -> int:
        newObservationId = self.vectorStore.GetObservationID(plainTextObservation)[0][0]
        return newObservationId
    
    def AddNewObservation(self, observation: str, maxImportantObservations: int):
        observationVector = self.embeddings.embed_query(observation)
        if len(self.importantObsList) > 0:
            hasDuplicate = self.__FindPossibleDuplicate(observation, observationVector)
            if hasDuplicate:
                return
        if len(self.importantObsList) >= maxImportantObservations:
            leastRelevantButStillRelevantObservation = self.__FindLeastRelevantButStillRelevantObservation()
            newObservationRelevanceScore = self.__ComputeObservationRelevance(observationVector)
            if newObservationRelevanceScore < leastRelevantButStillRelevantObservation["relevanceScore"]:
                index = 0
                newLessRelevantObservation = self.importantObsList[0]
                while newLessRelevantObservation["id"] != leastRelevantButStillRelevantObservation["id"]:
                    index += 1
                    newLessRelevantObservation = self.importantObsList[index]
                self.vectorStore.SetMemoryAsLessRelevant(newLessRelevantObservation["id"])
                self.vectorStore.InsertObservation(observation, json.dumps(observationVector), "important")
                newObservationId = self.__GetObservationID(observation)
                self.importantObsList[index] = {
                    "id": newObservationId,
                    "plain_text": observation
                }
            else:
                self.vectorStore.InsertObservation(observation, json.dumps(observationVector), "less_relevant")
        else:
            self.vectorStore.InsertObservation(observation, json.dumps(observationVector), "important")
            newObservationId = self.__GetObservationID(observation)
            self.importantObsList.append({
                "id": newObservationId,
                "plain_text": observation
            })
        print(f"Memory updated: {observation}")

    def ShutdownDB(self):
        self.vectorStore.Close()