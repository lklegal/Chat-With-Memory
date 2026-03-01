from langchain_openai import OpenAIEmbeddings
import vector_store
import numpy as np
import json
from typing import Literal
from output_schemas import DeduplicationModelOutput
from langchain.chat_models import init_chat_model

class Retrieval:
    def __InsertAchorIntoDB(self, anchorText: str, anchorVector: list[float], anchorName: str):
            self.dbCursor.execute(("INSERT INTO memories (vector, plain_text, category) "
            "VALUES (vec_f32(?), ?, ?);"), (json.dumps(anchorVector), anchorText, anchorName))
            self.dbConnection.commit()
    
    def __init__(self):        
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.embeddings_dim = 1536
        self.dbConnection, self.dbCursor = vector_store.InitializeDB()
        deduplicationModel = init_chat_model("gpt-4o-mini")
        self.deduplicationModelWithStructure = deduplicationModel.with_structured_output(DeduplicationModelOutput)
        self.dbCursor.execute("SELECT vector FROM memories WHERE category = 'stability_anchor';")
        stabilityAnchorQueryResult = self.dbCursor.fetchall()
        if stabilityAnchorQueryResult == []:
            stabilityAnchor = ("A stable, long-term characteristic, preference, goal, "
            "constraint, or behavioral tendency of the user. This includes enduring response "
            "style preferences, recurring values, consistent habits, long-term projects, "
            "professional focus, technical stack, tool usage rules, or repeated patterns "
            "across conversations. It represents something that would likely remain true "
            "across many future interactions. It does NOT include temporary moods, one-time "
            "events, or situational emotional states.")
            self.stabilityAnchorVector = self.embeddings.embed_query(stabilityAnchor)
            self.__InsertAchorIntoDB(stabilityAnchor, self.stabilityAnchorVector, 'stability_anchor')
        else:
            blob = stabilityAnchorQueryResult[0][0]
            self.stabilityAnchorVector = (np.frombuffer(blob, dtype=np.float32)).tolist()
        
        self.dbCursor.execute("SELECT vector FROM memories WHERE category = 'future_usefulness_anchor';")
        futureUsefulnessAnchorQueryResult = self.dbCursor.fetchall()
        if futureUsefulnessAnchorQueryResult == []:
            futureUsefulnessAnchor = ("Information about the user that would meaningfully "
            "improve the assistant’s ability to provide better answers in future "
            "conversations. This includes preferences about response length, tone, "
            "structure, technical depth, programming languages, tools, constraints, "
            "recurring goals, or communication style. It is the kind of information that, "
            "if remembered, would reduce friction or increase relevance in many future "
            "interactions. It excludes trivial personal trivia or facts unlikely to affect "
            "future responses.")
            self.futureUsefulnessAnchorVector = self.embeddings.embed_query(futureUsefulnessAnchor)
            self.__InsertAchorIntoDB(futureUsefulnessAnchor, self.futureUsefulnessAnchorVector, 'future_usefulness_anchor')
        else:
            blob = futureUsefulnessAnchorQueryResult[0][0]
            self.futureUsefulnessAnchorVector = (np.frombuffer(blob, dtype=np.float32)).tolist()
        
        self.dbCursor.execute("SELECT vector FROM memories WHERE category = 'non_importance_anchor';")
        nonImportanceAnchorQueryResult = self.dbCursor.fetchall()
        if nonImportanceAnchorQueryResult == []:    
            nonImportanceAnchor = ("A temporary mood, fleeting emotion, one-time event, "
            "situational update, random fact, or trivial personal detail that is unlikely "
            "to influence future conversations in a meaningful way. This includes transient "
            "feelings, daily activities, casual preferences with no impact on interaction "
            "style, or information that would not help the assistant adapt its behavior over "
            "time.")
            self.nonImportanceAnchorVector = self.embeddings.embed_query(nonImportanceAnchor)
            self.__InsertAchorIntoDB(nonImportanceAnchor, self.nonImportanceAnchorVector, 'non_importance_anchor')
        else:
            blob = nonImportanceAnchorQueryResult[0][0]
            self.nonImportanceAnchorVector = (np.frombuffer(blob, dtype=np.float32)).tolist()

    def FetchImportantObservations(self) -> list[dict]:
        observations = []
        self.dbCursor.execute("SELECT rowid, plain_text FROM memories WHERE category = 'important';")
        data = self.dbCursor.fetchall()
        for memory in data:
            memDict = {}
            memDict["id"] = memory[0]
            memDict["plain_text"] = memory[1]
            observations.append(memDict)
        return observations
    
    def FetchLessImportantObservations(self, humanMessage: str, maxLessImportantObservations: int) -> str:
        lessImportantObsStr = ""
        self.dbCursor.execute("SELECT EXISTS (\
            SELECT 1 FROM memories WHERE category = 'less_relevant'\
        );")
        if (self.dbCursor.fetchall())[0][0] == 1:
            humanMessageVector = self.embeddings.embed_query(humanMessage)
            self.dbCursor.execute("SELECT plain_text FROM memories WHERE \
            category = 'less_relevant' ORDER BY vec_distance_cosine(vector, vec_f32(?)) ASC \
            LIMIT ?;", (json.dumps(humanMessageVector), maxLessImportantObservations))
            data = self.dbCursor.fetchall()
            for observation in data:
                lessImportantObsStr += (observation[0] + "\n")
            lessImportantObsStr = lessImportantObsStr[:-1]
        else:
            lessImportantObsStr = "No observations registered yet."
        return lessImportantObsStr
    
    def __FindPossibleDuplicate(self, observation: str, observationVector: list[float]) -> bool:
        self.dbCursor.execute(("SELECT plain_text FROM memories WHERE category IN ('important', 'less_relevant') ORDER BY " 
        "vec_distance_cosine(vec_f32(?), vector) ASC LIMIT 10;"), (json.dumps(observationVector),))
        duplicateCandidates = self.dbCursor.fetchall()
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
        observationVectorStr = json.dumps(observationVector)
        self.dbCursor.execute(("SELECT vec_distance_cosine(vec_f32(?), vec_f32(?)), "
        "vec_distance_cosine(vec_f32(?), vec_f32(?)), vec_distance_cosine(vec_f32(?), vec_f32(?));"), 
        (json.dumps(self.stabilityAnchorVector), observationVectorStr, 
         json.dumps(self.futureUsefulnessAnchorVector),observationVectorStr,
         json.dumps(self.nonImportanceAnchorVector), observationVectorStr))
        data = self.dbCursor.fetchone()
        stabilityScore = data[0]
        futureUsefulnessScore = data[1]
        nonImportanceScore = data[2]
        relevanceScore = (stabilityScore + futureUsefulnessScore + (1-nonImportanceScore))/3
        return relevanceScore

    def __FindLeastRelevantButStillRelevantObservation(self) -> list:
        self.dbCursor.execute("SELECT rowid, vector FROM memories WHERE category = 'important';")
        data = self.dbCursor.fetchall()
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
    
    def __InsertObservationIntoDB(self, observation: str, observationVector: list[float],
                                     category: Literal["important", "less_relevant"]):
        self.dbCursor.execute("INSERT INTO memories (vector, plain_text, category) \
                        VALUES (vec_f32(?), ?, ?);", 
                        (json.dumps(observationVector), observation, category))
        self.dbConnection.commit()

    def __GetObservationID(self, plainTextObservation: str) -> int:
        self.dbCursor.execute("SELECT rowid FROM memories WHERE plain_text = ?",
                        (plainTextObservation,))
        newObservationId = (self.dbCursor.fetchall())[0][0]
        return newObservationId
    
    def AddNewObservation(self, observation: str, importantObservations: list[dict], maxImportantObservations: int):
        observationVector = self.embeddings.embed_query(observation)
        if len(importantObservations) > 0:
            hasDuplicate = self.__FindPossibleDuplicate(observation, observationVector)
            if hasDuplicate:
                return
        print(f"Memory updated: {observation}")
        if len(importantObservations) >= maxImportantObservations:
            leastRelevantButStillRelevantObservation = self.__FindLeastRelevantButStillRelevantObservation()
            newObservationRelevanceScore = self.__ComputeObservationRelevance(observationVector)
            if newObservationRelevanceScore < leastRelevantButStillRelevantObservation["relevanceScore"]:
                index = 0
                newLessRelevantObservation = importantObservations[0]
                while newLessRelevantObservation["id"] != leastRelevantButStillRelevantObservation["id"]:
                    index += 1
                    newLessRelevantObservation = importantObservations[index]
                self.dbCursor.execute("UPDATE memories SET category = 'less_relevant' WHERE \
                                      rowid = ?", (newLessRelevantObservation["id"],))
                self.__InsertObservationIntoDB(observation, observationVector, "important")
                newObservationId = self.__GetObservationID(observation)
                importantObservations[index] = {
                    "id": newObservationId,
                    "plain_text": observation
                }
            else:
                self.__InsertObservationIntoDB(observation, observationVector, "less_relevant")
        else:
            self.__InsertObservationIntoDB(observation, observationVector, "important")
            newObservationId = self.__GetObservationID(observation)
            importantObservations.append({
                "id": newObservationId,
                "plain_text": observation
            })
        #print(f"########\nADD NEW OBSERVATION FUNCTION OUTPUT: {importantObservations}\n########\n\n")