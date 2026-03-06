from langchain.chat_models import init_chat_model
from output_schemas import SummarizationModelOutput
from pydantic import BaseModel
from langchain.messages import HumanMessage, SystemMessage, AIMessage
from memory_management import retriever
from typing import Literal
import os
import json

class ConversationStateSchema(BaseModel):
    generalSummary: str
    lastBatchSummary: str
    indexOfBatchCutoff: int | None
    history: list[str]

class ContextManagement:
    def __TryToRecoverSavedState(self) -> tuple[dict, bool]:
        savedState = {}
        jsonFileName = "conversation_state.json"
        if not os.path.exists(jsonFileName):
            with open(jsonFileName, "w") as file:
                json.dump({}, file)
        hasValidSavedState = True
        with open(jsonFileName, "r") as file:
            savedState = json.load(file)
        try:
           ConversationStateSchema.model_validate_json(json.dumps(savedState))
        except:
            hasValidSavedState = False
        return savedState, hasValidSavedState
    
    def __ApplySavedState(self, savedState: dict):
        self.generalSummary = savedState["generalSummary"]
        self.lastBatchSummary = savedState["lastBatchSummary"]
        self.indexOfBatchCutoff = savedState["indexOfBatchCutoff"]
        for i, message in enumerate(savedState["history"]):
            if i == 0:
                self.history.append(SystemMessage(message))
            elif i%2 == 0:
                self.history.append(AIMessage(message))
            else:
                self.history.append(HumanMessage(message))

    def __init__(self, maxImportantObservations, maxLessImportantObservations, maxPromptLength):
        self.maxImportantObservations = maxImportantObservations
        self.maxLessImportantObservations = maxLessImportantObservations
        self.maxPromptLength = maxPromptLength
        self.batchSizeInTokens = 5000
        self.noSummaryString = "No summary yet. The conversation is not yet long enough for this summary to be necessary."
        self.history = []
        summarizationModel = init_chat_model("gpt-4o-mini", max_tokens=2000)
        self.summarizationModelWithStructure = summarizationModel.with_structured_output(SummarizationModelOutput)
        savedState, hasValidSavedState = self.__TryToRecoverSavedState()
        if hasValidSavedState:
            self.__ApplySavedState(savedState)
        else:
            self.generalSummary = self.noSummaryString
            self.lastBatchSummary = self.noSummaryString
            self.indexOfBatchCutoff = None

    def __GetStringOfObservations(self, observations: list[str]) -> str:
        if len(observations) == 0:
            return "No observations registered yet."
        else:
            importantObsStr = ""
            for observation in observations:
                importantObsStr += (observation + "\n")
            return importantObsStr[:-1]

    def __GenerateSystemPromptStr(self, humanMessage: str) -> str:
        importantObs = retriever.FetchImportantObservationsStrings()
        lessImportantObs = retriever.FetchLessImportantObservations(humanMessage, self.maxLessImportantObservations)
        importantObsStr = self.__GetStringOfObservations(importantObs)
        lessImportantObsStr = self.__GetStringOfObservations(lessImportantObs)
        return ("You're chatting with a user as a charismatic conversation partner. You subtly and implicitly use your "
                "knowledge about the user to make your conversational approach more personal, but without exaggerating. " \
                "You keep your responses short, like in a more casual conversation. "
                "You have two sets of observations about the user: important and less important.\n\n"
                f"Here are your important observations so far:\n\n{importantObsStr}\n\n"
                f"Here are your less important observations so far:\n\n{lessImportantObsStr}\n\n"
                "You should know that this conversation is periodically summarized, to keep responses "
                "fast for the user and not fill your context window. Here is a summary of the conversation, "
                f"except the most recent part: '{self.generalSummary}' \n\n" 
                f"And here is the summary of the most recent part: '{self.lastBatchSummary}'")
    
    def __GetBatchStr(self) -> str:
        batchStr = ""
        for i in range(1, self.indexOfBatchCutoff+1):
            whoWrote = ""
            if i % 2 != 0: whoWrote = "Human: "
            else: whoWrote = "Assistant: "
            batchStr += whoWrote + self.history[i].content + "\n\n"
        return batchStr[:-2]
    
    def __SummarizeBatch(self, batchStr: str) -> str:
        summarizationPrompt = ("You are tasked with summarizing a conversation snippet between a "
        "human and an AI assistant. For context, here is a summary of the older interactions: "
        f"\n\n'{self.generalSummary}'\n\nAnd here is a summary of the most recent conversation snippet " 
        f"to be summarized: \n\n'{self.lastBatchSummary}'\n\nOne or both summaries may be absent, if the "
        "conversation is still too short for summaries. In any case, given this context, here is the "
        "actual conversation snippet you're tasked with summarizing. Please, keep it under 2000 characters: "
        f"\n\n'{batchStr}'")
        summarizationModelResponse = self.summarizationModelWithStructure.invoke(summarizationPrompt)
        return summarizationModelResponse.summary
    
    def __MergeSummaries(self) -> str:
        summaryMerge = ""
        if self.generalSummary == self.noSummaryString:
            summaryMerge = self.lastBatchSummary
        else:
            summarizationMergerPrompt = ("You are tasked with merging two summaries of a conversation "
            "between a human and an AI assistant. The first summary is of the entire conversation, except "
            "the most recent part. The second summary is of the most recent part of the conversation. Your "
            "job is to merge them into a single summary, giving more weight to the older summary, since "
            "it represents most of the conversation. Please merge them together, and summarize the merger, so that "
            "the result is a unified summary of the same length as any one of the previous two summaries. "
            f"So here is the summary of the older parts of the conversation: \n\n'{self.generalSummary}\n\n'"
            f"And here is the summary of the most recent part of the conversation: \n\n'{self.lastBatchSummary}'\n\n "
            "Please, keep the merged summary at less than 2000 characters.")
            summarizationModelResponse = self.summarizationModelWithStructure.invoke(summarizationMergerPrompt)
            summaryMerge = summarizationModelResponse.summary
        return summaryMerge
    
    def __CompactContext(self):
        batchStr = self.__GetBatchStr()
        lastBatchSummary = self.__SummarizeBatch(batchStr)
        summaryMerge = self.__MergeSummaries()
        self.generalSummary = summaryMerge
        self.lastBatchSummary = lastBatchSummary
        self.history = [self.history[0]] + self.history[self.indexOfBatchCutoff+1:]
        #print(f"########## General Summary ##########\n{self.generalSummary}##########")
        #print(f"########## Last Batch Summary ##########\n{self.lastBatchSummary}##########")

    def ManageContextCompaction(self, totalTokens: int):
        if totalTokens >= self.batchSizeInTokens:
            if self.indexOfBatchCutoff == None:
                self.indexOfBatchCutoff = len(self.history)-1
            elif totalTokens >= self.batchSizeInTokens*2:
                self.__CompactContext()
                self.indexOfBatchCutoff = None

    def PersistConversation(self):
        data = {
            "generalSummary": self.generalSummary,
            "lastBatchSummary": self.lastBatchSummary,
            "indexOfBatchCutoff": self.indexOfBatchCutoff,
            "history": [message.content for message in self.history]
        }
        with open("conversation_state.json", "w") as file:
            json.dump(data, file, indent=4)

    def ClassifyUserPrompt(self, prompt: str) -> Literal["normal", "above_character_limit", "quit"]:
        if len(prompt) > self.maxPromptLength:
            print(f"Error: This message is over the {self.maxPromptLength} characters limit.")
            return "above_character_limit"
        elif prompt == "/exit":
            return "quit"
        else:
            return "normal"
        
    def ManageContexBeforeModelResponse(self, humanMessage: str):
        systemPrompt = self.__GenerateSystemPromptStr(humanMessage)
        if len(self.history) == 0:
            self.history.append(SystemMessage(systemPrompt))
        else:
            self.history[0] = SystemMessage(systemPrompt)
        self.history.append(HumanMessage(humanMessage))

    def ManageContexAfterModelResponse(self, modelResponse: str, totalTokens: int):
        self.history.append(AIMessage(modelResponse))
        self.ManageContextCompaction(totalTokens)
        self.PersistConversation()

    def RecoverContexFromError(self):
        self.history = self.history[:-1]