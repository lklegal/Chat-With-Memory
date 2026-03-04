from langchain.chat_models import init_chat_model
from output_schemas import SummarizationModelOutput
import json

class ContextManagement:
    def __init__(self, maxImportantObservations, maxLessImportantObservations, maxPromptLength, retriever):
        savedState = {}
        with open("conversation_state.json", "w") as file:
            savedState = json.load(file)
        # here goes the saved state loading logic
        self.maxImportantObservations = maxImportantObservations
        self.maxLessImportantObservations = maxLessImportantObservations
        self.maxPromptLength = maxPromptLength
        self.batchSizeInTokens = 15000
        self.noSummaryString = "No summary yet. The conversation is not yet long enough for this summary to be necessary."
        self.generalSummary = self.noSummaryString
        self.lastBatchSummary = self.noSummaryString
        self.indexOfBatchCutoff = None
        summarizationModel = init_chat_model("gpt-4o-mini", max_tokens=2000)
        self.summarizationModelWithStructure = summarizationModel.with_structured_output(SummarizationModelOutput)
        #tech debt, refactor later
        self.retriever = retriever
        self.history = []

    def __GetStringOfObservations(self, observations: list) -> str:
        if len(observations) == 0:
            return "No observations registered yet."
        else:
            importantObsStr = ""
            for observation in observations:
                if type(observation) == dict:
                    importantObsStr += (observation["plain_text"] + "\n")
                else: #has to be a string
                    importantObsStr += (observation + "\n")
            return importantObsStr[:-1]

    def GenerateSystemPrompt(self, humanMessage: str) -> str:
        lessImportantObs = self.retriever.FetchLessImportantObservations(humanMessage, self.maxLessImportantObservations)
        importantObsStr = self.__GetStringOfObservations(self.retriever.importantObsList)
        lessImportantObsStr = self.__GetStringOfObservations(lessImportantObs)
        #print(lessImportantObsStr)
        return ("You're chatting with a user as a charismatic conversation partiner. You subtly and implicitly use your "
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
            if i % 2 != 0:
                whoWrote = "Human: "
            else:
                whoWrote = "Assistant: "
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