from langchain.agents import create_agent
from langchain.agents.structured_output import ProviderStrategy
from output_schemas import MainModelOutput
from context_management import ContextManagement
from hot_output_parser import HotParser
import json
from memory_management import storer

def CallModelAndStreamResponse(agent, history: list, hotParser: HotParser) -> tuple[dict, int]:
    jsonString = ""
    totalTokens = 0
    donePrintingStream = False
    print("\nAI: ", end="", flush=True)
    for chunk in agent.stream({"messages": history}, stream_mode="messages"):
        jsonString += chunk[0].content
        if chunk[0].usage_metadata:
            totalTokens = chunk[0].usage_metadata["total_tokens"]
        if not donePrintingStream:
            printableChunk, donePrintingStream = hotParser.HotParseChunk(chunk[0].content)
            print(printableChunk, end="", flush=True)
    print("\n")
    structuredResponse = json.loads(jsonString)
    return structuredResponse, totalTokens

if __name__ == "__main__":
    contextManager = ContextManagement(maxImportantObservations=10, maxLessImportantObservations=10, 
                                       maxPromptLength=8000)
    hotParser = HotParser()
    agent = create_agent(model="gpt-4o-mini", response_format=ProviderStrategy(MainModelOutput))

    while True:
        userPrompt = input(">>>")
        promptClass = contextManager.ClassifyUserPrompt(userPrompt)
        if promptClass == "above_character_limit": continue
        elif promptClass == "quit":
            storer.ShutdownDB()
            break

        contextManager.ManageContexBeforeModelResponse(userPrompt)
        try:
            structuredResponse, totalTokens = CallModelAndStreamResponse(agent, contextManager.history, hotParser)
            possibleObservation = structuredResponse["optionalShortUserObservation"]
            if type(possibleObservation) == str:
                #heuristically treating weird edge cases that actually happened in testing (LLMs being LLMs)
                if len(possibleObservation) > 10:
                    storer.AddNewObservation(possibleObservation, contextManager.maxImportantObservations)
        except:
            print("\nError: something went wrong with processing this message.")
            contextManager.RecoverContexFromError()
            continue
        modelResponse = structuredResponse["answer"]
        contextManager.ManageContexAfterModelResponse(modelResponse, totalTokens)