from langchain.messages import HumanMessage, SystemMessage, AIMessage, AIMessageChunk
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.agents.structured_output import ProviderStrategy
from dotenv import load_dotenv
from output_schemas import MainModelOutput
from context_management import ContextManagement
from hot_output_parser import HotParser
from retrieval import Retrieval
import json

if __name__ == "__main__":
    load_dotenv()
    retriever = Retrieval()
    contextManager = ContextManagement(10, 10, 8000, retriever)
    hotParser = HotParser()
    agent = create_agent(model="gpt-4o-mini", response_format=ProviderStrategy(MainModelOutput))

    while True:
        prompt = input(">>>")
        if len(prompt) > contextManager.maxPromptLength:
            print(f"Error: This message is over the {contextManager.maxPromptLength} characters limit.")
            continue
        if prompt == "/exit":
            contextManager.retriever.dbConnection.close()
            break
        systemPrompt = contextManager.GenerateSystemPrompt(prompt)
        #print(systemPrompt)
        if len(contextManager.history) == 0:
            contextManager.history.append(SystemMessage(systemPrompt))
        else:
            contextManager.history[0] = SystemMessage(systemPrompt)
        contextManager.history.append(HumanMessage(prompt))
        hasIdentifiedTheAI = False
        jsonString = ""
        totalTokens = 0
        donePrintingStream = False
        print("\nAI:", end="", flush=True)
        for chunk in agent.stream({"messages": contextManager.history}, stream_mode="messages"):
            jsonString += chunk[0].content
            if chunk[0].usage_metadata:
                totalTokens = chunk[0].usage_metadata["total_tokens"]
            if not donePrintingStream:
                printableChunk, donePrintingStream = hotParser.HotParseChunk(chunk[0].content)
                print(printableChunk, end="", flush=True)
        print("\n")
        structuredResponse = json.loads(jsonString)
        contextManager.history.append(AIMessage(structuredResponse["answer"]))
        contextManager.ManageContextCompaction(totalTokens)
        contextManager.PersistConversation()
        if structuredResponse["optionalShortUserObservation"] != None:
            #weird edge case that actually happened in testing (LLMs being LLMs):
            if structuredResponse["optionalShortUserObservation"] != "":
                retriever.AddNewObservation(structuredResponse["optionalShortUserObservation"],
                                            contextManager.maxImportantObservations)