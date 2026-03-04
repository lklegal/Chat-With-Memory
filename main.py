from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, SystemMessage, AIMessage#, AIMessageChunk
from dotenv import load_dotenv
from output_schemas import MainModelOutput
from context_management import ContextManagement
from retrieval import Retrieval

if __name__ == "__main__":
    load_dotenv()
    retriever = Retrieval()
    contextManager = ContextManagement(10, 10, 8000, retriever)
    model = init_chat_model("gpt-4o-mini")
    modelWithStructure = model.with_structured_output(MainModelOutput, include_raw=True)

    while True:
        prompt = input(">>>")
        if len(prompt) > contextManager.maxPromptLength:
            print(f"Error: This message is over the {contextManager.maxPromptLength} characters limit.")
            continue
        if prompt == "/exit":
            contextManager.retriever.dbConnection.close()
            contextManager.PersistConversation()
            break
        systemPrompt = contextManager.GenerateSystemPrompt(prompt)
        #print(systemPrompt)
        if len(contextManager.history) == 0:
            contextManager.history.append(SystemMessage(systemPrompt))
        else:
            contextManager.history[0] = SystemMessage(systemPrompt)
        contextManager.history.append(HumanMessage(prompt))
        response = modelWithStructure.invoke(contextManager.history)
        structuredResponse = response["parsed"]
        contextManager.history.append(AIMessage(structuredResponse.answer))
        totalTokens = response["raw"].usage_metadata["total_tokens"]
        contextManager.ManageContextCompaction(totalTokens)
        print(structuredResponse.answer)

        if structuredResponse.optionalShortUserObservation != None:
            #weird edge case that actually happened in testing (LLMs being LLMs):
            if structuredResponse.optionalShortUserObservation != "":
                retriever.AddNewObservation(structuredResponse.optionalShortUserObservation,
                                            contextManager.maxImportantObservations)