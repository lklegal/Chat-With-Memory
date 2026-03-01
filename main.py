from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, SystemMessage, AIMessage#, AIMessageChunk
from dotenv import load_dotenv
from retrieval import Retrieval
from output_schemas import MainModelOutput

def GetStringOfImportantObservations(observations: list[dict]) -> str:
    if len(observations) == 0:
        return "No observations registered yet."
    else:
        importantObsStr = ""
        for observation in observations:
            importantObsStr += (observation["plain_text"] + "\n")
        return importantObsStr[:-1]

def GenerateSystemPrompt(importantObsStr: str, humanMessage: str, maxLessImportantObservations: int) -> str:
    lessImportantObsStr = retrieval.FetchLessImportantObservations(humanMessage, maxLessImportantObservations)
    #print(lessImportantObsStr)
    return ("You're charismatic and funny. You engage well with the user. You use your accumulated "
            "observations about the user to make your conversational approach more personalized. "
            "You have two sets of observations about the user: important and less important.\n\n"
            f"Here are your important observations so far:\n\n{importantObsStr}\n\n"
            f"Here are your less important observations so far:\n\n{lessImportantObsStr}")

if __name__ == "__main__":
    load_dotenv()
    retrieval = Retrieval()
    maxImportantObservations = 10
    maxLessImportantObservations = 10
    importantObsList = retrieval.FetchImportantObservations() 
    history = []
    model = init_chat_model("gpt-4o-mini")
    modelWithStructure = model.with_structured_output(MainModelOutput, include_raw=True)

    while True:
        prompt = input(">>>")
        if prompt == "/exit":
            retrieval.dbConnection.close()
            break
        importantObsStr = GetStringOfImportantObservations(importantObsList)
        systemPrompt = GenerateSystemPrompt(importantObsStr, prompt, maxLessImportantObservations)
        #print(systemPrompt)
        if len(history) == 0:
            history.append(SystemMessage(systemPrompt))
        else:
            history[0] = SystemMessage(systemPrompt)
        history.append(HumanMessage(prompt))
        response = modelWithStructure.invoke(history)
        structuredResponse = response["parsed"]
        history.append(AIMessage(structuredResponse.answer))
        #print(response)
        print(structuredResponse.answer)

        if structuredResponse.optionalShortUserObservation != None:
            retrieval.AddNewObservation(structuredResponse.optionalShortUserObservation,
                                        importantObsList, maxImportantObservations)
