from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, SystemMessage, AIMessage#, AIMessageChunk
#from langchain.tools import tool
from dotenv import load_dotenv
from pydantic import BaseModel, Field
import json
from retrieval import Retrieval

class ModelOutput(BaseModel):
    """
    This is the format of your responses to queries.
    """
    answer: str = Field(description="This is your response to the user. It's what they see.")
    reasoningAboutShortUserObservation: str = Field(description="This is where you write your reasoning \
    as to wheather or not to write a new short oservation about the user. You only write relevant things \
    you want to remember about the user. So here is where you decide if something is relevant enough\
    to be registered. It's fine to not write anything. Desireable, even. Only decide to write when relevant.\
    Don't write things you already know.")
    optionalShortUserObservation: str | None = Field(description="This is where you write\
    something short about the user you want to remember, provided you concluded in your reasoning\
    that the observation you're about to write is relevant. It's fine to not write anything. Desireable, even.\
    Only write if you decided it's something relevant. If nothing is written, the type of this field must be None.\
    Examples of observations: 'User has a cat', or 'User is afraid of heights', or 'User likes coffee'. Things like that.")

def GetStringOfObservations(observations):
    if len(observations) == 0:
        return "No observations registered yet."
    else:
        return "\n".join(observations)

def GetObservations():
    allObservations = {}
    with open("memory.json", "r") as file:
        allObservations = json.load(file)
    importantObsList = allObservations["important"]
    lessImportantObsList = allObservations["lessImportant"]
    importantObsStr = GetStringOfObservations(importantObsList)
    lessImportantObsStr = GetStringOfObservations(lessImportantObsList)
    return importantObsList, lessImportantObsList, importantObsStr, lessImportantObsStr

def GenerateSystemPrompt(importantObsStr, lessImportantObsStr):
    return f"You're charismatic and funny. You engage well with the user. You use your accumulated \
observations about the user to make your conversational approach more personalized. You have two sets of \
observations about the user: important and less important.\n\n\
Here are your important observations so far:\n\n{importantObsStr}\n\n\
Here are your less important observations so far:\n\n{lessImportantObsStr}"

if __name__ == "__main__":
    load_dotenv()
    retrieval = Retrieval()
    maxImportantObservations = 10
    maxLessImportantObservations = 15
    importantObsList, lessImportantObsList, importantObsStr, lessImportantObsStr = GetObservations()
    system_prompt = GenerateSystemPrompt(importantObsStr, lessImportantObsStr)
    history = [SystemMessage(system_prompt)]
    model = init_chat_model("gpt-4o-mini")
    modelWithStructure = model.with_structured_output(ModelOutput, include_raw=True)

    while True:
        prompt = input(">>>")
        if prompt == "/exit":
            retrieval.dbConnection.close()
            break
        history.append(HumanMessage(prompt))
        response = modelWithStructure.invoke(history)
        structuredResponse = response["parsed"]
        history.append(AIMessage(structuredResponse.answer))
        #print(response)
        print(structuredResponse)

        if structuredResponse.optionalShortUserObservation != None:
            obsList = importantObsList if len(importantObsList) < maxImportantObservations \
                else lessImportantObsList
            obsList.append(structuredResponse.optionalShortUserObservation)
            if len(importantObsList) < maxImportantObservations:
                importantObsStr = GetStringOfObservations(obsList)
            else:
                lessImportantObsStr = GetStringOfObservations(obsList)
            newSystemPrompt = GenerateSystemPrompt(importantObsStr, lessImportantObsStr)
            history[0] = SystemMessage(newSystemPrompt)
            print(history)
            with open("memory.json", "w") as file:
                allObservations = {    
                    "important": importantObsList,
                    "lessImportant": lessImportantObsList
                }
                json.dump(allObservations, file, indent=4)
