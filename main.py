from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, SystemMessage, AIMessage#, AIMessageChunk
#from langchain.tools import tool
from dotenv import load_dotenv
from pydantic import BaseModel, Field
import json

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

if __name__ == "__main__":
    load_dotenv()

    observations = []
    with open("memory.json", "r") as file:
        observations = json.load(file)
    listedObservations = ""
    if len(observations) == 0:
        listedObservations = "No observations registered yet."
    else:
        for observation in observations:
            listedObservations += observation + "\n"
    system_prompt = "You're charismatic and funny. You engage well with the user. You use your accumulated\
    observations about the user to make your conversational approach more personalized.\n\n\
    Here are your observations about the user so far:\n\n" + listedObservations

    history = [SystemMessage(system_prompt)]
    model = init_chat_model("gpt-4o-mini")
    modelWithStructure = model.with_structured_output(ModelOutput)

    while True:
        prompt = input(">>>")
        if prompt == "/exit":
            break
        history.append(HumanMessage(prompt))
        response = modelWithStructure.invoke(history)
        history.append(AIMessage(response.answer))
        print(response)

        if response.optionalShortUserObservation != None:
            observations.append(response.optionalShortUserObservation)
            with open("memory.json", "w") as file:
                json.dump(observations, file, indent=4)
