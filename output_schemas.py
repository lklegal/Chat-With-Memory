from pydantic import BaseModel, Field

class MainModelOutput(BaseModel):
    """
    This is the format of your responses to queries.
    """
    answer: str = Field(description="This is your response to the user. It's what they see.")
    reasoningAboutShortUserObservation: str = Field(description=("This is where you write your reasoning "
    "as to wheather or not to write a new short oservation about the user. You only write relevant things "
    "you should remember about the user. That includes things like stable, long term characteristics of the user"
    "(long term preferences, goals, values, consistent habits, professional focus, etc. - something likely to remain"
    "true across many conversations). Relevance also includes observations likely to improve the assistant’s ability "
    "to provide better answers in future conversations. This includes preferences about response length, tone, "
    "structure, etc - something likely to be useful in the future. So, this space is where you decide if something is "
    "relevant enough to be registered. It's fine and even desirable to not write anything. Only decide to write "
    "when relevant. Don't write things you already know."))
    optionalShortUserObservation: str | None = Field(description=("This is where you write "
    "something short about the user you want to remember, provided you concluded in your reasoning "
    "that the observation you're about to write is relevant. It's fine to not write anything. Desirable, even. "
    "Only write if you decided it's something relevant. If nothing is written, the type of this field must be None. "
    "Examples of observations: 'User has a cat', or 'User is afraid of heights', or 'User likes coffee'. Things like that."
    "Try to generalize your observations to increase the likelihood of them being relevant. For instance, if "
    "the user says they're working on a python project, a less relevant observation would be something like 'User is working "
    "on a python project', while more relevant observations would be things like 'User is a programmer', or 'User knows python'."))

class DeduplicationModelOutput(BaseModel):
     reasoningAboutPossibleDuplication: str = Field(description=("This is where you write "
     "your reasoning as to wheather or not the new observation given is similar to any one of the "
     "others provided. Remember, a duplicate observation can be near identical, yes, but also a paraphrase, or something "
     "that was already alluded to before (even if fragmented in multiple observations), or "
     "something that just has the same general meaning as previous observations, even if phrased "
     "very differently."))
     newObservationIsDuplicate: bool = Field(description=("This should be True if you concluded the "
     "new observation is a duplication (near identical, paraphrase or same general meaning) of any of "
     "the other observations provided, and False otherwise."))