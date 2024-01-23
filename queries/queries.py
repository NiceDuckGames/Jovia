import lmql
from util.prompts import *

@lmql.query
async def task_decomposition(system_prompt: str, prompt: str):
    '''lmql
    """
    {system_prompt}
    {:user}
    Here is the following task {prompt}. Decompose it into a JSON list of subtasks like follows:

    [["subtask1","subtask2","subtaskN"]]

    Do not include subtasks that would require the user to use tools that are outside of Godot 4.
    ONLY generate subtasks which are actions that can be carried out within the Godot 4 game engine.
    Do not include newlines or any sort of numbering or list items that prefix each subtask.

    [JSON_LIST_OF_SUBTASKS]
    """
    return JSON_LIST_OF_SUBTASKS.strip()
    '''

@lmql.query
async def task_summarization(system_prompt: str, prompt: str):
    '''lmql
    """
    {system_prompt}
    {:assistant}
    A high level summary of the following list of subtasks {prompt} [HIGH_LEVEL_SUMMARY]
    """
    return HIGH_LEVEL_SUMMARY.strip() 
    '''

@lmql.query
async def internal_reasoning(system_prompt: str):
    '''lmql
    """
    {system_prompt}
    {:assistant} Internal Reasoning: [REASONING]
    """ \
        where STOPS_AT(REASONING, "\n") and STOPS_BEFORE(REASONING, "External Answer:")
    return REASONING.strip()
    '''

@lmql.query
async def chat_interaction(system_prompt: str, prompt: str):
    '''lmql
    """
    {system_prompt}
    {:user} {prompt}
    {:assistant} [RESPONSE]
    """ 
    return RESPONSE.strip()
    '''