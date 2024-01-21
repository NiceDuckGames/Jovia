from .prompts import *
import numpy as np
from sentence_transformers import SentenceTransformer

DISALLOWED_THRESHOLD = 0.7
VAUGUE_THRESHOLD = 0.7
PROMPT_INJECTION_THRESHOLD = 0.7

model = SentenceTransformer('all-MiniLM-L6-v2')
disallowed_prompts_embedded = np.stack(model.encode(disallowed_prompts), axis=0)
vague_prompts_embedded = np.stack(model.encode(vague_prompts), axis=0)
prompt_injections_embedded = np.stack(model.encode(prompt_injections), axis=0)

def is_disallowed(sentence):
    embedded = model.encode([sentence])
    return (embedded @ disallowed_prompts_embedded.T).max().item() > DISALLOWED_THRESHOLD 

def is_vague(sentence):
    embedded = model.encode([sentence])
    return (embedded @ vague_prompts_embedded.T).max().item > VAUGUE_THRESHOLD 

def is_prompt_injection(sentence):
    embedded = model.encode([sentence])
    return (embedded @ prompt_injections_embedded.T).max().item > PROMPT_INJECTION_THRESHOLD 