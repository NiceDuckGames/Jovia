from langchain.embeddings.openai import OpenAIEmbeddings
import chromadb
import json

data = []

# The assistants memory
storage = {}
def assign(key, value): 
    # store a value
    storage[key] = value; return f'{{{key}: "{value}"}}'
def get(key): 
    # retrieve a value
    return storage.get(key)

# load in an example command list
with open("../util/command_list_data.jsonl") as f:
    for line in f:
        data_line = json.loads(line)
        data.append(data_line)

# map the embedding of the command's prompts to the actual command
command_prompts = []
command_sequences = []
for data_line in data:
    command_prompts.append(data_line['prompt'])
    command_sequences.append(data_line['response'])

# Store embeddings and their documents in Chroma
client = chromadb.EphemeralClient()
collection = client.get_or_create_collection("commands")
collection.add(
    documents=command_prompts, # embed the prompts
    metadatas=[{'command_sequence': json.dumps(c)} for c in command_sequences], # attach the command seq to the prompt
    ids=[str(i) for i in range(len(command_prompts))] # generate ids for each stored document
)

def retreive_command_sequences(prompts):
    return collection.query(
        query_texts=prompts
    )
