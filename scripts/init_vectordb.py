# Initialize a local test vector store with command sequence data

import chromadb
import json

data = []

# load in an example command list
with open("./command_list_data.jsonl") as f:
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
client = chromadb.PersistentClient(path="../commands-index/")
collection = client.get_or_create_collection("commands")
collection.add(
    documents=command_prompts, # embed the prompts
    metadatas=[{'command_sequence': json.dumps(c)} for c in command_sequences], # attach the command seq to the prompt
    ids=[str(i) for i in range(len(command_prompts))] # generate ids for each stored document
)