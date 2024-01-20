import prompts
from sentence_transformers import SentenceTransformer

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

model = SentenceTransformer('all-MiniLM-L6-v2')
disallowed_prompts_embedded = np.stack(model.encode(prompts.disallowed_prompts), axis=0)
vague_prompts_embedded = np.stack(model.encode(prompts.vague_prompts), axis=0)

def is_disallowed(sentence):
    embedded = model.encode([sentence])
    return (embedded @ disallowed_prompts_embedded.T).max().item() > 0.7

def is_vague(sentence):
    embedded = model.encode([sentence])
    return (embedded @ vague_prompts_embedded.T).max().item > 0.7