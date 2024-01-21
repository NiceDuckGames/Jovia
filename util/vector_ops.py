def retreive_command_sequences(collection, prompts):
    return collection.query(
        query_texts=prompts
    )
