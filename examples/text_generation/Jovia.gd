extends Node


# Called when the node enters the scene tree for the first time.
func _ready() -> void:
	print_debug("fn binding test: ", Jovia.add(1, 3))
	#print_debug("text generation: <user>Hello how are you\n<text>", Jovia.text("Hello how are you?"))
	var sentences: Array[String] = ["I like dogs!", "I like dogs!", "I like cats!"]
	print_debug("embeddings ", Jovia.embed(sentences))
	#print_debug("cosine similarity ", Jovia.similarity("I like Rust!", "I like Rust!"))


# Called every frame. 'delta' is the elapsed time since the previous frame.
func _process(delta: float) -> void:
	pass
