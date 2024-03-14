extends Node

var gen: TextGenerator
var recv: TextReceiver
var tokens: Array[String] = []
var generating = false
var finished = false
var loaded = false
var prompted = false
var thread: Thread
var loader_thread: Thread

var model_id = "karpathy/tinyllamas"
var which_model = "stories15m.bin"
var tokenizer_id = "hf-internal-testing/llama-tokenizer"

# Called when the node enters the scene tree for the first time.
func _ready() -> void:
	thread = Thread.new()
	gen = TextGenerator.new()
	gen.loaded.connect(_on_model_loaded)
	loader_thread = Thread.new()
	loader_thread.start(gen.load_model.bind(model_id, which_model, tokenizer_id))
	
func _on_model_loaded() -> void:
	print_debug("Model loaded")
	loaded = true

func _on_token(token: String) -> void:
	tokens.append(token)
	print(token)

func _on_finished() -> void:
	print_debug("Text generation finished!")
	finished = true
	print_debug("".join(tokens))
	# unload the generator here if you want or keep it in memory for more generations using prompt model
	# gen.unload()

func _on_disconnected() -> void:
	print_debug("Receiver disconnected!")
	get_tree().quit()

# Called every frame. 'delta' is the elapsed time since the previous frame.
func _process(delta: float) -> void:
	if loaded:
		if not prompted:
			var prompt = "Once upon a time"
			print_debug("Prompting model: ", prompt)
			recv = gen.prompt(prompt)
			recv.token.connect(_on_token)
			recv.finished.connect(_on_finished)
			recv.disconnected.connect(_on_disconnected)
			print_debug(recv)
			prompted = true
		# poll for a token
		if recv:
			recv.poll() # tokens can be received via a signal

func _exit_tree() -> void:
	loader_thread.wait_to_finish()
	thread.wait_to_finish()
