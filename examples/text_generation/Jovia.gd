extends Node

var gen
var recv
var tokens: Array[String] = []
var generating = false
var finished = false
var loaded = false
var thread: Thread

# Called when the node enters the scene tree for the first time.
func _ready() -> void:
	thread = Thread.new()
	gen = TokenGenerator.new()
	gen.model_loaded.connect(_on_model_loaded)
	gen.finished.connect(_on_finished)

func prompt_model(prompt: String) -> void:
	# Load the model in a thread to avoid blocking unless you want to block
	thread.start(inf.load_model.bind())
	
func _on_model_loaded(recv: TextReceiver) -> void:
	self.recv = recv
	self.recv.token.connect(_on_token)
	self.recv.finished.connect(_on_finished)
	
func _on_token(token: String) -> void:
	tokens.append(token)
	print_debug("Got token: ", token)

func _on_finished() -> void:
	print_debug("Text generation finished!")
	finished = true
	print_debug("".join(tokens))
	# unload the generator here if you want or keep it in memory for more generations using prompt model
	# gen.unload()

# Called every frame. 'delta' is the elapsed time since the previous frame.
func _process(delta: float) -> void:
	if loaded:
		# poll for a token
		recv.poll()

func _exit_tree() -> void:
	thread.wait_to_finish()
