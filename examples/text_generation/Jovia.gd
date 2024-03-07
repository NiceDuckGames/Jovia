extends Node

var inf
var tokens: Array[String] = []
var generating = false
var finished = false
var thread: Thread

# Called when the node enters the scene tree for the first time.
func _ready() -> void:
	inf = Jovia.new()
	inf.token.connect(_on_token)
	inf.finished.connect(_on_finished)
	
func _on_token(token: String) -> void:
	print_debug("Got token: ", token)

func _on_finished() -> void:
	print_debug("Text generation finished!")
	finished = true

# Called every frame. 'delta' is the elapsed time since the previous frame.
func _process(delta: float) -> void:
	if not generating:
		generating = true
		inf.text("What is Godot Engine?")
	
	if generating and not finished:
		print_debug("Got tokens: ", tokens)
		print_debug("".join(tokens))
