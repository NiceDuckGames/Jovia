@tool
extends Node

var gen: TextGenerator
var recv: TextReceiver

var threads_joined = false
var finished = false
var loaded = false
var prompted = false

var loader_thread: Thread

var tokens: Array[String] = []


signal token(tok: String)


func _ready() -> void:
	
	gen = TextGenerator.new()
	gen.loaded.connect(_on_model_loaded)
	
	loader_thread = Thread.new()
	loader_thread.start(gen.load_model.bind())


func _on_model_loaded() -> void:
	loaded = true


func _on_token(token: String) -> void:
	tokens.append(token)
	
	emit_signal("token", token)
	print(token)


func _on_finished() -> void:
	print("finish")
	finished = true


func _on_disconnected() -> void:
	finished = true


func prompt(prompt: String) -> void:
	
	if !loaded: return
	if recv: return
	
	recv = gen.prompt(prompt)
	recv.token.connect(_on_token)
	recv.finished.connect(_on_finished)
	recv.disconnected.connect(_on_disconnected)


func _process(delta: float) -> void:
	
	if loaded:
		if recv:
			recv.poll()
	
	if finished && !loader_thread.is_alive() && !threads_joined:
		
		loader_thread.wait_to_finish()
		
		recv.free()
		recv = null
		
		threads_joined = true


func _exit_tree() -> void:
	loader_thread.wait_to_finish()
