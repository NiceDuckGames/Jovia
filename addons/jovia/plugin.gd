@tool
extends EditorPlugin

var jovia_chat_panel_res_path: PackedScene = preload("res://addons/jovia/jovia_chat.tscn")
var jovia_chat: MarginContainer

func _enter_tree() -> void:
	
	jovia_chat = jovia_chat_panel_res_path.instantiate()
	
	add_control_to_bottom_panel(jovia_chat, "Jovia")


func _exit_tree() -> void:
	
	remove_control_from_bottom_panel(jovia_chat)
