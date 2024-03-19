@tool
extends MarginContainer

@export var prompt_edit: TextEdit
@export var response_label: RichTextLabel

var last_prompt: String


func _ready() -> void:
	
	JoviaTest.connect("token", self._on_jovia_token)


func _on_send_button_pressed() -> void:
	
	response_label.text = ""
	
	var prompt: String = prompt_edit.text.strip_escapes()
	
	response_label.text = prompt
	
	JoviaTest.prompt(prompt)


func _on_jovia_token(token: String):
	
	response_label.text = response_label.text + " " + token
