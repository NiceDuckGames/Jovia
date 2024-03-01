use godot::engine::INode;
use godot::engine::Node;
use godot::prelude::*;
use inference::text_generation::TextGeneration;

struct GDExtTest;

#[gdextension]
unsafe impl ExtensionLibrary for Jovia {}

#[derive(GodotClass)]
#[class(base=Node)]
struct Jovia {
    base: Base<Node>,
}

#[godot_api]
impl INode for Jovia {
    fn init(base: Base<Node>) -> Self {
        // This function might be where we should handle loading the models?
        Self { base }
    }
}

#[godot_api]
impl Jovia {
    #[func]
    fn add(left: i32, right: i32) -> i32 {
        let sum = left + right;
        godot_print!("Hello from Rust! {} + {} = {}", left, right, left + right);
        sum
    }

    #[func]
    fn generate(prompt: String) -> String {
        let mut pipeline =
            TextGeneration::new(None, None, 299792458, None, None, 1.1, 64, false).unwrap();
        pipeline.run(&prompt, 256).unwrap();

        "Not Implemented".to_owned()
    }
}
