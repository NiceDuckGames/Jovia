use candle_core::Tensor;
use godot::engine::INode;
use godot::engine::Node;
use godot::prelude::*;
use inference::embedding::EmbeddingModel;
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
    fn text(prompt: String) -> String {
        let mut pipeline =
            TextGeneration::new(None, None, 299792458, None, None, 1.1, 64, false).unwrap();
        pipeline.run(&prompt, 256).unwrap();

        "Not Implemented".to_owned()
    }

    #[func]
    fn embed(sentences: Array<GString>) -> Array<Array<Array<f32>>> {
        let sentences: Vec<String> = sentences.iter_shared().map(|s| s.to_string()).collect();

        // TODO: Implement a global model pool that can be passed to these methods
        let em = EmbeddingModel::new(true, false, None, None).unwrap();
        // TODO: Handle the error case here in place of .unwrap()
        let embeddings: Tensor = em.embed_batch(sentences).unwrap();
        godot_print!("embedding tensor dims {}", embeddings.dims().len());
        let vec3: Vec<Vec<Vec<f32>>> = embeddings.to_vec3().unwrap();
        let mut outer_arr: Array<Array<Array<f32>>> = Array::new();

        // Want to copy the innermost f32 data in vec3 into arr3
        // First approach we nest loops on iterators
        // Eventually we will want to consider cache locality friendly approaches
        for middle_vec in &vec3 {
            let mut middle_arr: Array<Array<f32>> = Array::new();
            for inner_vec in middle_vec {
                let mut val_array: Array<f32> = Array::new();
                for inner_val in inner_vec {
                    val_array.push(inner_val.to_godot());
                }
                middle_arr.push(val_array);
            }
            outer_arr.push(middle_arr)
        }

        godot_print!("{:?}", outer_arr);
        outer_arr
    }

    #[func]
    fn similarity(sentence1: String, sentence2: String) -> f64 {
        1.0
    }
}
