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
        let arr3: Array<Array<Array<f32>>> = vec3
            .into_iter()
            .map(|x| Array::<Vec<Vec<f32>>>::from(&x[..]))
            .collect();
        godot_print!("{:?}", arr3);
        arr3
    }

    #[func]
    fn similarity(sentence1: String, sentence2: String) -> f64 {
        1.0
    }
}
