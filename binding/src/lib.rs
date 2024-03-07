use candle_core::Tensor;
use godot::engine::IObject;
use godot::engine::Object;
use godot::obj::WithBaseField;
use godot::prelude::*;
use inference::embedding::EmbeddingModel;
use inference::text_generation::TextGeneration;
use std::sync::mpsc;
use std::sync::mpsc::Receiver;
use std::sync::mpsc::TryRecvError;
use std::thread;
use std::time::Duration;

#[gdextension]
unsafe impl ExtensionLibrary for Jovia {}

#[derive(GodotClass)]
#[class(base=Object)]
pub struct Jovia {
    base: Base<Object>,
}

#[godot_api]
impl IObject for Jovia {
    fn init(base: Base<Object>) -> Self {
        godot_print!("Jovia init called");
        // This function might be where we should handle loading the models?
        Self { base }
    }
}

#[godot_api]
impl Jovia {
    #[signal]
    pub fn token(token: GString);

    #[signal]
    pub fn finished();

    #[func]
    fn add(left: i32, right: i32) -> i32 {
        let sum = left + right;
        godot_print!("Hello from Rust! {} + {} = {}", left, right, left + right);
        sum
    }

    #[func]
    pub fn text(&mut self, prompt: String) {
        // This takes a string prompt and instantiates a InferenceChannel
        // which uses threading to perform streaming token inference in a non-blocking manner.
        // each time a new token is generated a godot signal on_token will be emitted.
        // This will allow user to grab new tokens in a non-blocking even driven manner.

        let mut pipeline =
            TextGeneration::new(None, None, 299792458, None, None, 1.1, 64, false).unwrap();
        let (tx, rx) = mpsc::channel::<String>();

        // Kick off text generation in a seaparate thread
        let handle: thread::JoinHandle<()> = thread::spawn(move || {
            pipeline.run(&prompt, 256, tx).unwrap();
        });

        // this will block, we will eventually want to make it non-blocking
        loop {
            match rx.try_recv() {
                Ok(token) => {
                    godot_print!("Token {}", token);
                    self.base_mut()
                        .emit_signal("token".into(), &[token.to_string().to_variant()]);
                }
                Err(TryRecvError::Empty) => {
                    //godot_print!("No tokens available");
                }
                Err(TryRecvError::Disconnected) => {
                    //godot_print!("All tokens consumed");
                    self.base_mut().emit_signal("finished".into(), &[]);
                    break;
                }
            }
            thread::sleep(Duration::from_millis(100));
        }

        // Wait for the generation thread to finish
        // Should consider bubbling the join handle up to the binding code
        handle.join().unwrap();
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
