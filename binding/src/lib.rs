use candle_core::Tensor;
use godot::engine::IObject;
use godot::engine::Object;
use godot::obj::WithBaseField;
use godot::prelude::*;
use inference::embedding::EmbeddingModel;
use inference::text_generation::TextGeneration;
use std::rc::Rc;
use std::sync::mpsc;
use std::sync::mpsc::Receiver;
use std::sync::mpsc::TryRecvError;
use std::sync::Arc;
use std::sync::Mutex;
use std::thread;
use std::time::Duration;

#[gdextension]
unsafe impl ExtensionLibrary for Jovia {}

#[derive(GodotClass)]
#[class(base=Object)]
pub struct TextGenerator {
    base: Base<Object>,
    pipeline: Rc<Option<TextGeneration>>,
}

#[godot_api]
impl IObject for TextGenerator {
    fn init(base: Base<Object>) -> Self {
        Self {
            base,
            pipeline: Rc::new(None),
        }
    }
}

#[godot_api]
impl TextGenerator {
    #[signal]
    pub fn loaded(pipeline: Gd<TextReceiver>);

    #[func]
    pub fn load_model(&mut self) {
        let pipeline =
            TextGeneration::new(None, None, 299792458, None, None, 1.1, 64, false).unwrap();
        self.pipeline = Rc::new(Some(pipeline));
        self.base_mut().emit_signal("model_loaded".into(), &[]);
    }

    #[func]
    pub fn prompt(&mut self, prompt: String) -> Gd<TextReceiver> {
        // This takes a string promp= t and instantiates a InferenceChannel
        // which uses threading to perform streaming token inference in a non-blocking manner.
        // each time a new token is generated a godot signal on_token will be emitted.
        // This will allow user to grab new tokens in a non-blocking even driven manner.

        let (tx, rx) = mpsc::channel::<String>();
        let text_receiver = Gd::from_init_fn(move |base| TextReceiver { base, rx: Some(rx) });

        let mut pipeline_ref = self.pipeline.as_ref().clone().unwrap();
        let _ = pipeline_ref.run(&prompt, 255, tx);

        // This is the object that is used in GDScript to receive tokens
        // from the generator.
        text_receiver
    }
}

#[derive(GodotClass)]
#[class(base=Object)]
pub struct TextReceiver {
    base: Base<Object>,
    rx: Option<Receiver<String>>,
}

#[godot_api]
impl IObject for TextReceiver {
    fn init(base: Base<Object>) -> Self {
        Self { base, rx: None }
    }
}

#[godot_api]
impl TextReceiver {
    #[signal]
    pub fn token(token: GString);

    #[signal]
    pub fn finished();

    #[func]
    pub fn poll(&mut self) {
        // Receive doesn't return a token itself as that would block threaded use.
        // Instead it returns a token via a Godot signal named "token".
        let rx = self.rx.as_ref().unwrap();
        match rx.try_recv() {
            Ok(token) => {
                self.base_mut()
                    .emit_signal("token".into(), &[token.to_variant()]);
            }
            Err(TryRecvError::Empty) => {
                godot_print!("No token");
            }
            Err(TryRecvError::Disconnected) => {
                godot_print!("Disconnected");
            }
        };
    }
}

#[derive(GodotClass)]
#[class(base=Object)]
pub struct Jovia {
    base: Base<Object>,
}

#[godot_api]
impl IObject for Jovia {
    fn init(base: Base<Object>) -> Self {
        godot_print!("Jovia init called");
        Self { base }
    }
}

#[godot_api]
impl Jovia {
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
