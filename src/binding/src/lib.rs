use anyhow::{Error as E, Result};
use candle_core::Tensor;
use godot::engine::IObject;
use godot::engine::Object;
use godot::obj::WithBaseField;
use godot::prelude::*;
use inference::embedding::EmbeddingModel;
use inference::text_generation::TextGeneration;
use std::borrow::BorrowMut;
use std::cell::RefCell;
use std::sync::mpsc::Receiver;

#[gdextension]
unsafe impl ExtensionLibrary for Jovia {}

#[derive(GodotClass)]
#[class(base=Object)]
pub struct TextGenerator {
    base: Base<Object>,
    pipeline: RefCell<Option<TextGeneration>>,
    tokens: Vec<String>,
    rx: Option<Receiver<String>>,
}

#[godot_api]
impl IObject for TextGenerator {
    fn init(base: Base<Object>) -> Self {
        Self {
            base,
            pipeline: RefCell::new(None),
            rx: None,
            tokens: Vec::new(),
        }
    }
}

#[godot_api]
impl TextGenerator {
    #[signal]
    pub fn loaded();

    #[signal]
    pub fn token(token: String);

    #[signal]
    pub fn finished();

    #[func]
    pub fn load_model(&mut self, model_id: String, which_model: String) {
        let pipeline = TextGeneration::new(model_id, which_model, None, None, None, None).unwrap();
        self.pipeline = RefCell::new(Some(pipeline));
        self.base_mut().emit_signal("loaded".into(), &[]);
    }

    #[func]
    /// Starts text generation against the loaded model.
    /// The generated tokens are accumulated in the tokens vec of the this struct.
    /// This function is blocking as it runs a loop against the underlying model.
    /// It can however be called relatively safely from a multi-threaded context.
    ///
    /// This function will emit the "token" signal each time a new token is generated.
    /// The "token" signal does not consume the token.
    ///
    /// This function will also emit the "finished" signal when generation has finished.
    pub fn prompt(
        &mut self,
        system: String,
        prompt: String,
        sample_len: i32,
        repeat_penalty: f32,
        repeat_last_n: u64,
    ) {
        let eos_token_id = self
            .pipeline
            .borrow_mut()
            .as_mut()
            .unwrap()
            .tokenizer
            .token_to_id("</s>")
            .unwrap();
        let prompt_template =
            "<|system|>{system}</s>\n<|user|>{prompt}</s>\n<|assistant|>".to_string();

        let mut tokens = self
            .pipeline
            .borrow_mut()
            .as_mut()
            .unwrap()
            .tokenizer
            .encode(prompt.clone(), true)
            .map_err(E::msg)
            .unwrap()
            .get_ids()
            .to_vec();

        println!("Starting the inference loop");
        println!("{prompt:?}");
        // Inference loop
        let mut index_pos = 0;
        for i in 0..sample_len {
            let (context_size, context_index) = if self
                .pipeline
                .borrow_mut()
                .as_ref()
                .unwrap()
                .cache
                .use_kv_cache
                && i > 0
            {
                (1, index_pos)
            } else {
                (tokens.len(), 0)
            };

            let (token, ctxt_len) = {
                self.pipeline
                    .borrow_mut()
                    .as_mut()
                    .unwrap()
                    .next_token(
                        &tokens,
                        repeat_penalty,
                        repeat_last_n.try_into().unwrap(),
                        context_size,
                        context_index,
                    )
                    .unwrap()
            };

            index_pos += ctxt_len;

            // I think I need to simplify my types...
            let decoded_token = {
                self.pipeline
                    .borrow_mut()
                    .as_mut()
                    .unwrap()
                    .decode(&[token.clone()])
            };
            self.base_mut()
                .emit_signal("token".into(), &[decoded_token.to_variant()]);
            tokens.push(token);
            if token == eos_token_id {
                // If we get an eos token we stop generating
                break;
            }
        }
        self.base_mut().emit_signal("finished".into(), &[]);
    }

    #[func]
    /// This function empties the internal vector of tokens and returns a Godot Array of the
    /// tokens.
    pub fn consume_tokens(&mut self) -> godot::builtin::Array<GString> {
        let mut tokens_iter = self.tokens.clone();
        let tokens_drain = tokens_iter
            .drain(..)
            .into_iter()
            .map(|t: String| t.to_godot());
        let tokens_array: godot::builtin::Array<GString> =
            Array::from_iter(tokens_drain.into_iter());

        tokens_array
    }
}

pub struct Jovia {}

impl Jovia {
    fn embed(sentences: Array<GString>) -> Array<Array<Array<f32>>> {
        let sentences: Vec<String> = sentences.iter_shared().map(|s| s.to_string()).collect();

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

    fn similarity(sentence1: String, sentence2: String) -> f64 {
        1.0
    }
}
