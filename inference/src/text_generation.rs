use anyhow::{Error as E, Result};
use candle_core::utils::cuda_is_available;
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::llama as model;
use hf_hub::{Repo, RepoType};
use model::{Cache, Config, Llama, LlamaConfig};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex};
use tokenizers::Tokenizer;

#[derive(Clone)]
pub struct TextGeneration {
    pub model: Arc<Mutex<Llama>>,
    pub model_id: String,
    pub device: Device,
    pub tokenizer: Arc<Mutex<Tokenizer>>,
    pub logits_processor: Arc<Mutex<LogitsProcessor>>,
    pub tokens: Vec<String>,
    pub cache: Cache,
    pub config: Config,
}

impl TextGeneration {
    pub fn new(
        model_id: String,
        which_model: String,
        revision: Option<String>,
        dtype: Option<String>,
        temp: Option<f64>,
        top_p: Option<f64>,
    ) -> Result<Self, E> {
        let device = Device::Cpu;
        let dtype = match dtype.as_deref() {
            Some("f16") => DType::F16,
            Some("bf16") => DType::BF16,
            Some("f32") => DType::F32,
            Some(dtype) => anyhow::bail!("Unsupported dtype {dtype}"),
            None => DType::F16,
        };

        let (llama, tokenizer_filename, mut cache, config) = {
            let api = hf_hub::api::sync::Api::new()?;
            let revision = revision.unwrap_or("main".to_string());
            let api = api.repo(Repo::with_revision(
                model_id.clone(),
                RepoType::Model,
                revision,
            ));
            let tokenizer_filename = api.get("tokenizer.json")?;
            let config_filename = api.get("config.json")?;
            let config: LlamaConfig = serde_json::from_slice(&std::fs::read(config_filename)?)?;
            let config = config.into_config(false);
            let filenames = vec![api.get("model.safetensors")?];
            let cache = model::Cache::new(true, dtype, &config, &device)?;
            let vb = unsafe { VarBuilder::from_mmaped_safetensors(&filenames, dtype, &device)? };

            (Llama::load(vb, &config)?, tokenizer_filename, cache, config)
        };

        let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;
        let logits_processor = LogitsProcessor::new(299792458, temp, top_p);

        Ok(Self {
            model: Arc::new(Mutex::new(llama)),
            model_id,
            device,
            tokenizer: Arc::new(Mutex::new(tokenizer)),
            tokens: Vec::new(),
            logits_processor: Arc::new(Mutex::new(logits_processor)),
            cache,
            config,
        })
    }

    pub fn tokenize(&self, input: String) -> Result<Vec<u32>, anyhow::Error> {
        Ok(self
            .tokenizer
            .clone()
            .lock()
            .unwrap()
            .encode(input, true)
            .map_err(E::msg)?
            .get_ids()
            .to_vec())
    }

    pub fn encode(&self, tokens: &[u32]) -> String {
        // TODO: Figure out how to avoid cloning the tokenizer
        let tokenizer = self.tokenizer.clone().lock().unwrap().clone();
        let mut tokenizer = TokenOutputStream::new(tokenizer);

        // TODO: look into cleaning up the double unwrap and handling the errors.
        // probably should bubble up the option type and handle the Result here.
        tokens
            .iter()
            .map(|&t| tokenizer.next_token(t).unwrap().unwrap())
            .collect::<Vec<String>>()
            .join("")
    }

    /// Generates the next token given a prompt string which serves as the on going context.
    /// This is intended to be used within a loop allowing the controller to manage the progress of
    /// inference against the initial prompt. Because of this the prompt is intended to be passed
    /// in as it accumulates previously generated tokens.
    pub fn next_token(
        &mut self,
        tokens: Vec<u32>,
        repeat_penalty: f32,
        repeat_last_n: usize,
        context_size: usize,
        context_index: usize,
        index_pos: usize,
    ) -> Result<(u32, usize), anyhow::Error> {
        // =====
        // dirty unwrap of our arc mutexd tokeinzer in self and a clone of the inner struct
        // So when we prompt we are duplicating the tokenizer, but this removes a disk IO bound on
        // prompting.
        // =====
        //let tokenizer = self.tokenizer.clone().lock().unwrap().clone();

        let ctxt = &tokens[tokens.len().saturating_sub(context_size)..];
        let input = Tensor::new(ctxt, &self.device)?.unsqueeze(0)?;
        let logits =
            self.model
                .clone()
                .lock()
                .unwrap()
                .forward(&input, context_index, &mut self.cache)?;
        let logits = logits.squeeze(0)?;
        let logits = if repeat_penalty == 1. {
            logits
        } else {
            let start_at = tokens.len().saturating_sub(repeat_last_n);
            candle_transformers::utils::apply_repeat_penalty(
                &logits,
                repeat_penalty,
                &tokens[start_at..],
            )?
        };

        let next_token = self
            .logits_processor
            .clone()
            .lock()
            .unwrap()
            .sample(&logits)?;
        // Return the next token and the updated index position for iteration control
        // by the caller of this function.
        Ok((next_token, index_pos + ctxt.len()))
    }

    /*pub fn run(
        &mut self,
        prompt: &str,
        sample_len: usize,
        repeat_last_n: usize,
        repeat_penalty: f32,
        tx: mpsc::Sender<String>,
    ) -> Result<f64, anyhow::Error> {
    }*/
}

/// This is a wrapper around a tokenizer to ensure that tokens can be returned to the user in a
/// streaming way rather than having to wait for the full decoding.
pub struct TokenOutputStream {
    tokenizer: tokenizers::Tokenizer,
    tokens: Vec<u32>,
    prev_index: usize,
    current_index: usize,
}

// https://github.com/huggingface/candle/blob/main/candle-examples/src/token_output_stream.rs
impl TokenOutputStream {
    pub fn new(tokenizer: tokenizers::Tokenizer) -> Self {
        Self {
            tokenizer,
            tokens: Vec::new(),
            prev_index: 0,
            current_index: 0,
        }
    }

    pub fn into_inner(self) -> tokenizers::Tokenizer {
        self.tokenizer
    }

    fn decode(&self, tokens: &[u32]) -> Result<String> {
        match self.tokenizer.decode(tokens, true) {
            Ok(str) => Ok(str),
            Err(err) => anyhow::bail!("cannot decode: {err}"),
        }
    }

    // https://github.com/huggingface/text-generation-inference/blob/5ba53d44a18983a4de32d122f4cb46f4a17d9ef6/server/text_generation_server/models/model.py#L68
    pub fn next_token(&mut self, token: u32) -> Result<Option<String>> {
        let prev_text = if self.tokens.is_empty() {
            String::new()
        } else {
            let tokens = &self.tokens[self.prev_index..self.current_index];
            self.decode(tokens)?
        };
        self.tokens.push(token);
        let text = self.decode(&self.tokens[self.prev_index..])?;
        if text.len() > prev_text.len() && text.chars().last().unwrap().is_alphanumeric() {
            let text = text.split_at(prev_text.len());
            self.prev_index = self.current_index;
            self.current_index = self.tokens.len();
            Ok(Some(text.1.to_string()))
        } else {
            Ok(None)
        }
    }

    pub fn decode_rest(&self) -> Result<Option<String>> {
        let prev_text = if self.tokens.is_empty() {
            String::new()
        } else {
            let tokens = &self.tokens[self.prev_index..self.current_index];
            self.decode(tokens)?
        };
        let text = self.decode(&self.tokens[self.prev_index..])?;
        if text.len() > prev_text.len() {
            let text = text.split_at(prev_text.len());
            Ok(Some(text.1.to_string()))
        } else {
            Ok(None)
        }
    }

    pub fn decode_all(&self) -> Result<String> {
        self.decode(&self.tokens)
    }

    pub fn get_token(&self, token_s: &str) -> Option<u32> {
        self.tokenizer.get_vocab(true).get(token_s).copied()
    }

    pub fn tokenizer(&self) -> &tokenizers::Tokenizer {
        &self.tokenizer
    }

    pub fn clear(&mut self) {
        self.tokens.clear();
        self.prev_index = 0;
        self.current_index = 0;
    }
}

pub fn hub_load_safetensors(
    repo: &hf_hub::api::sync::ApiRepo,
    json_file: &str,
) -> Result<Vec<std::path::PathBuf>, E> {
    // https://github.com/huggingface/candle/blob/5cdd84e0f6365df832a9dbb062ad3a9a34bb65b3/candle-examples/src/lib.rs#L122
    let json_file = repo.get(json_file).map_err(candle_core::Error::wrap)?;
    let json_file = std::fs::File::open(json_file)?;
    let json: serde_json::Value = serde_json::from_reader(&json_file).map_err(E::msg)?;

    // Not sure why this works :think:
    let weight_map = match json.get("weight_map") {
        None => anyhow::bail!("no weight map in {json_file:?}"),
        Some(serde_json::Value::Object(map)) => map,
        Some(_) => anyhow::bail!("weight map in {json_file:?} is not a map"),
    };
    let mut safetensors_files = std::collections::HashSet::new();
    for value in weight_map.values() {
        if let Some(file) = value.as_str() {
            safetensors_files.insert(file.to_string());
        }
    }
    let safetensors_files = safetensors_files
        .iter()
        .map(|v| repo.get(v).map_err(E::msg))
        .collect::<Result<Vec<_>>>()?;
    Ok(safetensors_files)
}
