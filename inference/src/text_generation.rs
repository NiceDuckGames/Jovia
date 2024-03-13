use anyhow::{Error as E, Result};
use candle_core::utils::cuda_is_available;
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::llama2_c as model;
use candle_transformers::models::llama2_c_weights as weights;
use candle_transformers::models::quantized_llama2_c as qmodel;
use hf_hub::{Repo, RepoType};
use model::{Cache, Config, Llama};
use qmodel::QLlama;
use std::sync::mpsc::{self, Receiver};
use std::sync::{Arc, Mutex};
use tokenizers::Tokenizer;
use weights::TransformerWeights;

enum Model {
    Llama(Llama),
    QLlama(QLlama),
}

impl Model {
    fn forward(&self, xs: &Tensor, pos: usize, cache: &mut Cache) -> anyhow::Result<Tensor> {
        match self {
            Self::Llama(l) => Ok(l.forward(xs, pos, cache)?),
            Self::QLlama(l) => Ok(l.forward(xs, pos, cache)?),
        }
    }
}

#[derive(Clone)]
pub struct TextGeneration {
    model: Arc<Mutex<Model>>,
    model_id: String,
    device: Device,
    tokenizer: Arc<Mutex<Tokenizer>>,
    logits_processor: Arc<Mutex<LogitsProcessor>>,
    cache: Cache,
    config: Config,
}

impl TextGeneration {
    pub fn new(
        model_id: String,
        which_model: String,
        temp: Option<f64>,
        top_p: Option<f64>,
    ) -> Result<Self, E> {
        let device = Device::Cpu;

        // Get the tokenizer instantiated
        let tokenizer_id = "hf-internal-testing/llama-tokenizer".to_string();
        let api = hf_hub::api::sync::Api::new()?;
        let api = api.model(tokenizer_id.clone());
        let tokenizer_path = api.get("tokenizer.json")?;
        let tokenizer = match Tokenizer::from_file(tokenizer_path) {
            Ok(f) => f,
            Err(err) => panic!("Error loading tokenizer file: {:?}", err),
        };
        let tokenizer = Arc::new(Mutex::new(tokenizer));

        // Get the model loaded up
        let api = hf_hub::api::sync::Api::new()?;
        let api = api.model(model_id.clone());

        let config_path = api.get(&which_model)?;

        let is_gguf = config_path.extension().map_or(false, |v| v == "gguf");
        println!("Model is gguf: {:?}", is_gguf);

        let (model, config, mut cache) = if is_gguf {
            // Only support gguf for now
            let vb = qmodel::VarBuilder::from_gguf(config_path, &device)?;
            let (_vocab_size, dim) = vb
                .get_no_shape("model.embed_tokens.weight")?
                .shape()
                .dims2()?;
            let config = match dim {
                64 => Config::tiny_260k(),
                288 => Config::tiny_15m(),
                512 => Config::tiny_42m(),
                768 => Config::tiny_110m(),
                _ => anyhow::bail!("no config for dim {dim}"),
            };
            let freq_cis_real = vb
                .get(
                    (config.seq_len, config.head_size() / 2),
                    "rot.freq_cis_real",
                )?
                .dequantize(&device)?;
            let freq_cis_imag = vb
                .get(
                    (config.seq_len, config.head_size() / 2),
                    "rot.freq_cis_imag",
                )?
                .dequantize(&device)?;

            let fake_vb = candle_nn::VarBuilder::from_tensors(
                [
                    ("freq_cis_real".to_string(), freq_cis_real),
                    ("freq_cis_imag".to_string(), freq_cis_imag),
                ]
                .into_iter()
                .collect(),
                candle_core::DType::F32,
                &device,
            );
            let cache = model::Cache::new(true, &config, fake_vb)?;
            let model = Model::QLlama(QLlama::load(vb, config.clone())?);
            (model, config, cache)
        } else {
            // bin
            let mut file = std::fs::File::open(config_path)?;
            let config = Config::from_reader(&mut file)?;
            println!("{config:?}");
            let weights = TransformerWeights::from_reader(&mut file, &config, &device)?;
            let vb = weights.var_builder(&config, &device)?;
            let cache = model::Cache::new(true, &config, vb.pp("rot"))?;
            let model = Model::Llama(Llama::load(vb, config.clone())?);
            (model, config, cache)
        };

        let logits_processor = LogitsProcessor::new(299792458, temp, top_p);

        Ok(Self {
            model: Arc::new(Mutex::new(model)),
            model_id,
            device,
            tokenizer,
            logits_processor: Arc::new(Mutex::new(logits_processor)),
            cache,
            config,
        })
    }

    pub fn run(
        &mut self,
        prompt: &str,
        sample_len: usize,
        repeat_last_n: usize,
        repeat_penalty: f32,
        tx: mpsc::Sender<String>,
    ) -> Result<f64, anyhow::Error> {
        let mut index_pos = 0;
        let tokenizer = self.tokenizer.clone();
        let tokenizer = tokenizer.lock().unwrap();
        let mut tokens = tokenizer
            .encode(prompt, true)
            .map_err(E::msg)?
            .get_ids()
            .to_vec();
        let mut tokenizer = TokenOutputStream::new(tokenizer.to_owned());
        let model = self.model.clone();
        let model = model.lock().unwrap();

        let start_gen = std::time::Instant::now();
        for index in 0.. {
            if tokens.len() >= self.config.seq_len {
                break;
            }
            let context_size = if index > 0 { 1 } else { tokens.len() };
            let ctxt = &tokens[tokens.len().saturating_sub(context_size)..];
            let input = Tensor::new(ctxt, &self.device)?.unsqueeze(0)?;
            let logits = model.forward(&input, index_pos, &mut self.cache)?;
            let logits = logits.i((0, logits.dim(1)? - 1))?;
            let logits = if repeat_penalty == 1. || tokens.is_empty() {
                logits
            } else {
                let start_at = tokens.len().saturating_sub(repeat_last_n);
                candle_transformers::utils::apply_repeat_penalty(
                    &logits,
                    repeat_penalty,
                    &tokens[start_at..],
                )?
            };
            index_pos += ctxt.len();

            let next_token = self
                .logits_processor
                .clone()
                .lock()
                .unwrap()
                .sample(&logits)?;
            tokens.push(next_token);
            if let Some(t) = tokenizer.next_token(next_token)? {
                let _ = tx.send(t);
            }
        }

        let dt = start_gen.elapsed();
        Ok(tokens.len() as f64 / dt.as_secs_f64())
    }
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
