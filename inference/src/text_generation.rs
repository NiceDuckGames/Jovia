use anyhow::{Error as E, Result};
use candle_core::utils::cuda_is_available;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::phi::{Config as PhiConfig, Model as Phi};
use hf_hub::api::sync::Api;
use hf_hub::{Repo, RepoType};
use std::sync::mpsc::{self, Receiver};
use std::thread;
use std::time::Duration;
use tokenizers::{PaddingParams, Tokenizer};

pub struct TextGeneration {
    model: Phi,
    model_id: Option<String>,
    revision: Option<String>,
    device: Device,
    tokenizer: Tokenizer,
    logits_processor: LogitsProcessor,
    repeat_penalty: f32,
    repeat_last_n: usize,
    verbose_prompt: bool,
}

impl TextGeneration {
    pub fn new(
        model_id: Option<String>,
        revision: Option<String>,
        seed: u64,
        temp: Option<f64>,
        top_p: Option<f64>,
        repeat_penalty: f32,
        repeat_last_n: usize,
        verbose_prompt: bool,
    ) -> Result<Self, E> {
        let device = Device::Cpu;
        let mut default_model = "microsoft/phi-2".to_string();
        let mut default_revision = "main".to_string();

        if let Some(model_id) = model_id {
            default_model = model_id;
        }

        if let Some(revision) = revision {
            default_revision = revision;
        }

        let repo = Repo::with_revision(
            default_model.clone(),
            RepoType::Model,
            default_revision.clone(),
        );
        let (config_filename, tokenizer_filename, weights_filenames) = {
            let api = Api::new()?;
            let api = api.repo(repo);
            let config = api.get("config.json")?;
            let tokenizer = api.get("tokenizer.json")?;
            //let weights = api.get("model.safetensors.index.json")?;
            let weights = hub_load_safetensors(&api, "model.safetensors.index.json")?;
            (config, tokenizer, weights)
        };

        let config = std::fs::read_to_string(config_filename)?;
        let mut config: PhiConfig = serde_json::from_str(&config)?;
        let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&weights_filenames, DType::F32, &device)?
        };
        let model = Phi::new(&config, vb)?;
        let logits_processor = LogitsProcessor::new(seed, temp, top_p);

        Ok(Self {
            model,
            model_id: Some(default_model),
            revision: Some(default_revision),
            device,
            tokenizer,
            logits_processor,
            repeat_penalty,
            repeat_last_n,
            verbose_prompt,
        })
    }

    pub fn run(
        &mut self,
        prompt: &str,
        sample_len: usize,
    ) -> Result<Receiver<String>, anyhow::Error> {
        // TODO: implement this function, possibly in a manner that allows token streaming.

        let tokens = self.tokenizer.encode(prompt, true).map_err(E::msg)?;
        if tokens.is_empty() {
            anyhow::bail!("Empty prompts are not supported in the phi model.")
        }
        if self.verbose_prompt {
            for (token, id) in tokens.get_tokens().iter().zip(tokens.get_ids().iter()) {
                let token = token.replace('_', " ").replace("<0x0A>", "\n");
                println!("{id:7} -> '{token}'");
            }
        }
        let mut tokens = tokens.get_ids().to_vec();
        let mut generated_tokens = 0usize;
        let eos_token = match self.tokenizer.get_vocab(true).get("<|endoftext|>") {
            Some(token) => *token,
            None => anyhow::bail!("cannot find the endoftext token"),
        };

        // Channel that will be used to consume tokens as they are generated
        let (tx, rx) = mpsc::channel();

        // spawn a producer thread
        thread::spawn(move || -> Result<()> {
            for index in 0..sample_len {
                let context_size = if index > 0 { 1 } else { tokens.len() };
                let ctxt = &tokens[tokens.len().saturating_sub(context_size)..];
                let input = Tensor::new(ctxt, &self.device)?.unsqueeze(0)?;
                let logits = self.model.forward(&input)?;
                let logits = logits.squeeze(0)?.to_dtype(DType::F32)?;
                let logits = if self.repeat_penalty == 1. {
                    logits
                } else {
                    let start_at = tokens.len().saturating_sub(self.repeat_last_n);
                    candle_transformers::utils::apply_repeat_penalty(
                        &logits,
                        self.repeat_penalty,
                        &tokens[start_at..],
                    )?
                };

                let next_token = self.logits_processor.sample(&logits)?;
                tokens.push(next_token);
                if next_token == eos_token {
                    break;
                }
                let token = self.tokenizer.decode(&[next_token], true).map_err(E::msg)?;
                if tx.send(token).is_err() {
                    break; // receiver has been dropped exit the loop
                }
            }
            Ok(())
        });

        Ok(rx) // return the receiving end of the channel to consume tokens
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
