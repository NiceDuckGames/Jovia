use anyhow::{Error as E, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::phi::{Config as PhiConfig, Model as Phi};
use hf_hub::api::sync::Api;
use hf_hub::{Repo, RepoType};
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
        let (config_filename, tokenizer_filename, weights_filename) = {
            let api = Api::new()?;
            let api = api.repo(repo);
            let config = api.get("config.json")?;
            let tokenizer = api.get("tokenizer.json")?;
            let weights = api.get("model.safetensors")?;
            (config, tokenizer, weights)
        };

        let config = std::fs::read_to_string(config_filename)?;
        let mut config: PhiConfig = serde_json::from_str(&config)?;
        let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[weights_filename], DType::F32, &device)?
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

    pub fn run(&mut self, prompt: &str, sample_len: usize) -> Result<()> {
        // TODO: implement this function, possibly in a manner that allows token streaming.
        Ok(())
    }
}
