use anyhow::{Error as E, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config, HiddenAct, DTYPE};
use hf_hub::api::sync::Api;
use hf_hub::{Repo, RepoType};
use tokenizers::{PaddingParams, Tokenizer};

pub struct EmbeddingModel {
    tracing: bool,
    model_id: Option<String>,
    revision: Option<String>,
    model: BertModel,
    tokenizer: Tokenizer,
    device: Device,
}

impl EmbeddingModel {
    // Construct the model wrapper
    pub fn new(
        cpu: bool,
        tracing: bool,
        model_id: Option<String>,
        revision: Option<String>,
    ) -> Result<Self, E> {
        let device = Device::Cpu;
        let mut default_model = "sentence-transformers/all-MiniLM-L6-v2".to_string();
        let mut default_revision = "refs/pr/21".to_string();

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
        let mut config: Config = serde_json::from_str(&config)?;
        let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;
        let vb =
            unsafe { VarBuilder::from_mmaped_safetensors(&[weights_filename], DTYPE, &device)? };
        let model = BertModel::load(vb, &config)?;

        Ok(EmbeddingModel {
            tracing,
            model_id: Some(default_model),
            revision: Some(default_revision),
            model,
            tokenizer,
            device,
        })
    }

    // Takes a prompt string and embeds it returning a Tensor result
    pub fn embed(&self, prompt: String) -> Result<Tensor, E> {
        let model = &self.model;
        let mut tokenizer = self.tokenizer.clone();
        let device = &model.device;

        let tokenizer = tokenizer
            .with_padding(None)
            .with_truncation(None)
            .map_err(E::msg)?;

        let tokens = tokenizer
            .encode(prompt, true)
            .map_err(E::msg)?
            .get_ids()
            .to_vec();

        let token_ids = Tensor::new(&tokens[..], device)?.unsqueeze(0)?;
        let token_type_ids = token_ids.zeros_like()?;

        let embedding = model.forward(&token_ids, &token_type_ids)?;

        Ok(embedding)
    }
}

pub fn similarity(a: Tensor, b: Tensor) -> Result<i32, E> {
    Ok(1)
}
