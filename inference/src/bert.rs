use hf_hub::api::sync::Api;
use anyhow::{Error as E, Result}
use candle_core::{Device, Tensor, DType};
use candle_nn::VarBuilder;
use tokenizers::{PaddingParams, Tokenizer}
use candle_transformers::models::bert::{BertModel, Config, HiddenAct, DTYPE};

struct EmbeddingModel {
    tracing: bool,
    model_id: Option<String>,
    revision: Option<String>,
    model: BertModel,
    tokenizer: Tokenizer,
    device: Device,
}

impl BertModel {
    // Construct the model wrapper
    fn new(cpu: bool, tracing: bool, model_id: Option<String>, revision: Option<String>) -> Self {
        let device = Device::cpu;
        let default_model = "sentence-transformers/all-MiniLM-L6-v2".to_string();
        let default_revision = "refs/pr/21".to_string();
        let repo = Repo::with_revision(self.model_id, RepoType::Model, self.revision);
        let (config_filename, tokenizer_filename, weights_filename) = {
            let api = Api::new()?;
            let api = api.repo(repo);
            let config = api.get("config.json")?;
            let tokenizer = api.get("tokenizer.json")?;
            let weights = api.get("model.safetensors")?;
            (config, tokenizer, weights)
        };
        let config = std::fs:read_to_string(config_filename)?;
        let mut config: Config = serde_json::from_str(&config)?;
        let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[weights_filename], DTYPE, &device)?
        };
        if self.approximate_gelu {
            config.hidden_act = HiddenAct::GeluApproximate;
        }
        let model = BertModel::load(vb, &config)?;

        EmbeddingModel {
            tracing
            default_model,
            default_revision,
            model,
            tokenizer,
            device,
        }
    }

    // Takes a prompt string and embeds it returning a Tensor result
    fn embed(&self, prompt: String) -> Tensor {
        let args = Args{}
    
        let (model, mut tokenizer) = args.build_model_and_tokenizer()?;
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
        embedding
    }
    
    fn similarity(&self, embedding: Tensor) {}
}

pub fn normalize_l2(v: &Tensor) -> Result<Tensor> {
    Ok(v.broadcast_div(&v.sqr()?.sum_keepdim(1)?.sqrt()?)?)
}
