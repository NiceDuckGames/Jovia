// expose an inference API
pub mod embedding;
pub mod prompts;
pub mod text_generation;

#[cfg(test)]
mod tests {
    use super::*;

    use anyhow::{Error as E, Result};
    use candle_core::Tensor;
    use embedding::*;

    #[test]
    fn instantiate_embedding_model() {
        let em_result = EmbeddingModel::new(true, false, None, None);
        match em_result {
            Ok(_) => {
                assert!(true);
            }
            Err(_) => {
                assert!(false);
            }
        }
    }

    #[test]
    fn test_sentence_embedding() {
        let sentences = [
            "The cat sits outside",
            "A man is playing guitar",
            "I love pasta",
            "The new movie is awesome",
            "The cat plays in the garden",
            "A woman watches TV",
            "The new movie is so great",
            "Do you like pizza?",
        ];
        let n_sentences = sentences.len();

        let em = EmbeddingModel::new(true, false, None, None).unwrap();

        let tokens = em
            .tokenizer
            .encode_batch(sentences.to_vec(), true)
            .map_err(E::msg)
            .unwrap();

        let token_ids = tokens
            .iter()
            .map(|tokens| {
                let tokens = tokens.get_ids().to_vec();
                Ok(Tensor::new(tokens.as_slice(), &em.device)?)
            })
            .collect::<Result<Vec<_>>>()
            .unwrap();

        let token_ids = Tensor::stack(&token_ids, 0).unwrap();
        let token_type_ids = token_ids.zeros_like().unwrap();
        let embeddings = em.model.forward(&token_ids, &token_type_ids).unwrap();

        assert!(true);
    }

    #[test]
    fn test_embedding_consine_similarity() {
        let sentences = ["The cat sits outside", "The cat sits outside"];
        let n_sentences = sentences.len();

        let em = EmbeddingModel::new(true, false, None, None).unwrap();

        let tokens = em
            .tokenizer
            .encode_batch(sentences.to_vec(), true)
            .map_err(E::msg)
            .unwrap();

        let token_ids = tokens
            .iter()
            .map(|tokens| {
                let tokens = tokens.get_ids().to_vec();
                Ok(Tensor::new(tokens.as_slice(), &em.device)?)
            })
            .collect::<Result<Vec<_>>>()
            .unwrap();

        let token_ids = Tensor::stack(&token_ids, 0).unwrap();
        let token_type_ids = token_ids.zeros_like().unwrap();
        let embeddings = em.model.forward(&token_ids, &token_type_ids).unwrap();

        let threshold = 0.7;
        let e1 = embeddings.get(0).unwrap();
        let e2 = embeddings.get(1).unwrap();
        let similarity = cos_similarity(e1, e2).unwrap();
        println!("self similarity {:?} {:?}", sentences.get(0), similarity);

        assert_eq!(similarity, 1.0);
    }
}
