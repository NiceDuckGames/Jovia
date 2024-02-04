// expose an inference API
pub mod embedding;
pub mod prompts;
pub mod text_generation;

#[cfg(test)]
mod tests {
    use crate::text_generation::TextGeneration;

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
        let sentences: Vec<String> = [
            "The cat sits outside",
            "A man is playing guitar",
            "I love pasta",
            "The new movie is awesome",
            "The cat plays in the garden",
            "A woman watches TV",
            "The new movie is so great",
            "Do you like pizza?",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();

        let n_sentences = sentences.len();

        let em = EmbeddingModel::new(true, false, None, None).unwrap();
        let embeddings = em.embed_batch(sentences);

        assert!(true);
    }

    #[test]
    fn test_embedding_consine_similarity() {
        let sentences: Vec<String> = ["The cat sits outside", "The cat sits outside"]
            .iter()
            .map(|s| s.to_string())
            .collect();

        let em = EmbeddingModel::new(true, false, None, None).unwrap();
        let embeddings = em.embed_batch(sentences).unwrap();

        let e1 = embeddings.get(0).unwrap();
        let e2 = embeddings.get(1).unwrap();
        let similarity = cos_similarity(e1, e2).unwrap();

        assert_eq!(similarity, 1.0);
    }

    #[test]
    fn test_textgeneration_instantiate() {
        let prompt = "What is the capital Ireland?".to_string();
        let mut pipeline =
            TextGeneration::new(None, None, 299792458, None, None, 1.1, 64, false).unwrap();
        pipeline.run(&prompt, 500).unwrap();
    }
}
