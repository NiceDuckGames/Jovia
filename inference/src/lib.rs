// expose an inference API
pub mod embedding;
pub mod prompts;
pub mod text_generation;

#[cfg(test)]
mod tests {
    use std::sync::mpsc::TryRecvError;

    use crate::text_generation::TextGeneration;

    use super::*;

    use anyhow::{Error as E, Result};
    use candle_core::Tensor;
    use embedding::*;

    /*#[test]
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
    }*/

    #[test]
    fn test_textgeneration_run() -> Result<(), anyhow::Error> {
        use std::time::Instant;

        let mut prompt = "Once upon a time ".to_string();
        let repeat_penalty = 1.1;
        let repeat_last_n = 64;
        let sample_len = 255;

        println!("Loading model");
        let now = Instant::now();
        let mut pipeline = TextGeneration::new(
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0".to_string(),
            "model.safetensors".to_string(),
            None,
            None,
            None,
            None,
        )
        .unwrap();

        let eos_token = pipeline.tokenize("</s>".to_string())?[0];
        let mut tokens = pipeline.tokenize(prompt)?;

        let elapsed = now.elapsed();
        println!("Took {:.2?} to load model", elapsed);
        let _ = now;
        let _ = elapsed;

        let now = Instant::now();
        // Inference loop
        let mut index_pos = 0;
        let mut tokens_generated = 0;
        for i in 0..sample_len {
            let (context_size, context_index) = if pipeline.cache.use_kv_cache && i > 0 {
                (1, index_pos)
            } else {
                (tokens.len(), 0)
            };

            let (token, new_index_pos) = pipeline.next_token(
                tokens.clone(),
                repeat_penalty,
                repeat_last_n,
                context_size,
                context_index,
                index_pos,
            )?;

            //println!("{:?}", pipeline.encode(&[token.clone()]));
            tokens.push(token.clone());
            tokens_generated += 1;
            if token == eos_token {
                // If we get an eos token we stop generating
                break;
            }

            index_pos += new_index_pos;
        }
        let elapsed = now.elapsed();

        println!("Took {:.2?} to complete inference", elapsed);
        println!("{:?} tok/s", tokens_generated as u64 / elapsed.as_secs());
        println!("Generated:");
        let generated_text = pipeline.encode(&tokens);
        println!("{generated_text:?}");
        Ok(())
    }
}
