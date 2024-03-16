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
        let mut tokens: Vec<String> = Vec::new();

        println!("Loading model");
        let now = Instant::now();
        let mut pipeline = TextGeneration::new(
            "karpathy/tinyllamas".to_string(),
            "stories15M.bin".to_string(),
            "hf-internal-testing/llama-tokenizer".to_string(),
            None,
            None,
        )
        .unwrap();

        let elapsed = now.elapsed();
        println!("Took {:.2?} to load model", elapsed);
        let _ = now;
        let _ = elapsed;

        let now = Instant::now();
        // Inference loop
        let mut index_pos = 0;
        for _i in 0..sample_len {
            if prompt.len() > pipeline.config.seq_len {
                break;
            }
            match pipeline.next_token(prompt.clone(), repeat_penalty, repeat_last_n, index_pos) {
                Ok((Some(next_token), new_index_pos)) => {
                    prompt.push_str(&next_token);
                    tokens.push(next_token);
                    index_pos = new_index_pos;
                }
                Ok((None, _)) => {}
                Err(e) => {
                    println!("error {:?}", e);
                    break;
                }
            }
        }
        let elapsed = now.elapsed();

        println!("Took {:.2?} to complete inference", elapsed);
        //println!("{:?} tok/s", tokens.len() as u64 / elapsed.as_secs());
        println!("Generated:");
        println!("{:?}", tokens.join(""));
        Ok(())
    }
}
