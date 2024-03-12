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
        use std::io::Write;
        use std::sync::mpsc::{self, Receiver, Sender};
        use std::thread;
        use std::time::Instant;

        let prompt = "What is the capital Ireland?".to_string();

        println!("Loading model");
        let now = Instant::now();
        let mut pipeline =
            TextGeneration::new(None, None, 299792458, None, None, 1.1, 64, false).unwrap();
        let elapsed = now.elapsed();
        println!("Took {:.2?} to load model", elapsed);
        let _ = now;
        let _ = elapsed;

        println!("Text pipeline created, beginning inference");

        let (tx, rx): (Sender<String>, Receiver<String>) = mpsc::channel();

        let now = Instant::now();
        // produce tokens
        let handle = thread::spawn(move || {
            pipeline.run(&prompt, 256, tx).unwrap();
        });

        // consume the tokens
        loop {
            match rx.try_recv() {
                Ok(token) => {
                    print!("{token}");
                }
                Err(TryRecvError::Empty) => {}
                Err(TryRecvError::Disconnected) => {
                    print!("All tokens consumed");
                    break;
                }
            }
            std::io::stdout().flush()?;
        }

        // wait for the producer thread to finish
        handle.join().unwrap();
        let elapsed = now.elapsed();

        println!("Took {:.2?} to complete inference", elapsed);
        Ok(())
    }
}
