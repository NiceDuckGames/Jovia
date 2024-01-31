// expose an inference API
pub mod embedding;
pub mod prompts;
pub mod text_generation;

pub fn add(left: usize, right: usize) -> usize {
    left + right
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let result = add(2, 2);
        assert_eq!(result, 4);
    }

    use embedding::*;

    #[test]
    fn instantiate_embedding_modle() {
        let em = EmbeddingModel::new(true, false, None, None);
        println!("{em:?}");
    }
}
