use ferritin_amplify::ModelOutput;

pub trait ToParquet {
    fn top_hits(self);
    fn contacts(self);
}

impl ToParquet for ModelOutput {
    fn top_hits(self) {}
    fn contacts(self) {}
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Tensor};
    use polars::prelude::*;
    use std::fs::File;

    #[test]
    fn test_parquet_conversion() -> anyhow::Result<()> {
        let tensor = Tensor::new(&[[0f32, 1., 3.], [2., 3., 4.], [4., 5., 6.]], &Device::Cpu)?;
        let (length, width) = tensor.dims2()?;
        println!("Tensor Dims: {:?}. {}, {}", tensor.dims(), length, width);
        let flattened = tensor.flatten_all()?;

        let values: Vec<f32> = flattened.to_vec1()?;
        let indices: Vec<i32> = (1..=width)
            .map(|x| x as i32)
            .cycle()
            .take(width * width)
            .collect();

        let mut df = df! [
            "index_1" => &indices,
            "index_2" => &indices,
            "values" => &values,
        ]?;

        let path = "output.parquet";
        ParquetWriter::new(File::create(path)?).finish(&mut df)?;
        Ok(())
    }
}
