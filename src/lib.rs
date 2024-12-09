use anyhow::Result;
use candle_core::utils::{cuda_is_available, metal_is_available};
use candle_core::{Device, D};
use ferritin_amplify::ModelOutput;
use polars::prelude::{df, DataFrame};

pub fn device(cpu: bool) -> Result<Device> {
    if cpu {
        Ok(Device::Cpu)
    } else if cuda_is_available() {
        Ok(Device::new_cuda(0)?)
    } else if metal_is_available() {
        Ok(Device::new_metal(0)?)
    } else {
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        {
            println!(
                "Running on CPU, to run on GPU(metal), build this example with `--features metal`"
            );
        }
        #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
        {
            println!("Running on CPU, to run on GPU, build this example with `--features cuda`");
        }
        Ok(Device::Cpu)
    }
}

#[derive(Default)]
enum OutputType {
    #[default]
    CSV,
    PARQUET,
}

#[derive(Default)]
pub struct OutputConfig {
    pub contact_output: OutputType,
    pub top_k_output: OutputType,
}

pub trait ModelIO {
    fn generate_contacts(&self, config: OutputConfig) -> Result<DataFrame>;
    fn top_hits(&self,OutputConfig) -> Result<DataFrame>;
    fn to_disk(&self,OutputConfig);
}

impl ModelIO for ModelOutput {
    fn top_hits(&self, OutputConfig) -> Result<DataFrame> {
        // let predictions = self.logits.argmax(D::Minus1)?;
        todo!("Need to think through the API a bit");
    }
    fn generate_contacts(&self, OutputConfig) -> Result<DataFrame> {
        let apc = self.get_contact_map()?;
        if apc.is_none() {
            Ok(DataFrame::empty())
        } else {
            let restensor = apc.unwrap();
            let (seqlen, _seqlen2, _) = restensor.dims3()?;
            let contact_probs = candle_nn::ops::softmax(&restensor, D::Minus1)?;
            let max_probs = contact_probs.max(D::Minus1)?;
            let flattened = max_probs.flatten_all()?;
            let values: Vec<f32> = flattened.to_vec1()?;
            let indices_1: Vec<i32> = (1..=seqlen)
                .map(|x| x as i32)
                .cycle()
                .take(seqlen * seqlen)
                .collect();
            let indices_2: Vec<i32> = (1..=seqlen)
                .map(|x| x as i32)
                .flat_map(|x| std::iter::repeat(x).take(seqlen))
                .collect();
            let df = df! [
                "index_1" => &indices_1,
                "index_2" => &indices_2,
                "value" => &values,
            ]?;
            Ok(df)
        }
    }
    fn to_disk(&self, OutputConfig) {}
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
        let indices_01: Vec<i32> = (1..=width)
            .map(|x| x as i32)
            .cycle()
            .take(width * width)
            .collect();

        let indices_02: Vec<i32> = (1..=width)
            .map(|x| x as i32)
            .flat_map(|x| std::iter::repeat(x).take(width))
            .take(width * width)
            .collect();

        let mut df = df! [
            "index_1" => &indices_01,
            "index_2" => &indices_02,
            "values" => &values,
        ]?;

        let path = "output.parquet";
        ParquetWriter::new(File::create(path)?).finish(&mut df)?;

        let csv_path = "output.csv";
        CsvWriter::new(File::create(csv_path)?).finish(&mut df)?;
        Ok(())
    }
}
