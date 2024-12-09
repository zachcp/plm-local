use anyhow::Result;
use candle_core::utils::{cuda_is_available, metal_is_available};
use candle_core::{Device, D};
use ferritin_amplify::ModelOutput;
use polars::io::parquet::*;
use polars::prelude::*;
use polars::prelude::{df, CsvWriter, DataFrame, ParquetWriter};
use tokenizers::Tokenizer;

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

pub enum OutputType {
    CSV,
    PARQUET,
}

pub struct OutputConfig {
    pub contact_output: OutputType,
    pub top_k_output: OutputType,
    pub sequence: String,
    pub outdir: String,
    pub tokenizer: Tokenizer,
}

pub trait ModelIO {
    fn generate_contacts(&self, config: &OutputConfig) -> Result<DataFrame>;
    fn top_hits(&self, config: &OutputConfig) -> Result<DataFrame>;
    fn to_disk(&self, config: &OutputConfig) -> Result<()>;
}

impl ModelIO for ModelOutput {
    fn top_hits(&self, config: &OutputConfig) -> Result<DataFrame> {
        // let predictions = self.logits.argmax(D::Minus1)?;
        todo!("Need to think through the API a bit");
    }
    fn generate_contacts(&self, config: &OutputConfig) -> Result<DataFrame> {
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
    fn to_disk(&self, config: &OutputConfig) -> Result<()> {
        // Validated the pytorch/python AMPLIFY model has the same dims...
        // 350M: Contact Map: Ok(Some(Tensor[dims 254, 254, 480; f32, metal:4294969344]))
        // 120M: Contact Map: Ok(Some(Tensor[dims 254, 254, 240; f32, metal:4294969344]))
        // Lets take the max() of the Softmax values....

        let mut contacts = self.generate_contacts(config)?;

        println!("Writing Contact Parquet File");
        std::fs::create_dir_all(&config.outdir)?;
        let outdir = std::path::PathBuf::from(&config.outdir);
        match &config.contact_output {
            OutputType::CSV => {
                let contact_map_file = outdir.join("contact_map.csv");
                let mut file = std::fs::File::create(&contact_map_file)?;
                CsvWriter::new(&mut file).finish(&mut contacts)?;
            }
            OutputType::PARQUET => {
                let contact_map_file = outdir.join("contact_map.parquet");
                let mut file = std::fs::File::create(&contact_map_file)?;
                ParquetWriter::new(&mut file).finish(&mut contacts)?;
            }
        }

        println!("Writing Top Output...");
        let predictions = self.logits.argmax(D::Minus1)?;
        let indices: Vec<u32> = predictions.to_vec2()?[0].to_vec();
        let decoded = config
            .tokenizer
            .decode(indices.as_slice(), true)
            .map_err(|e| anyhow::anyhow!("{}", e))?;
        // std::fs::write(&decoded_path, &decoded)?;
        // let decoded = &config.tokenizer.decode(indices.as_slice(), true)?;
        // println!("Decoded: {:?}", decoded);

        let decoded_path = outdir.join("decoded.txt");
        std::fs::write(&decoded_path, decoded)?;

        println!("Writing Sequence...");
        let sequence_path = outdir.join("sequence.txt");
        std::fs::write(&sequence_path, &config.sequence)?;

        Ok(())
    }
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
