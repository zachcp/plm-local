use anyhow::{Error as E, Result};
use candle_core::{DType, Tensor};
use candle_hf_hub::{api::sync::Api, Repo, RepoType};
use candle_nn::VarBuilder;
use clap::Parser;
use ferritin_amplify::{AMPLIFYConfig as Config, AMPLIFY};
use plm_local::{device, ModelIO, OutputConfig, OutputType};
use tokenizers::Tokenizer;

pub const DTYPE: DType = DType::F32;

#[derive(Parser, Debug)]
#[command(
    author,
    version,
    about,
    long_about = "plm-local in an application to explore the use of protein-language models locally on your machine. Currently it supports only the AMPLIFY model and string input on Macos. On the first run, the weights wil be downloaded from the HuggingFace repo."
)]
struct Args {
    /// Run on CPU rather than on GPU.
    #[arg(long, default_value_t = false)]
    cpu: bool,

    /// Which AMPLIFY Model to use, either '120M' or '350M'.
    #[arg(long, value_parser = ["120M", "350M"], default_value = "120M")]
    model_id: String,

    /// Protein String
    #[arg(long)]
    protein_string: Option<String>,

    /// Path to a protein FASTA file
    #[arg(long)]
    protein_fasta: Option<String>,

    /// Output directory for files
    #[arg(long)]
    output_dir: Option<String>,
}

impl Args {
    fn build_model_and_tokenizer(&self) -> Result<(AMPLIFY, Tokenizer)> {
        let device = device(self.cpu)?;
        let (model_id, revision) = match self.model_id.as_str() {
            "120M" => ("chandar-lab/AMPLIFY_120M", "main"),
            "350M" => ("chandar-lab/AMPLIFY_350M", "main"),
            _ => panic!("Amplify models are either `120M` or `350M`"),
        };
        let repo = Repo::with_revision(model_id.to_string(), RepoType::Model, revision.to_string());
        let (config_filename, tokenizer_filename, weights_filename) = {
            let api = Api::new()?;
            let api = api.repo(repo);
            let config = api.get("config.json")?;
            let tokenizer = api.get("tokenizer.json")?;
            let weights = api.get("model.safetensors")?;
            (config, tokenizer, weights)
        };
        let config_str = std::fs::read_to_string(config_filename)?;
        let config_str = config_str
            .replace("SwiGLU", "swiglu")
            .replace("Swiglu", "swiglu");
        let config: Config = serde_json::from_str(&config_str)?;
        let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;
        let vb =
            unsafe { VarBuilder::from_mmaped_safetensors(&[weights_filename], DTYPE, &device)? };
        let model = AMPLIFY::load(vb, &config)?;
        Ok((model, tokenizer))
    }
}

fn main() -> Result<()> {
    let args = Args::parse();

    println!("Loading the Model and Tokenizer.......");
    let (model, tokenizer) = args.build_model_and_tokenizer()?;
    let device = &model.get_device();

    let protein_sequences = if let Some(seq) = args.protein_string {
        vec![seq]
    } else if let Some(fasta_path) = args.protein_fasta {
        todo!("fasta processing unimplimented")
        // std::fs::read_to_string(fasta_path)?
    } else {
        return Err(E::msg(
            "Either protein_string or protein_fasta must be provided",
        ));
    };

    // default is datetime-model
    let output_dir = args.output_dir.unwrap_or_else(|| {
        let now = chrono::Local::now();
        format!("{}_{}", now.format("%Y%m%d_%H%M%S"), args.model_id)
    });

    //
    for prot in protein_sequences.iter() {
        let config = OutputConfig {
            contact_output: OutputType::CSV,
            top_k_output: OutputType::CSV,
            sequence: prot.clone(),
            outdir: output_dir.clone(),
            tokenizer: tokenizer.clone(),
        };

        let tokens = tokenizer
            .encode(prot.to_string(), false)
            .map_err(E::msg)?
            .get_ids()
            .to_vec();
        let token_ids = Tensor::new(&tokens[..], device)?.unsqueeze(0)?;

        println!("Encoding.......");
        let encoded = model.forward(&token_ids, None, false, true)?;

        println!("Writing Outputs... ");
        let _ = encoded.to_disk(&config)?;
    }

    Ok(())
}
