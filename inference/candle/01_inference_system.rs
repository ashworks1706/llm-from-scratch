// first we tokenize what we have
// then we use schedular batcher for assigning kv cache slots and scheduling model execution
// then we use candle to run the model layers
// then we convert the raw logits to token ids and then use tokenizer again to convert it to text


// key latency are TTFT -> time to first token, and TTT -> time to total tokens. We want to minimize TTFT and maximize TTT.
// time per output token is latency , dominated by memory bandwidth 
// throughput is tokens per second 

#[allow(unused_imports)]
use candle_core::{Device, Result, Tensor};
use tokenizers::Tokenizer;
fn main() -> Result<()> {
    let device = Device::cuda_if_available(0)?;

    let dummy_input = candle_core::Tensor::zeros((1,4,8), candle_core::DType::F32, &device)?;
    // Tensor::zeros creates a python equivalent shaped multidimensional array like np.zeros((1,4,8), dtype=np.float32) which looks like 
    // [[[0. 0. 0. 0. 0. 0. 0. 0.]
    //   [0. 0. 0. 0. 0. 0. 0.]
    //   [0. 0. 0. 0. 0. 0. 0.]
    //   [0. 0. 0. 0. 0. 0. 0.]]] 
    // here there's 1 batch, 4 rows and 8 columns.

    // so basically we specify the shape of the tensor we want to create, the data type of the tensor, and the device on which we want to create the tensor. The function returns a Result<Tensor> which is a wrapper around the Tensor object. If the tensor creation is successful, we can use the tensor object for further computations.


    // tokenize the input -- this is cpu
    let tokenizer = Tokenizer::from_pretrained("gpt2", None).unwrap();

    let input_text = "Hello, how are you?";

    let encoding = tokenizer.encode(input_text, true).unwrap();

    // memory allocation (model weights, kv cache, tmp activations)
    let seq_len = encoding.get_ids().len();

    let hidden_dim = 64;

    let weights = Tensor::randn(0.0f32, 1.0, (hidden_dim, hidden_dim), &device)?;


    println!("weights: {:?}", weights);


    // println!("dummy_input: {:?}", dummy_input);
    println!("tokens: {:?}", encoding.get_ids());
    Ok(())
}
