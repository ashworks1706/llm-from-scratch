
#[allow(unused_imports)]
use candle_core::{DType, Device, Tensor};

fn main() -> anyhow::Result<()> {
    // okay so lets look at how candle tensors wokr 
    // tensors are basically ways ot express multiple relationships between multiple dimensions of data. they are like multidimensional arrays, they can be used to represent images, audio, video, text, etc.
    // how? well if u take an image, pass it through a neural network, the output of each layer is a tensor, but wait, what exactly do i mean by that? 
    // An image is converted into a tensor by translating the raw electrical voltages captured by a camera sensor into physical numbers, which are then organized into a structured, multi-dimensional matrix.
    // suppose there's a 4x4 image
    // for example, on frontend there's upload button, we click on it, we select that image, then we click upload
    // now on server we recieve image, then under the hood we decode it's compressed data into raw byte array which is sequential and is a structure raw matrix, then we loop through rows and columns to find specific memory address for any X and y coordinate, like addr = base_addr + (y * width + x) * bytes_per_pixel, then we read the pixel value at that address, then frameworks like candle, wrap this block of memory with tensor metadata, which is basically a struct that contains information about the shape, layout, dtype and device placement of the tensor. this way we can use the tensor to perform operations on the image without having to worry about the underlying memory representation.

    // okay anways thats too mcuh, letsp roceed

    let device = Device::cuda_if_available(0)?;

    let tf32 = Tensor::zeros(&[1, 4, 8], DType::F32, &device)?; // why & here for arr? because if we passed arr directly, it would be moved into the function and we would lose ownership of it. by passing a reference, we can keep ownership of the array and still allow the function to read its contents.

    // 1,4,8 -> 1*4*8 = 32 elements, each element is 4 bytes, so total size is 32*4 = 128 bytes
    let device = tf32.device();


    println!("Tensor size: {:?}", tf32.dtype().size_in_bytes());
    println!("Tensor device: {:?}", device);


    // normally, we use f32 for training, but for inference we can use f16 or bf16 to save memory and speed up computation. this is because f16 and bf16 have lower precision than f32, but they are still good enough for inference. how so?

    // f16 has 1 sign bit, 5 exponent bits, and 10 mantissa bits, while f32 has 1 sign bit, 8 exponent bits, and 23 mantissa bits. this means that f16 can represent numbers in the range of approximately 6.1e-5 to 65504, while f32 can represent numbers in the range of approximately 1.4e-45 to 3.4e38. this means that f16 can represent numbers with less precision than f32, but it can still represent a wide range of numbers. bf16 has 1 sign bit, 8 exponent bits, and 7 mantissa bits, which means that it can represent numbers in the range of approximately 1.2e-38 to 3.4e38, which is similar to f32.

    // but why do we care about precision? well, in deep learning, we care about precision because it affects the accuracy of the model. if we use lower precision, we might lose some information and the model might not be able to learn as well. but for inference, we don't need to learn, we just need to make predictions, so we can use lower precision to save memory and speed up computation.

    let tf16 = tf32.to_dtype(DType::F16)?;
    let bf16 = tf32.to_dtype(DType::BF16)?;

    println!("Tensor size: {:?}", tf16.dtype().size_in_bytes());
    println!("Tensor size: {:?}", bf16.dtype().size_in_bytes());


    // lets measure their performance real quick 

    let start = std::time::Instant::now();
    let _ = tf32.add(&tf32)?;
    let duration = start.elapsed();
    println!("Time taken for f32 addition: {:?}", duration);

    let start = std::time::Instant::now();
    let _ = tf16.add(&tf16)?;
    let duration = start.elapsed();
    println!("Time taken for f16 addition: {:?}", duration);

    let start = std::time::Instant::now();
    let _ = bf16.add(&bf16)?;
    let duration = start.elapsed();
    println!("Time taken for bf16 addition: {:?}", duration);


    

    // contiguous layouts are basically a way to store multi-dimensional arrays in a linear fashion in memory, so that the elements of the array are stored in a contiguous block of memory. this is important because it allows for efficient access to the elements of the array, as well as efficient use of memory.

    let transposed = tf32.transpose(1, 2)?;

    // transpose() does not change the underlying data, it just changes the way we view the data. that means that the transposed tensor is still backed by the same memory as the original tensor, but the way we access the data is different. this is why the transposed tensor is not contiguous, because the elements are not stored in a contiguous block of memory, but rather in a strided fashion.

    // if we wanted to not have a strided tensor, we would have to create a new tensor that is contiguous, and copy the data from the original tensor into the new tensor. this is what the contiguous() method does, it creates a new tensor that is contiguous, and copies the data from the original tensor into the new tensor.

    println!("Transposed tensor is contiguous: {:?}", transposed.is_contiguous());

    // lets test if its contiguous or not with a normal math operation

    // let result = transposed.add(&tf32)?; // this will fail because the transposed tensor is not contiguous, and the add operation requires contiguous tensors. the error message will be something like "add: input tensor is not contiguous"

    let contiguous = transposed.contiguous()?; 
    println!("Contiguous tensor is contiguous: {:?}", contiguous.is_contiguous());

    // now when its contiguous now it can be used in operations that require contiguous tensors, like add, sub, mul, div, etc.

    let result = contiguous.add(&contiguous)?;

    println!("Result tensor is contiguous: {:?}", result.is_contiguous());
    println!("Result tensor device: {:?}", result.device());
    println!("Results: {:?}", result);
    

    
    Ok(())
}
