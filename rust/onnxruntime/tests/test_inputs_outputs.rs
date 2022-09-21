use image::{imageops::FilterType, ImageBuffer, Luma, Pixel};
use ndarray::Array;
use ndarray::Ix4;
use onnxruntime::tensor::WithOutputTensor;
use onnxruntime::{
    download::vision::DomainBasedImageClassification, environment::Environment, session::Session,
    GraphOptimizationLevel, LoggingLevel,
};
use std::path::Path;
use test_log::test;

fn mnist_session() -> (Environment, Session) {
    let environment = Environment::builder()
        .with_name("integration_test")
        .with_log_level(LoggingLevel::Warning)
        .build()
        .unwrap();

    let session = environment
        .new_session_builder()
        .unwrap()
        .with_graph_optimization_level(GraphOptimizationLevel::Basic)
        .unwrap()
        .with_intra_op_num_threads(1)
        .unwrap()
        .with_model_downloaded(DomainBasedImageClassification::Mnist)
        .expect("Could not download model from file");

    (environment, session)
}

fn input_vector(session: &Session) -> Vec<f32> {
    const IMAGE_TO_LOAD: &str = "mnist_5.jpg";

    let input0_shape: Vec<usize> = session.inputs[0].dimensions().map(|d| d.unwrap()).collect();
    let output0_shape: Vec<usize> = session.outputs[0]
        .dimensions()
        .map(|d| d.unwrap())
        .collect();

    assert_eq!(input0_shape, [1, 1, 28, 28]);
    assert_eq!(output0_shape, [1, 10]);

    // Load image and resize to model's shape, converting to RGB format
    let image_buffer: ImageBuffer<Luma<u8>, Vec<u8>> = image::open(
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests")
            .join("data")
            .join(IMAGE_TO_LOAD),
    )
    .unwrap()
    .resize(
        input0_shape[2] as u32,
        input0_shape[3] as u32,
        FilterType::Nearest,
    )
    .to_luma8();
    (0..1)
        .flat_map(|_i| (0..1))
        .flat_map(|c| (0..28).map(move |i| (c, i)))
        .flat_map(|(c, i)| (0..28).map(move |j| (c, i, j)))
        .map(|(c, i, j)| {
            let pixel = image_buffer.get_pixel(i as u32, j as u32);
            let channels = pixel.channels();

            // range [0, 255] -> range [0, 1]
            (channels[c] as f32) / 255.0
        })
        .collect::<Vec<_>>()
}

#[test]
fn test_ndarray_to_slice() {
    let (_env, session) = mnist_session();

    let array: Array<f32, Ix4> =
        Array::from_shape_vec((1, 1, 28, 28), input_vector(&session)).unwrap();

    // Batch of 1
    let input_tensor_values = vec![array];

    // Perform the inference
    let outputs: onnxruntime::Result<Vec<WithOutputTensor<&[f32]>>> =
        session.run(input_tensor_values);

    let output = &outputs.unwrap()[0];

    assert_eq!(output.len(), 10);

    assert!(output[4] > output[3]);
    assert!(output[4] > output[1]);
}
