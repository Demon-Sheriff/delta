use deltaml::activations::{ReluActivation, SoftmaxActivation};
use deltaml::common::ndarray::{IxDyn, Shape};
use deltaml::dataset::{ImageDatasetOps, MnistDataset};
use deltaml::losses::SparseCategoricalCrossEntropyLoss;
use deltaml::neuralnet::{Dense, Flatten, Sequential};
use deltaml::optimizers::Adam;

#[tokio::main]
async fn main() {
    // Create a neural network
    let mut model = Sequential::new()
        // .add(Conv2D::new(32, 3, 1, 1, Some(Box::new(ReluActivation::new())), true)) // Conv2D layer with 32 filters, kernel size 3x3
        // .add(MaxPooling2D::new(2, 2)) // MaxPooling2D layer with pool size 2x2
        .add(Flatten::new(Shape::from(IxDyn(&[28, 28])))) // Flatten layer
        .add(Dense::new(128, Some(ReluActivation::new()), true)) // Dense layer with 128 units
        .add(Dense::new(10, None::<SoftmaxActivation>, false)); // Output layer with 10 classes

    // Display the model summary
    model.summary();

    // Chose either CPU or GPU
    model.use_optimized_device();

    // Define an optimizer
    let optimizer = Adam::new(0.001);

    // Compile the model
    // model.compile(optimizer, CrossEntropyLoss::new());
    model.compile(optimizer, SparseCategoricalCrossEntropyLoss::new());

    // Loading the train and test dataset
    let mut train_data = MnistDataset::load_train().await;
    #[allow(unused_mut)]
    let mut test_data = MnistDataset::load_test().await;
    #[allow(unused_mut)]
    let mut val_data = MnistDataset::load_val().await;

    println!("Training the model...");
    println!("Train dataset size: {}", train_data.len());

    let epoch = 10;
    let batch_size = 32;

    match model.fit(&mut train_data, epoch, batch_size) {
        Ok(_) => println!("Model trained successfully"),
        Err(e) => println!("Failed to train model: {}", e),
    }

    // Validate the model
    match model.validate(&mut val_data, batch_size) {
        Ok(validation_loss) => println!("Validation Loss: {:.6}", validation_loss),
        Err(e) => println!("Failed to validate model: {}", e),
    }

    // Evaluate the model
    let accuracy = match model.evaluate(&mut test_data, batch_size) {
        Ok(accuracy) => accuracy,
        Err(e) => panic!("Failed to evaluate model: {}", e),
    };
    println!("Test Accuracy: {:.2}%", accuracy * 100.0);

    // Save the model
    model.save(".cache/models/mnist/mnist").unwrap();
}
