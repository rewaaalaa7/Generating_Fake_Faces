# Face Generation with DCGAN and Autoencoder

This project implements two different approaches to generate and enhance facial images using deep learning: a Deep Convolutional Generative Adversarial Network (DCGAN) and an Autoencoder. The models are trained on the CelebA dataset to generate realistic human faces.

## Features

- DCGAN implementation for generating synthetic face images
- Autoencoder for image enhancement and reconstruction
- Early stopping mechanism to prevent overfitting
- Support for batch processing
- Visualization utilities for generated images

## Requirements

- Python 3.10+
- TensorFlow
- OpenCV (cv2)
- NumPy
- Matplotlib
- Pandas
- tqdm

## Installation

1. Clone this repository:
```bash
git clone [repository-url]
cd face-generation
```

2. Install the required packages:
```bash
pip install tensorflow opencv-python numpy matplotlib pandas tqdm
```

3. Download the CelebA dataset and place it in the appropriate directory.

## Project Structure

- `generating_fake_faces.py`: Main script containing both DCGAN and Autoencoder implementations
- Components:
  - DCGAN:
    - Generator: Creates synthetic images from random noise
    - Discriminator: Distinguishes between real and generated images
  - Autoencoder:
    - Encoder: Compresses input images
    - Latent Space: Dense layer representation
    - Decoder: Reconstructs images from compressed representation

## Usage

1. Train the DCGAN model:
```python
model = GAN(Generator, Discriminator)
model.compile(
    gen_optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    disc_optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    criterion=tf.keras.losses.BinaryCrossentropy(from_logits=True)
)
model.fit(X_train, epochs=100, batch_size=100)
```

2. Train the Autoencoder:
```python
autoencoder = Autoencoder()
autoencoder.compile(
    tf.keras.optimizers.Adam(learning_rate=0.001), 
    tf.keras.losses.BinaryCrossentropy()
)
autoencoder.fit(X_train, epochs=50)
```

3. Generate new faces:
```python
noise = tf.random.normal([num_images, 128])
generated_images = model.generator.predict(noise)
```

4. Enhance generated images:
```python
enhanced_images = autoencoder.predict(generated_images)
```

## Model Architecture

### DCGAN
- Generator:
  - Input: 128-dimensional noise vector
  - Multiple transposed convolution layers
  - Output: 64x64x3 RGB image
- Discriminator:
  - Input: 64x64x3 RGB image
  - Multiple convolution layers
  - Output: Binary classification

### Autoencoder
- Encoder:
  - 4 convolutional layers with batch normalization
  - Progressively reduces spatial dimensions
- Latent Space:
  - Flattening and dense layers
  - Bottleneck of 32 dimensions
- Decoder:
  - 4 transposed convolution layers
  - Progressively increases spatial dimensions
  - Reconstructs original image size

## Training Details

- DCGAN:
  - Learning rate: 0.0001 for both generator and discriminator
  - Batch size: 100
  - Early stopping with 10 epochs patience
  - Binary cross-entropy loss

- Autoencoder:
  - Learning rate: 0.001
  - Binary cross-entropy loss
  - 50 epochs of training

## License

[Add your license information here]

## Contributing

[Add contribution guidelines if applicable]

## Acknowledgments

- CelebA dataset for providing the training data
- TensorFlow team for the deep learning framework
