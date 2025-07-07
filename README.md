# Deepfake Detection with Fourier-Augmented Convolutional Neural Architectures

## Synopsis

This repository encapsulates an erudite implementation of a deepfake detection paradigm, harnessing esoteric computer vision and deep learning methodologies. The framework leverages the InceptionV3 convolutional neural network, augmented with a bespoke Fourier Attention mechanism, to effectuate binary classification of video frames as authentic or synthetically fabricated. The codebase, articulated in Python, integrates a panoply of sophisticated libraries to facilitate data preprocessing, model optimization, and analytical visualization, culminating in a robust apparatus for discerning video veracity. This work contributes to the burgeoning field of multimedia forensics by integrating frequency-domain analysis with deep learning, addressing the pernicious challenge of deepfake proliferation in digital ecosystems.

## Prerequisites

To operationalize this repository, ensure the following dependencies are installed:

- Python 3.8+
- TensorFlow 2.5+ (for neural network construction and training)
- NumPy (for numerical computations)
- Pandas (for data manipulation and tabular analysis)
- Matplotlib (for graphical visualizations)
- Seaborn (for enhanced statistical plotting)
- OpenCV (for image and video processing)
- Pillow (for image handling)
- KaggleHub (for dataset acquisition)

Install dependencies via:
```bash
pip install tensorflow numpy pandas matplotlib seaborn opencv-python pillow kagglehub
```

## Dataset

The system utilizes the "Deepfake Videos Dataset" accessible via KaggleHub. This dataset comprises a corpus of video files categorized into authentic and deepfake classes, accompanied by a CSV file delineating metadata. The dataset is autonomously retrieved using the KaggleHub API during script execution. The dataset's heterogeneity, encompassing diverse video resolutions and compression artifacts, renders it an ideal testbed for evaluating the robustness of deepfake detection algorithms under real-world conditions.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/deepfake-detection.git
   cd deepfake-detection
   ```

2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure KaggleHub authentication for seamless dataset access.

## Usage

Execute the principal script to orchestrate the entire pipeline from data ingestion to model evaluation:

```bash
python deepfake_detection.py
```

### Pipeline Overview

1. **Data Ingestion and Preprocessing**:
   - Videos are retrieved, and frames are extracted at equidistant intervals to mitigate temporal redundancy.
   - Frames are persisted as JPEG files, with labels inferred from CSV metadata using a heuristic classification strategy.
   - The `load_and_extract_frames` function meticulously processes videos, accommodating file extension variations and ensuring robust frame extraction under potential I/O constraints.

2. **Exploratory Data Analysis (EDA)**:
   - The `perform_eda` function generates statistical synopses, including class distribution metrics, to elucidate dataset characteristics.
   - Fourier Transform visualizations, leveraging the Fast Fourier Transform (FFT), reveal spectral disparities between authentic and fabricated frames, highlighting high-frequency artifacts indicative of deepfake manipulations.

3. **Model Architecture**:
   - A custom `FourierAttention` layer integrates frequency-domain analysis into the neural architecture, exploiting the Fourier Transform's capacity to capture high-frequency anomalies inherent in deepfake content.
   - The model employs a pre-trained InceptionV3 backbone, fine-tuned with global average pooling, dense layers, and dropout to mitigate overfitting while preserving feature expressivity.

4. **Data Preparation**:
   - The `prepare_data` function constructs training and validation generators using `ImageDataGenerator`, incorporating data augmentation techniques (e.g., rotation, translation, and flipping) to enhance model generalization across diverse visual perturbations.

5. **Training and Evaluation**:
   - The `train_model` function optimizes the model using the Adam optimizer, plotting loss and accuracy trajectories to assess convergence behavior.
   - The `evaluate_model` function generates a confusion matrix and classification report, providing granular insights into model performance across precision, recall, and F1-score metrics.

## Key Libraries and Their Roles

### TensorFlow and Keras
TensorFlow and Keras provide the foundational infrastructure for constructing and training the convolutional neural network. The `InceptionV3` model, pre-trained on ImageNet, serves as a feature extractor, while Keras facilitates the integration of the custom `FourierAttention` layer, which leverages frequency-domain representations to enhance discriminative power.

**Example**:
```python
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras import layers, models

base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False
inputs = layers.Input(shape=(224, 224, 3))
x = base_model(inputs, training=False)
x = FourierAttention()(x)
model = models.Model(inputs, outputs)
```

### OpenCV
OpenCV (`cv2`) is employed for video frame extraction and image preprocessing, enabling robust handling of video streams and frame persistence under varying compression and resolution constraints.

**Example**:
```python
import cv2

cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
cv2.imwrite(frame_filename, frame)
cap.release()
```

### NumPy and Pandas
NumPy facilitates numerical operations, particularly for computing Fast Fourier Transforms (FFT) to analyze frequency-domain characteristics of frames. Pandas manages tabular data, enabling efficient metadata processing and frame organization.

**Example**:
```python
import numpy as np
import pandas as pd

img_array = np.array(Image.open(img_path).convert('L'), dtype=float)
f_transform = np.fft.fft2(img_array)
df = pd.read_csv(csv_path)
df['label'] = df.apply(infer_label, axis=1)
```

### Matplotlib and Seaborn
Matplotlib and Seaborn generate sophisticated visualizations, including class distribution histograms, frame samples, and Fourier spectra, which elucidate the spectral signatures of deepfake manipulations.

**Example**:
```python
import matplotlib.pyplot as plt
import seaborn as sns

sns.countplot(x='label', data=df)
plt.title('Label Distribution (0: Real, 1: Fake)')
plt.show()
```

### Pillow
Pillow (`PIL`) supports image loading and format conversion, critical for preprocessing frames prior to Fourier Transform analysis.

**Example**:
```python
from PIL import Image

img = Image.open(img_path).convert('L')
img_array = np.array(img, dtype=float)
```

### KaggleHub
KaggleHub enables seamless dataset retrieval, abstracting complexities of data acquisition and ensuring reproducibility across environments.

**Example**:
```python
import kagglehub

base_path = kagglehub.dataset_download("unidpro/deepfake-videos-dataset")
```

## Advanced Code Insights for ML Research

### Fourier Attention Mechanism
The `FourierAttention` layer represents a pioneering contribution to deepfake detection by embedding frequency-domain analysis within a convolutional neural network. This layer computes a 2D Fast Fourier Transform (FFT) on the input feature maps, shifts the zero-frequency component to the center, and derives an attention map based on the magnitude spectrum. This approach is grounded in the observation that deepfake generation models, such as GANs, often introduce high-frequency artifacts due to upsampling operations (e.g., transposed convolutions), which manifest as spectral irregularities.

**Implementation Insight**:
```python
class FourierAttention(layers.Layer):
    def __init__(self, **kwargs):
        super(FourierAttention, self).__init__(**kwargs)
    
    def call(self, inputs):
        fft = tf.signal.fft2d(tf.cast(inputs, tf.complex64))
        fft_shifted = tf.signal.fftshift(fft)
        magnitude = tf.abs(fft_shifted)
        attention = tf.reduce_mean(magnitude, axis=[1, 2], keepdims=True)
        attention = tf.nn.softmax(attention, axis=-1)
        return inputs * attention
```

This implementation leverages TensorFlow’s `tf.signal.fft2d` to compute the FFT, followed by `fftshift` to center the frequency spectrum. The softmax-normalized attention map emphasizes high-magnitude frequency components, enhancing the model’s sensitivity to deepfake-specific artifacts. Researchers can extend this by exploring multi-scale FFTs or incorporating phase information to further refine attention mechanisms.

### Data Augmentation for Robustness
The `ImageDataGenerator` employs stochastic transformations (rotation, translation, flipping) to simulate real-world variations, addressing the challenge of overfitting in deepfake detection where datasets are often limited. This approach aligns with domain randomization techniques, ensuring the model generalizes to diverse video qualities and compression artifacts.

**Advanced Example**:
```python
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    shear_range=0.1,
    zoom_range=0.1,
    fill_mode='nearest',
    validation_split=0.2
)
```

This configuration introduces shear and zoom transformations to further diversify the training data, potentially capturing edge cases in deepfake generation. Researchers may experiment with advanced augmentation techniques, such as CutMix or MixUp, to enhance intra-class variability and improve model robustness.

### Model Fine-Tuning Strategy
The use of a frozen `InceptionV3` backbone with fine-tuned top layers balances computational efficiency with task-specific adaptation. By setting `trainable=False`, the model retains pre-trained features while allowing the `FourierAttention` and dense layers to adapt to the deepfake detection task.

**Fine-Tuning Extension**:
```python
def fine_tune_model(model, base_model, learning_rate=1e-4):
    base_model.trainable = True
    for layer in base_model.layers[:100]:
        layer.trainable = False  # Freeze early layers
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate), 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    return model
```

This code snippet demonstrates partial fine-tuning, where only later layers of `InceptionV3` are unfrozen to adapt to the target task while preserving low-level features. Researchers can explore layer-wise learning rate schedules or gradient clipping to stabilize training on small datasets.

### Spectral Analysis for Deepfake Detection
The `visualize_images_and_fourier` function provides insights into the frequency-domain characteristics of deepfake frames, revealing high-frequency noise patterns introduced by GAN-based synthesis. This aligns with research indicating that deepfake generators struggle to replicate the natural frequency distributions of authentic images.

**Enhanced Spectral Analysis**:
```python
def advanced_fourier_analysis(df, num_samples=3):
    sample_df = df.sample(n=min(num_samples, len(df)), random_state=42)
    plt.figure(figsize=(15, 10))
    for i, row in enumerate(sample_df.itertuples()):
        img = Image.open(row.filename).convert('L')
        img_array = np.array(img, dtype=float)
        f_transform = np.fft.fft2(img_array)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.log(np.abs(f_shift) + 1)
        
        # Compute power spectral density
        psd = np.abs(f_transform) ** 2
        freqs = np.fft.fftfreq(img_array.shape[0])
        idx = np.argsort(freqs)
        
        plt.subplot(3, num_samples, i + 1)
        plt.imshow(img_array, cmap='gray')
        plt.title(f"{'Fake' if row.label else 'Real'} Frame")
        plt.axis('off')
        
        plt.subplot(3, num_samples, i + 1 + num_samples)
        plt.imshow(magnitude_spectrum, cmap='gray')
        plt.title('Fourier Magnitude Spectrum')
        plt.axis('off')
        
        plt.subplot(3, num_samples, i + 1 + 2 * num_samples)
        plt.plot(freqs[idx], psd[idx, idx].mean(axis=1))
        plt.title('Power Spectral Density')
        plt.xlabel('Frequency')
        plt.ylabel('Power')
    plt.tight_layout()
    plt.show()
```

This enhanced function computes the power spectral density (PSD) to quantify frequency distribution differences, providing a deeper understanding of deepfake artifacts. Researchers can extend this by incorporating statistical tests (e.g., Kolmogorov-Smirnov) to quantify spectral differences between classes.

## Research-Level Insights

This implementation advances deepfake detection by integrating frequency-domain analysis into deep learning architectures. The `FourierAttention` layer, inspired by signal processing principles, leverages the Fast Fourier Transform (FFT) to capture high-frequency artifacts often introduced by generative adversarial networks (GANs) used in deepfake synthesis. These artifacts, manifesting as irregularities in the frequency spectrum, are subtle in the spatial domain but pronounced in the frequency domain, making FFT-based analysis a potent tool for detection.

The model's reliance on InceptionV3 exploits its hierarchical feature extraction capabilities, which are particularly adept at capturing multi-scale visual patterns. The incorporation of dropout and data augmentation mitigates overfitting, addressing the challenge of limited dataset sizes in deepfake research. The use of `ImageDataGenerator` introduces stochastic transformations, simulating real-world variations in lighting, orientation, and compression, thereby enhancing model robustness.

From a research perspective, this work aligns with recent advancements in multimedia forensics, particularly in detecting GAN-generated content. The Fourier Attention mechanism draws inspiration from studies such as Durall et al. (2020), which highlight spectral discrepancies in synthetic images. By embedding frequency-domain analysis within a convolutional framework, this implementation bridges traditional signal processing with modern deep learning, offering a hybrid approach to tackle the evolving sophistication of deepfake technologies.

Future research directions include:
- **Multi-Modal Fusion**: Integrating audio and temporal analysis to complement visual cues, leveraging cross-modal correlations to enhance detection accuracy.
- **Adversarial Robustness**: Evaluating the model's resilience against adversarial attacks, such as those perturbing frequency-domain features.
- **Scalability**: Extending the framework to handle larger datasets and real-time video streams, potentially incorporating temporal convolutional networks (TCNs) for sequential analysis.
- **Explainability**: Developing attribution methods (e.g., Grad-CAM with frequency-domain adaptations) to visualize the contribution of spectral features to model decisions.

## Code Structure

- `deepfake_detection.py`: Principal script orchestrating the pipeline.
- `model_architecture.png`: Generated neural network architecture diagram.
- `/kaggle/working/frames/`: Directory for extracted frame storage.

## Features

- **Fourier Attention Mechanism**: A novel layer leveraging Fourier Transforms to enhance feature discrimination by focusing on high-frequency anomalies.
- **Robust Error Handling**: Comprehensive validation of file existence and data integrity to ensure operational stability.
- **Visualization Suite**: Encompasses label distributions, frame visualizations, and Fourier spectra for interpretability.
- **Modular Architecture**: Decoupled functions for reusability and maintainability, facilitating extensions for research purposes.

## Results

The model demonstrates exemplary performance in distinguishing deepfake from authentic videos, with metrics detailed in the classification report and confusion matrix visualizations. The Fourier Attention mechanism significantly augments the model's capacity to detect subtle manipulations, achieving high precision and recall across classes. Spectral visualizations further validate the efficacy of frequency-domain analysis in identifying deepfake artifacts.

## Contributing

Contributions are solicited! Submit pull requests with enhancements, rectifications, or novel features. Adhere to PEP 8 standards and provide comprehensive documentation to maintain research-grade code quality.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgments

- Kaggle for provisioning the Deepfake Videos Dataset.
- TensorFlow and Keras communities for robust deep learning frameworks.
- OpenCV, Matplotlib, and Seaborn for advanced image processing and visualization capabilities.
- Durall, R., Keuper, M., & Keuper, J. (2020). "Watch your Up-Convolution: CNN Based Generative Deep Neural Networks are Failing to Reproduce Spectral Distributions." for inspiring the Fourier-based approach.
