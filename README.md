# Advanced DeepFake Detection with InceptionV3 and Fourier Attention

![DeepFake Classification](https://img.shields.io/badge/Download%20Releases-blue?style=for-the-badge&logo=github&link=https://github.com/Adilali15/DeepFake-Classification/releases)

## Overview

DeepFake-Classification is a robust framework for detecting deepfake videos using a state-of-the-art InceptionV3 model enhanced with a novel Fourier Attention layer. This repository provides a comprehensive solution for synthetic video frame detection, leveraging the Fast Fourier Transform (FFT) for improved accuracy. 

## Features

- **Advanced Model Architecture**: Utilizes InceptionV3 combined with a Fourier Attention layer for enhanced feature extraction.
- **Robust Data Pipeline**: Efficiently processes large datasets, ensuring smooth data handling and manipulation.
- **Exploratory Data Analysis (EDA)**: Provides tools for in-depth analysis of data, helping users understand the characteristics of deepfake content.
- **Visualizations**: Includes various visualizations to illustrate model performance and data distributions.
- **Built with Popular Libraries**: Developed using TensorFlow, OpenCV, and other essential libraries.
- **Scalable and Interpretable**: Designed to handle large datasets while providing insights into model predictions.
- **MIT License**: Freely available for modification and distribution.

## Technologies Used

- **TensorFlow**: For building and training the deep learning model.
- **Keras**: Simplifies the creation of neural networks.
- **OpenCV**: For image and video processing.
- **Pillow**: For image manipulation.
- **Seaborn**: For statistical data visualization.

## Installation

To get started, clone the repository and install the required dependencies.

```bash
git clone https://github.com/Adilali15/DeepFake-Classification.git
cd DeepFake-Classification
pip install -r requirements.txt
```

## Usage

After installation, you can run the model with the following command:

```bash
python main.py --input <path_to_video> --output <output_path>
```

Replace `<path_to_video>` with the path to your video file and `<output_path>` with where you want to save the results.

## Data Pipeline

The data pipeline is designed to handle large volumes of video data efficiently. It includes:

1. **Data Loading**: Loads videos from specified directories.
2. **Preprocessing**: Resizes frames and normalizes pixel values.
3. **Data Augmentation**: Applies transformations to increase dataset diversity.
4. **Batching**: Organizes data into batches for model training.

## Exploratory Data Analysis (EDA)

The EDA module helps users visualize and understand the dataset. Key functionalities include:

- **Frame Distribution**: Visualizes the distribution of frames in the dataset.
- **Class Imbalance**: Analyzes class distribution to identify any imbalances.
- **Sample Visualizations**: Displays random samples of real and fake videos.

## Visualizations

Visualizations play a crucial role in understanding model performance. The repository includes:

- **Training Loss Graphs**: Shows how loss changes over epochs.
- **Accuracy Graphs**: Displays the accuracy of the model during training and validation.
- **Confusion Matrix**: Helps in understanding misclassifications.

## Model Architecture

The core of the project is the InceptionV3 model, which is enhanced with a Fourier Attention layer. This layer focuses on the most relevant features in the frequency domain, improving detection accuracy. 

### InceptionV3

InceptionV3 is a deep convolutional neural network known for its efficiency and accuracy in image classification tasks. It uses multiple filter sizes and pooling operations to capture features at various scales.

### Fourier Attention Layer

The Fourier Attention layer transforms input features into the frequency domain using FFT. This allows the model to focus on the most significant frequency components, which are crucial for distinguishing between real and fake frames.

## Training the Model

To train the model, use the following command:

```bash
python train.py --epochs <number_of_epochs>
```

Adjust `<number_of_epochs>` based on your training needs.

## Evaluation

After training, evaluate the model using:

```bash
python evaluate.py --model <model_path> --test_data <test_data_path>
```

Replace `<model_path>` with the path to your trained model and `<test_data_path>` with the path to your test dataset.

## Performance Metrics

The model's performance is evaluated using:

- **Accuracy**: The ratio of correctly predicted instances to total instances.
- **Precision**: The ratio of true positives to the sum of true and false positives.
- **Recall**: The ratio of true positives to the sum of true positives and false negatives.
- **F1 Score**: The harmonic mean of precision and recall.

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature/YourFeature`).
3. Make your changes and commit them (`git commit -m 'Add new feature'`).
4. Push to the branch (`git push origin feature/YourFeature`).
5. Open a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Thanks to the TensorFlow community for their ongoing support and contributions.
- Special thanks to the developers of OpenCV and Keras for their powerful libraries.

## Additional Resources

For further reading and resources, please visit the following links:

- [TensorFlow Documentation](https://www.tensorflow.org/)
- [OpenCV Documentation](https://opencv.org/)
- [Keras Documentation](https://keras.io/)

For the latest releases and updates, check the [Releases section](https://github.com/Adilali15/DeepFake-Classification/releases).

![DeepFake Detection](https://miro.medium.com/v2/resize:fit:1200/format:webp/1*8uFfUmj1QHn6IuYc8GZ6TA.png)

### Topics

- attention
- attention-mechanism
- cv2
- deepfakedetection
- deeplearning
- inceptionv3
- inceptionv3-model
- keras-tensorflow
- opencv
- opencv3
- pillow
- seaborn
- tensorflow

Explore the repository, contribute, and help enhance deepfake detection technologies.