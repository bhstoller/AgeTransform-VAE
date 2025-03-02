# Age-Conditioned Representation Learning with β-VAE
<p align="justify">
Our goal was to disentangle latent age factors in face images, allowing for controlled age manipulation. We will discuss the methods, models, data, evaluation, and timeline of our project, providing an overview of our approach and results. By controlling the Beta hyperparameter within a VAE framework, we aim to generate meaningful results and produce high-quality images.
</p>

![image](https://github.com/user-attachments/assets/29226943-95e9-4ce7-8449-c5c19ad1604d)

## Dataset:
Morph-2 Dataset.
Kurt Ricanek, and Tesfaye Tessema. (2023). MORPH-2 [Data set]. Kaggle. https://doi.org/10.34740/KAGGLE/DS/3576052

<img width="342" alt="Screenshot 2025-03-01 at 2 08 13 PM" src="https://github.com/user-attachments/assets/d555622a-5e99-46ca-aa63-179f6e4c44aa" />

## Methodology: β-VAE Implementation

### β-VAE Architecture

We implemented a β-VAE, an enhancement over standard VAEs, which uses a β hyperparameter to better control disentanglement. This allows for better isolation of the latent age factor while maintaining reconstruction quality. Our architecture features a CNN-based encoder for extracting meaningful features and a mirror-decoder for reconstruction.

### Loss Functions

Our training process used a combination of loss functions. We used MSE or L1 reconstruction loss to ensure high-quality image reconstruction. We also used a KL divergence loss to regularize the latent space. To further enhance age feature separation, we incorporated an age-conditioned loss function, providing additional guidance during training.

## Model Training and Baseline Comparison

### 1. Training from Scratch

We chose to train the β-VAE from scratch on a local GPU, allowing us full control over the learning process and parameters. Rather than relying on pre-trained models, this approach enabled us to customize the architecture and training regime to our specific goals of age disentanglement.

### 2. Architecture Inspiration

While training from scratch, we referenced architectures such as ResNet-like CNNs to inform our design for efficient feature extraction. These architectures provided a solid foundation for building an effective encoder and decoder.

### 3. Baseline Comparison

To benchmark our progress, we planned to compare our results against a pre-trained VAE model sourced from repositories like TensorFlow Model Garden or PapersWithCode. This comparison was intended to provide a quantitative measure of the improvements achieved through our β-VAE implementation and age-conditioned training approach.

## _Data Preparation and Preprocessing_

### 1. IMDB-Wiki Dataset

We utilized the IMDB-Wiki dataset, consisting of around 500,000 labeled face images from IMDb and Wikipedia. This dataset provided a rich source of data for training our model, including metadata such as age, gender, and timestamps.

### 2. Addressing Data Imbalance

Recognizing the dataset’s inherent imbalances (fewer samples for older individuals), we implemented several preprocessing steps to improve training quality and ensure fair representation across age ranges.

### 3. Preprocessing Steps

Our preprocessing steps included filtering low-quality images, correcting face crops, resizing all images to a consistent resolution (128x128 or 256x256 pixels), normalizing pixel values to a range between [-1,1] or [0,1], and encoding age as a normalized numerical condition [0,1].

## Evaluation Metrics and Performance Assessment

### Reconstruction Accuracy

We assessed the model’s performance using reconstruction accuracy, measured by MSE (Mean Squared Error) or L1 loss. This metric provided insight into how well the model could reconstruct input images after encoding and decoding.

### Latent Space Disentanglement

To quantify the disentanglement of the latent space, we employed metrics such as Mutual Information Gap (MIG). This metric helped us understand the extent to which individual latent variables captured distinct factors of variation (e.g., age).

### Age Shift Vector Application

We applied age shift vectors to test the model’s interpolation capabilities. By manipulating the latent representation, we evaluated how smoothly and realistically the model could transition between different ages.

![image](https://github.com/user-attachments/assets/2986904c-74b3-4986-b9ae-f366cd0e84ac)


## Key Challenges and Solutions

### Dataset Imbalance

The IMDB-Wiki dataset had a notable age imbalance, with fewer samples for older individuals, potentially skewing the model’s learning. To mitigate this, we employed data augmentation techniques and weighted loss functions to ensure fairer representation across all age ranges.

### Disentanglement Complexity

Achieving effective disentanglement of age from other facial features proved challenging. Tuning the β hyperparameter in the β-VAE was crucial. We conducted extensive experiments to find an optimal value that balanced reconstruction quality and disentanglement performance.

## Results: Age Transformation Simulation

### Aging Simulation

Our age transformation simulation demonstrated the model’s ability to realistically age faces. By manipulating the age shift vector in latent space, we generated plausible aging effects while preserving the individual’s identity.

<img width="307" alt="Screenshot 2025-03-01 at 2 22 37 PM" src="https://github.com/user-attachments/assets/831e1618-5d44-43aa-a915-ccb618afefc6" />

## Conclusion and Future Directions

## Summary of Work

We successfully implemented an age-conditioned β-VAE for disentangling latent age factors in face images. We addressed data imbalance, optimized the model architecture, and achieved realistic age transformation simulations.

<img width="302" alt="Screenshot 2025-03-01 at 2 25 31 PM" src="https://github.com/user-attachments/assets/22f37dc5-6f7f-46ac-bbab-bf7a8f2b008a" />

## Key Findings

Our work highlights the effectiveness of β-VAEs for controlled attribute manipulation in facial images. The age-conditioned loss function and careful hyperparameter tuning were crucial for achieving high-quality results.

## Future Work

Future directions include exploring more advanced disentanglement techniques, incorporating additional facial attributes, and applying the model to other image datasets.
