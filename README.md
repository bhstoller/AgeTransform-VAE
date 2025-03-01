# Age-Conditioned Representation Learning with β-VAE

Our goal was to disentangle latent age factors in face images, allowing for controlled age manipulation. We will discuss the methods, models, data, evaluation, and timeline of our project, providing an overview of our approach and results. By controlling the Beta hyperparameter within a VAE framework, we aim to generate meaningful results and produce high-quality images

## _Dataset:_
Morph-2 Dataset.
Kurt Ricanek, and Tesfaye Tessema. (2023). MORPH-2 [Data set]. Kaggle. https://doi.org/10.34740/KAGGLE/DS/3576052

<img width="342" alt="Screenshot 2025-03-01 at 2 08 13 PM" src="https://github.com/user-attachments/assets/d555622a-5e99-46ca-aa63-179f6e4c44aa" />

## _Methodology: β-VAE Implementation_

### β-VAE Architecture

We implemented a β-VAE, an enhancement over standard VAEs, which uses a β hyperparameter to better control disentanglement. This allows for better isolation of the latent age factor while maintaining reconstruction quality. Our architecture features a CNN-based encoder for extracting meaningful features and a mirror-decoder for reconstruction.

### Loss Functions

Our training process used a combination of loss functions. We used MSE or L1 reconstruction loss to ensure high-quality image reconstruction. We also used a KL divergence loss to regularize the latent space. To further enhance age feature separation, we incorporated an age-conditioned loss function, providing additional guidance during training.

## _Model Training and Baseline Comparison_

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

