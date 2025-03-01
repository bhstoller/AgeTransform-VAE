**Age-Conditioned Representation Learning with β-VAE**

Our goal was to disentangle latent age factors in face images, allowing for controlled age manipulation. We will discuss the methods, models, data, evaluation, and timeline of our project, providing an overview of our approach and results. By controlling the Beta hyperparameter within a VAE framework, we aim to generate meaningful results and produce high-quality images

**Dataset: **
Morph-2 Dataset.
Kurt Ricanek, and Tesfaye Tessema. (2023). MORPH-2 [Data set]. Kaggle. https://doi.org/10.34740/KAGGLE/DS/3576052

<img width="342" alt="Screenshot 2025-03-01 at 2 08 13 PM" src="https://github.com/user-attachments/assets/d555622a-5e99-46ca-aa63-179f6e4c44aa" />

** Methodology: β-VAE Implementation **

** β-VAE Architecture **

We implemented a β-VAE, an enhancement over standard VAEs, which uses a β hyperparameter to better control disentanglement. This allows for better isolation of the latent age factor while maintaining reconstruction quality. Our architecture features a CNN-based encoder for extracting meaningful features and a mirror-decoder for reconstruction.

** Loss Functions **

Our training process used a combination of loss functions. We used MSE or L1 reconstruction loss to ensure high-quality image reconstruction. We also used a KL divergence loss to regularize the latent space. To further enhance age feature separation, we incorporated an age-conditioned loss function, providing additional guidance during training.
