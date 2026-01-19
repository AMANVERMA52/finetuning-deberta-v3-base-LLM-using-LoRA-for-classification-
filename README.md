Project Overview
This notebook implements a fine-tuning pipeline for the DeBERTa-v3-base model to classify human preferences between two AI-generated responses. The model predicts whether Model A won, Model B won, or if there was a tie.

Data Processing
The script loads competition data from CSV files and converts one-hot labels into a single integer label for classification.

A custom PreferenceDataset class handles tokenization and input formatting, wrapping the prompt and both responses in specific tags.

The dataset class includes a swapping mechanism that doubles the training data by reversing the order of responses to prevent positional bias.

Model Configuration
The pipeline uses Parameter-Efficient Fine-Tuning (PEFT) through the Low-Rank Adaptation (LoRA) method.

LoRA targets the query_proj and value_proj modules with a rank of 8 and an alpha of 16.

Only 0.16% of the total parameters (approximately 297,219) are trainable, significantly reducing memory requirements.

Training and Inference
The model is trained for 2 epochs using the AdamW optimizer and mixed-precision training to optimize performance on GPU.

Inference is performed on the test set by applying a softmax function to the model's logits to generate class probabilities.

The final results are exported to a submission.csv file containing the probabilities for each outcome.
