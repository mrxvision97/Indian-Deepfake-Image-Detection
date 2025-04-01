# Indian Deepfake Human Face Detection

## Scope
The Indian Deepfake Human Face Detection project focuses on developing a deep learning model to distinguish real and AI-generated (deepfake) Indian human faces. The model is trained on a diverse dataset of real and synthetic images to ensure high accuracy in detecting deepfakes across various Indian contexts.

## Dataset Source
We collected a high-quality dataset of real and deepfake images. Deepfake images were generated using various models, ensuring diversity in age, gender, facial features, and environmental settings.

### Models Used to Generate Deepfake Images
- **Stable Diffusion v1.5**
- **OpenJourney**: Stable Diffusion fine-tuned model optimized for high-quality AI-generated artwork, focusing on artistic and illustrative styles.
- **Realistic Vision V3.0 VAE**: Generates synthetic images from text prompts, not designed for face swapping, facial reenactment, or video deepfakes.
- **Stable Diffusion 2.1**: Generates AI-synthesized images of diverse Indian people celebrating festivals and in professional attire.
- **Stable Diffusion 3.5**: Generates a large dataset of Indian human images.
- **Stable Diffusion 3.5 API (Hugging Face)**: Generates hyper-realistic Indian human images.

#### Generation Process
- **Model Selection**: Used the above models, fine-tuned for efficient image generation.
- **Prompt Engineering for Diversity**: Crafted prompts with varied age groups, genders, skin tones, hairstyles, attire, and professions to reflect Indian demographic diversity. Images were set in different Indian environments (rural, urban, cultural, festival settings) with photo style modifiers, lighting effects, and quality enhancements. Negative prompts removed unnatural distortions, unrealistic proportions, and poor contrast.
- **Batch-wise Image Generation**: Generated images in batches using CUDA-enabled GPUs for efficient processing.

### Real Image Collection
Real images were sourced from publicly available datasets and online repositories, including:
- [Kaggle: Indian Faces Image Classification](https://www.kaggle.com/datasets/shubh48/indian-faces-image-classification)
- [Mendeley: Dataset 1](https://data.mendeley.com/datasets/zfcd4bny82/1)
- [Kaggle: India Famous Personalities](https://www.kaggle.com/datasets/pranavchaniyara/india-famous-personalities-image-dataset?select=India_Famous_Person_Dataset.csv)
- [Kaggle: Indian People](https://www.kaggle.com/datasets/sinhayush29/indian-people)
- [Kaggle: Indian Cricketers Images](https://www.kaggle.com/datasets/omkarjc27/indian-cricketers-images)
- [Mendeley: Dataset 2](https://data.mendeley.com/datasets/nwkkd6k3gc/1)
- [FaceARG Database](https://www.cs.ubbcluj.ro/~dadi/FaceARG-database.html)
- [Kaggle: 100 Bollywood Celebrity Faces](https://www.kaggle.com/datasets/havingfun/100-bollywood-celebrity-faces)

Some datasets were accessed after requesting permission.

### Dataset Statistics
- **Total Images**: 140,000 (70,000 Real + 70,000 Deepfake)
- **Data Format**: Images
- **Diversity**: Multiple age groups, genders, skin tones, facial features, and Indian cultural backgrounds.

## Dataset Preparation and Pre-processing
- **Face Extraction**: Used MTCNN and YOLOv11 to extract faces from images.
- **Labeling**: Combined real and fake images with uniform labeling.
- **Splitting**:
  - 80% Training Set
  - 10% Validation Set
  - 10% Test Set

## Model Selection and Training

### 1. Custom Convolutional Neural Network (CNN)
#### Transformations (Training Set)
- Resizing & Scaling (224x224, Bicubic/Lanczos)
- Noise & Blurring (Gaussian noise, motion blur, JPEG compression artifacts)
- Color & Contrast Adjustments (BrightnessContrast, HueSaturation, no CLAHE)
- Geometric Transformations (Flips, Rotation, Scaling, Translation)
- CoarseDropout (Simulates occlusions)
- Normalization & Tensor Conversion

#### Transformations (Validation/Test Set)
- Normalization, Tensor Conversion, Resize only (no augmentation)

#### Training
- Four convolutional blocks with BatchNorm and Dropout.
- AdaptiveAvgPool2d layer for robustness to varying input sizes.
- Output: Single value per image, passed through sigmoid for binary classification (real vs. fake).
- Loss: Binary Cross-Entropy (BCEWithLogitsLoss).
- Optimizer: AdamW with weight decay.
- Gradient scaling (AMP) for efficiency.
- Learning rate reduction on validation loss plateau.
- Early stopping based on validation loss.

### 2. Pre-trained Model: Xception71
#### Transformations (Training Set)
- Resizing & Upscaling (Bicubic/Lanczos interpolation)
- Noise & Blurring (Gaussian noise, motion blur, compression artifacts)
- Color & Contrast Adjustments (CLAHE, BrightnessContrast, HueSaturation)
- Geometric Transformations (Horizontal/Vertical flips, rotation, scaling)
- CoarseDropout (Mimics occlusions)
- Normalization & Tensor Conversion

#### Transformations (Validation/Test Set)
- Normalization and Enhancement only (no augmentation)

#### Training
- Optimizer: AdamW with weight decay.
- Learning rate reduction via ReduceLROnPlateau.
- Automatic mixed precision (AMP) for efficiency.
- Early stopping based on validation loss.
- Saved best and final model checkpoints.

## Model Performance and Evaluation
Primary goal: Correctly predict fake images (True Negative Rate prioritized over accuracy).

| Sr. No. | Model          | Test Loss | Accuracy | Precision (Fake) | Recall (Fake) | F1 Score (Fake) | True Negative Rate |
|---------|----------------|-----------|----------|------------------|---------------|-----------------|--------------------|
| 1       | CustomCNN      | 0.0032    | 99.91%   | 0.9986           | 0.9997        | 0.9991          | 99.97%             |
| 2       | XceptionNet71  | 0.0009    | 99.96%   | 0.9991           | 1.000         | 0.9996          | 100%               |

XceptionNet71 outperformed CustomCNN in True Negative Rate and accuracy.

### Robustness Analysis
Tested on unseen images:
- Real Images: 213
- Fake Images: 4523

| Sr. No. | Model          | Test Loss | Accuracy | Precision (Fake) | Recall (Fake) | F1 Score (Fake) | True Negative Rate |
|---------|----------------|-----------|----------|------------------|---------------|-----------------|--------------------|
| 1       | CustomCNN      | 0.0279    | 99.03%   | 0.9969           | 0.9929        | 0.9949          | 99.29%             |
| 2       | XceptionNet71  | 0.1337    | 97.04%   | 0.9712           | 0.9987        | 0.9847          | 99.87%             |

XceptionNet71 excelled at identifying fake images but struggled slightly with real images. The model is tailored for Indian deepfakes and robustly detects some non-Indian deepfakes, though real image diversity (mostly actors/cricketers) is limited compared to fake image diversity.

## Model Deployment
Deployed via a web interface.

### Deploying on Netlify
- **Backend Setup**:
  - Flask/FastAPI app loads CustomCNN and Xception71 models, processes image uploads, and returns predictions.
  - Hosted on Replit (e.g., `https://your-backend.replit.app/predict`).
- **Frontend**:
  - HTML/CSS/JavaScript interface for image uploads and prediction display.
  - Hosted on Netlify (e.g., `https://your-site.netlify.app`).
- **Steps**:
  1. Push frontend to GitHub.
  2. Deploy on Netlify with static site settings.
  3. Configure CORS on backend if needed.

## Future Plan: Human-in-the-Loop Active Learning
Improve the model over time with user feedback, minimizing incorrect input impact.

1. **Semi-Supervised Feedback**: Store misclassified images with user labels in a verification queue (no instant retraining).
2. **Verification**:
   - Confidence Thresholding: Flag high-confidence disagreements for review.
   - Multiple User Consensus: Require confirmation from multiple users/reviewers.
   - Automated Cross-Checking: Validate with heuristics or external tools.
3. **Selective Retraining**: Fine-tune periodically with verified corrections, prioritizing low-confidence/high-disagreement samples.
4. **Self-Supervised Learning**: Use contrastive learning and dataset patterns (e.g., GAN artifacts) for improvement.
5. **Trust Scores**: Weight user feedback by accuracy history.

## Dependencies
- accelerate: 1.4.0
- albumentations: 2.0.5
- diffusers: 0.32.2
- facenet-pytorch: 2.6.0
- huggingface-hub: 0.29.1
- numpy: 1.26.4
- opencv-python: 4.11.0.86
- pandas: 2.2.3
- pillow: 10.2.0
- scikit-learn: 1.6.1
- scipy: 1.15.2
- timm: 1.0.15
- torch: 2.2.2+cu121
- torchvision: 0.17.2
- transformers: 4.49.0
- (Full list in original document)

## Challenges
- Collecting diverse real data.
- Slow training & convergence.
- Enhancing model generalizability.
- Detecting subtle fake manipulations.
-
- # Deepfake Detection Application

This application consists of a React frontend and FastAPI backend for detecting deepfake images.

## Project Structure

```
├── src/                  # Frontend React application
│   ├── components/       # React components
│   ├── services/        # API services
│   └── config.ts        # Configuration
├── backend/             # FastAPI backend
│   ├── app/
│   │   ├── models/      # ML models and Pydantic models
│   │   ├── routers/     # API routes
│   │   └── utils/       # Utility functions
│   ├── main.py         # FastAPI application entry
│   └── requirements.txt # Python dependencies
```

## Setup Instructions

1. Frontend Setup:
   ```bash
   # Install dependencies
   npm install

   # Start development server
   npm run dev
   ```

2. Backend Setup:
   ```bash
   # Create virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate

   # Install dependencies
   pip install -r backend/requirements.txt

   # Start backend server
   python backend/main.py
   ```

3. Add your ML models:
   - Place your CNN model in `backend/app/models/cnn_model/`
   - Place your Xception model in `backend/app/models/xception_model/`

## Integration Points

1. Frontend API Integration:
   - API configuration in `src/config.ts`
   - API services in `src/services/api.ts`

2. Backend Integration:
   - Add your model inference code in `backend/app/models/`
   - Configure routes in `backend/app/routers/`
   - Add preprocessing in `backend/app/utils/`

## Development

- Frontend runs on: http://localhost:5173
- Backend runs on: http://localhost:8000
- API documentation: http://localhost:8000/docs
