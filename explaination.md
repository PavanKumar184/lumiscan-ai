# A Vision-Language Assisted DR-Ultra Framework for Robust Thoracic Disease Detection from Imbalanced Chest Radiographs

## 1. Project Overview

This project is titled **"A Vision-Language Assisted DR-Ultra Framework for Robust Thoracic Disease Detection from Imbalanced Chest Radiographs."**

The project focuses on building an AI-assisted chest X-ray analysis system that can help identify thoracic diseases from radiographic images. The system uses a trained deep learning model to analyze uploaded chest X-ray images and presents the prediction through a user-friendly web application called **LumiScan AI**.

The main goal of the project is to support radiology-style screening by combining:

- **Vision intelligence**: deep learning-based image analysis of chest X-rays.
- **Language assistance**: disease explanations, diagnostic summaries, class descriptions, and report-style output that help users understand the model result.
- **Imbalance-aware learning**: training strategies that reduce the impact of highly unequal class distribution in medical datasets.
- **Explainability**: Grad-CAM heatmaps that visually highlight image regions that influenced the model prediction.

The project is not intended to replace doctors or radiologists. It is designed as a **clinical decision-support prototype** and educational/research demonstration.

## 2. Problem Statement

Chest X-rays are among the most commonly used medical imaging techniques for detecting lung and thoracic abnormalities. They are used to identify conditions such as:

- Cardiomegaly
- Pneumonia
- Effusion
- Pneumothorax
- Atelectasis
- Mass
- Nodule
- Fibrosis
- Edema
- Emphysema
- Consolidation
- Pleural Thickening
- Hernia
- Infiltration

However, manual interpretation of chest X-rays can be challenging because:

1. Some diseases have subtle visual patterns.
2. Multiple thoracic diseases may look similar.
3. Expert radiologists may not always be available.
4. Large-scale screening requires time and consistency.
5. Medical datasets are usually imbalanced, meaning some diseases have many samples while rare diseases have very few samples.

The imbalance problem is very important. For example, in this project's class distribution, common classes such as **Infiltration** and **Effusion** have many more positive samples, while rare classes such as **Hernia** and **Pneumonia** have far fewer positive samples. A normal deep learning model may become biased toward common classes and perform poorly on rare but clinically important diseases.

Therefore, this project proposes an AI framework that is not only image-based but also **imbalance-aware, explainable, and report-oriented**.

## 3. Meaning Of The Project Title

### 3.1 Vision

The "Vision" part refers to the model's ability to process chest X-ray images. The system uses a deep convolutional neural network backbone, currently **DenseNet121**, to extract visual features from radiographs.

The input X-ray image is resized, normalized, and passed through the trained model. The model then predicts the most likely thoracic disease class.

### 3.2 Language Assisted

The "Language Assisted" part refers to the way the system supports the image prediction with text-based clinical explanation and report generation.

In the current implementation, the language assistance includes:

- Disease-specific explanation text.
- A diagnostic report page.
- Class descriptions and causes.
- Text-to-speech support in the interface.
- Human-readable prediction summary.
- Dataset class distribution explanation.

This helps the user understand not only **what** the model predicted, but also **what the condition means** and how it relates to the training data.

In a future extension, this framework can be expanded into a full vision-language model by adding:

- A text encoder for patient symptoms, clinical notes, or radiology reports.
- A fusion module that combines image features and text features.
- A final classifier that uses both visual and textual context.

So, the present system is a practical **vision-first DR-Ultra stage** with language-assisted reporting, and the architecture is designed so that a full vision-language fusion module can be added later.

### 3.3 DR-Ultra Framework

DR-Ultra is the proposed framework name used in this project. It represents a robust disease recognition pipeline for chest radiographs.

In this implementation, DR-Ultra contains:

- Image preprocessing
- DenseNet121 visual backbone
- Dropout-based classifier head
- Multi-class thoracic disease output
- Imbalance-aware class weighting during training
- Checkpoint-based inference
- Grad-CAM explainability
- Flask-based diagnostic user interface

DenseNet121 is not treated as the complete project by itself. Instead, DenseNet121 acts as the **visual encoder backbone** inside the larger DR-Ultra framework.

### 3.4 Robust Thoracic Disease Detection

"Robust" means the project attempts to make prediction more reliable by:

- Training on a large chest X-ray dataset.
- Handling class imbalance with weighted loss.
- Saving model metadata and class labels.
- Showing confidence values.
- Using Grad-CAM to provide visual explanation.
- Displaying class distribution to make users aware of data imbalance.

### 3.5 Imbalanced Chest Radiographs

Medical image datasets are rarely balanced. Some diseases are common, while others are rare. In the NIH ChestX-ray14 dataset used by this project, class counts differ greatly.

For example:

| Disease Class | Positive Count | Imbalance Nature |
|---|---:|---|
| Infiltration | 15986 | Highly represented |
| Effusion | 11383 | Highly represented |
| Atelectasis | 10153 | Highly represented |
| Cardiomegaly | 2776 | Less represented |
| Pneumonia | 1431 | Rare |
| Hernia | 227 | Extremely rare |

This imbalance affects model learning. Without special handling, the model may focus more on frequent diseases and ignore rare diseases. Therefore, this project uses class weighting during training to give more importance to underrepresented classes.

## 4. Dataset Used

The project uses the **NIH ChestX-ray14 dataset**, available on Kaggle:

```text
https://www.kaggle.com/datasets/nih-chest-xrays/data
```

This dataset contains chest X-ray images with disease labels. The model is trained to identify 14 thoracic disease classes:

1. Atelectasis
2. Cardiomegaly
3. Effusion
4. Infiltration
5. Mass
6. Nodule
7. Pneumonia
8. Pneumothorax
9. Consolidation
10. Edema
11. Emphysema
12. Fibrosis
13. Pleural Thickening
14. Hernia

The dataset labels are converted into machine-readable class vectors during preprocessing. The project also creates train, validation, and test metadata files.

## 5. Data Preprocessing Pipeline

The preprocessing pipeline prepares raw X-ray data for deep learning.

The main preprocessing steps are:

1. Load metadata from the NIH dataset.
2. Map each image name to its actual file path.
3. Clean missing or invalid image entries.
4. Convert disease labels into class vectors.
5. Split the dataset into training, validation, and test sets.
6. Compute class distribution and class weights.
7. Apply image transformations before training/inference.

During inference in the Flask app, each uploaded image is processed using:

```python
transforms.Resize((224, 224))
transforms.Grayscale(num_output_channels=3)
transforms.ToTensor()
transforms.Normalize([0.485, 0.456, 0.406],
                     [0.229, 0.224, 0.225])
```

The image is resized to `224 x 224`, converted into a 3-channel grayscale image, transformed into a tensor, and normalized using ImageNet-style normalization values.

## 6. Model Architecture

The current implementation uses a **DR-Ultra image branch** based on **DenseNet121**.

The model architecture in the application is:

```text
Input Chest X-ray Image
        |
Image Preprocessing
        |
DenseNet121 Visual Backbone
        |
Dropout Layer
        |
Fully Connected Classifier
        |
Disease Prediction
```

In code, the classifier head is modified as:

```python
self.model.classifier = nn.Sequential(
    nn.Dropout(p=0.3),
    nn.Linear(self.model.classifier.in_features, num_classes),
)
```

The model outputs logits for the disease classes. A sigmoid function is applied to convert logits into probabilities:

```python
probabilities = torch.sigmoid(logits)
```

The application then selects the highest-confidence class and compares it with the stored prediction threshold. If the confidence is below the threshold, the result is shown as:

```text
Normal / No abnormality detected
```

## 7. Why DenseNet121 Is Used

DenseNet121 is a strong convolutional neural network architecture commonly used in medical image classification tasks.

It is useful because:

- It reuses features through dense connections.
- It helps gradient flow during training.
- It can learn detailed image patterns.
- It works well for radiographic images.
- It is computationally practical compared with very large models.

In this project, DenseNet121 is used as the **visual feature extractor** inside the DR-Ultra framework.

## 8. Class Imbalance Handling

The project handles dataset imbalance using class weights.

The file `class_weights.csv` stores information such as:

- Positive count
- Negative count
- Frequency
- Inverse frequency weight
- BCE positive weight

Rare classes are assigned higher weights so that the model pays more attention to them during training.

For example:

- Hernia has only 227 positive samples and receives a very high positive weight.
- Pneumonia has 1431 positive samples and also receives a high weight.
- Common classes such as Infiltration and Effusion receive lower weights.

This helps reduce bias toward majority classes.

## 9. Training Strategy

The training process is implemented in `kaggle_train_nih_dr_ultra.py`.

The training script:

1. Detects the Kaggle NIH dataset path.
2. Loads and cleans metadata.
3. Builds an image index.
4. Encodes disease labels.
5. Splits data into train, validation, and test sets.
6. Computes class weights.
7. Builds the DR-Ultra image branch.
8. Trains the model using an imbalance-aware loss function.
9. Saves the best checkpoint.
10. Exports supporting files such as class labels and metrics.

Important training configuration:

```python
BATCH_SIZE = 32
EPOCHS = 8
LEARNING_RATE = 1e-4
VISUAL_BACKBONE = "densenet121"
THRESHOLD = 0.5
```

The final trained checkpoint is saved as:

```text
best_dr_ultra_densenet121_nih.pth
```

## 10. Model Checkpoint

The checkpoint file stores:

- DR-Ultra framework name
- Selected visual backbone
- Model architecture name
- Trained model weights
- Class names
- Image size
- Prediction threshold
- Class imbalance weights
- Best validation metrics

The Flask application loads this checkpoint during startup.

If the checkpoint is missing, the app will not start, because inference depends on the trained weights.

## 11. Evaluation Metrics

The checkpoint contains validation metrics. The current stored metrics include:

| Metric | Value |
|---|---:|
| Macro AUC | 0.8419 |
| Macro F1 | 0.0604 |
| Macro Precision | 0.0392 |
| Macro Recall | 0.3548 |
| Micro F1 | 0.0902 |
| Loss | 0.4548 |

### How To Explain These Metrics

**Macro AUC** is reasonably strong, meaning the model has learned useful ranking ability between positive and negative examples across classes.

However, **F1 and precision are low**, which shows that the model can still produce false positives, especially because the dataset is highly imbalanced and some classes are rare.

This is common in medical multi-label datasets. The model can identify patterns, but threshold tuning and per-class calibration are needed for stronger clinical reliability.

Therefore, the project should be presented as a **research prototype and decision-support system**, not a final diagnostic product.

## 12. Inference Workflow In The Web Application

The web application is built using Flask.

The inference workflow is:

1. User opens the LumiScan AI web interface.
2. User uploads a chest X-ray image.
3. Flask receives the image through the `/predict` route.
4. The image is saved to the upload folder.
5. The image is preprocessed using the same transformation used during training.
6. The trained DR-Ultra model predicts disease probabilities.
7. The top predicted disease is selected.
8. A Grad-CAM heatmap is generated for explainability.
9. A disease explanation is selected from the explanation dictionary.
10. Dataset imbalance information is prepared.
11. The result page displays the final report.

The report includes:

- Scan ID
- Original uploaded X-ray
- Grad-CAM heatmap
- Predicted disease
- Confidence score
- Disease explanation
- Dataset class distribution
- Medical disclaimer

## 13. Grad-CAM Explainability

Grad-CAM stands for **Gradient-weighted Class Activation Mapping**.

It is used to show which part of the image influenced the model's prediction.

In this project:

1. The final dense block of DenseNet121 is selected as the target layer.
2. A forward hook captures feature activations.
3. A backward hook captures gradients.
4. Gradients are averaged to produce class importance weights.
5. A heatmap is generated.
6. The heatmap is overlaid on the original X-ray.

This helps users understand whether the model is focusing on meaningful regions of the image.

Grad-CAM improves trust and transparency because the model is no longer a complete black box.

## 14. Web Application Design

The web application is named **LumiScan AI**.

It contains:

- Dashboard page
- Upload interface
- Disease classes page
- Diagnostic report page
- Quick links for learning resources
- Voice/text status support
- Responsive UI for desktop and mobile

The design follows a clinical workstation style so that the interface feels suitable for a medical AI tool.

## 15. Current Scope And Limitation

The current implemented system is a **vision-first DR-Ultra framework with language-assisted reporting**.

It currently uses:

- Chest X-ray image input
- DenseNet121 visual backbone
- Class imbalance-aware training
- Single top-label display
- Disease explanation text
- Grad-CAM visual explanation

The full future version can include:

- Patient symptoms
- Clinical notes
- Radiology report text
- Text encoder
- Image-text fusion module
- Multi-modal prediction

### Important Limitation

The current app displays the highest-confidence prediction rather than a fully calibrated clinical diagnosis. Since the model metrics show low precision/F1, the output should always be interpreted as AI assistance, not medical confirmation.

## 16. Why The Project Is Useful

This project is useful because it demonstrates how AI can assist chest X-ray interpretation by:

- Reducing manual screening effort.
- Giving quick preliminary predictions.
- Highlighting suspicious image regions.
- Explaining disease meaning in simple clinical language.
- Showing class imbalance awareness.
- Providing a complete web-based workflow.

It is especially useful for academic demonstration because it includes both model training and deployable inference.

## 17. Novelty Of The Project

The important contributions of this project are:

1. **DR-Ultra framework design** for thoracic disease detection.
2. **Imbalance-aware learning** using class weights.
3. **DenseNet121 visual encoder** integrated into a medical AI pipeline.
4. **Grad-CAM explainability** for model transparency.
5. **Language-assisted report generation** through disease explanations and clinical summaries.
6. **Interactive Flask web application** for end-to-end demonstration.
7. **Future-ready vision-language architecture** that can later include text encoders and fusion modules.

## 18. How To Explain The Project To A Panel

You can explain the project like this:

> Our project is an AI-based chest X-ray analysis system called LumiScan AI. The full project title is "A Vision-Language Assisted DR-Ultra Framework for Robust Thoracic Disease Detection from Imbalanced Chest Radiographs." The aim is to detect thoracic diseases from chest X-ray images while handling the challenge of class imbalance in medical datasets.
>
> We used the NIH ChestX-ray14 dataset, which contains multiple thoracic disease labels. Since some diseases have many samples and some rare diseases have very few samples, we used imbalance-aware class weighting during training. This helps the model give more importance to rare disease classes.
>
> The core model is the DR-Ultra image branch. It uses DenseNet121 as the visual backbone. DenseNet121 extracts deep visual features from the X-ray image, and a modified classifier head predicts disease probabilities. The trained model checkpoint is loaded into a Flask web application for real-time inference.
>
> When a user uploads a chest X-ray, the image is resized, normalized, and passed through the trained model. The system then displays the predicted disease, confidence value, disease explanation, and dataset distribution details. We also generate a Grad-CAM heatmap to show which region of the X-ray influenced the prediction.
>
> The language-assisted part of our current system is implemented through clinical explanations, disease descriptions, report-style output, and voice-supported status messages. This makes the model result easier for users to understand. In future work, we can extend the same framework by adding a text encoder for symptoms or radiology notes and fuse it with image features.
>
> This project is not a replacement for doctors. It is a decision-support prototype that demonstrates how AI can assist radiology screening with explainable and imbalance-aware deep learning.

## 19. Questions A Panel May Ask

### Q1. Why did you choose chest X-rays?

Chest X-rays are widely used, low-cost, and important for diagnosing thoracic diseases. They are also suitable for AI-based screening because large public datasets are available.

### Q2. Why did you use DenseNet121?

DenseNet121 is effective for medical imaging because its dense connections reuse features and improve gradient flow. It can capture subtle radiographic patterns while remaining practical to train and deploy.

### Q3. What is class imbalance?

Class imbalance means some disease classes have many samples while others have very few. This can make a model biased toward common classes. In this project, class weights are used to reduce that problem.

### Q4. What is Grad-CAM?

Grad-CAM is an explainability technique that highlights the image regions that influenced the model prediction. It helps users understand where the model focused.

### Q5. Is this a multi-label model?

The model internally outputs probabilities for 14 disease classes using sigmoid activation. The current deployed app displays the highest-confidence disease for clarity. A multi-label report can be added later, but it requires careful threshold tuning to avoid false positives.

### Q6. What does vision-language assisted mean in this project?

The current project uses vision-based X-ray prediction and language-assisted reporting through explanations, summaries, class descriptions, and voice guidance. The framework is designed so that future versions can include a text encoder and image-text fusion for full vision-language learning.

### Q7. Can this be used in hospitals now?

No. This is a research and educational prototype. Before clinical use, it would require stronger validation, better calibration, regulatory approval, privacy safeguards, and testing on diverse real-world hospital data.

### Q8. What are the limitations?

The main limitations are:

- Low precision/F1 in the current checkpoint.
- Possible false positives.
- Dependency on dataset quality.
- Single top-label report in the current app.
- No full clinical text encoder yet.
- Not validated for real hospital deployment.

### Q9. What are future improvements?

Future improvements include:

- Per-class threshold tuning.
- Multi-label report with calibrated confidence.
- Full vision-language fusion using clinical notes.
- Better model backbones or ensembles.
- PDF report generation.
- Database storage of scan history.
- More validation on external datasets.

## 20. Future Scope

The project can be extended in several directions:

1. **Per-class thresholds**  
   Each disease can have its own prediction threshold to reduce false positives.

2. **Multi-label reporting**  
   The system can show multiple suspected diseases when confidence values are properly calibrated.

3. **Full vision-language fusion**  
   A text encoder can process symptoms, patient history, or radiology notes and combine them with image features.

4. **PDF report generation**  
   The app can export reports containing X-ray, heatmap, prediction, confidence, and explanation.

5. **Database integration**  
   Scan history and predictions can be stored for later review.

6. **Model comparison dashboard**  
   DenseNet121, ResNet50, and EfficientNet-B0 can be compared using metrics.

7. **External validation**  
   The model can be tested on other chest X-ray datasets for better generalization.

8. **Clinical safety layer**  
   The app can display risk categories, uncertainty warnings, and stronger disclaimers.

## 21. Conclusion

This project demonstrates a complete AI pipeline for chest X-ray disease detection. It includes dataset preprocessing, imbalance-aware training, model checkpointing, Flask-based deployment, Grad-CAM explainability, and language-assisted diagnostic reporting.

The DR-Ultra framework provides the foundation for robust thoracic disease detection from imbalanced radiographs. The current implementation proves the image-based detection workflow, while the language-assisted report interface makes the output easier to interpret. With future text encoder and fusion modules, the framework can evolve into a full vision-language medical AI system.

Overall, LumiScan AI is a strong academic prototype that combines deep learning, explainability, medical dataset handling, and practical web deployment.
