Unified Explainable AI Interface
================================

Project Overview
----------------
This project integrates two existing Explainable AI (XAI) systems into a single
interactive platform supporting both image and audio classification.

The application allows users to:
- Upload an image or an audio file
- Select a compatible pretrained model
- Apply explainability techniques (Grad-CAM, LIME, SHAP)
- Compare multiple XAI methods side-by-side on the same input

The goal is to provide a unified, user-friendly interface for multi-modal
classification and explainability analysis.


Integrated Repositories
-----------------------
1) Deepfake Audio Detection with XAI
   - Audio classification (real vs fake)
   - Models: VGG16, MobileNet
   - XAI: Grad-CAM, LIME, SHAP

2) Lung Cancer Detection from Chest X-rays
   - Image classification (disease vs no_disease)
   - Models: AlexNet, DenseNet
   - XAI: Grad-CAM, LIME, SHAP


Project Structure
-----------------
```
streamlit/
│
├── app.py                     # Landing page + navigation
│
├── pages/
│   ├── 1_Inference.py          # Single model + single XAI
│   ├── 2_Comparison.py         # Side-by-side XAI comparison
│
models/
│   ├── image/                 # AlexNet, DenseNet
│   ├── audio/                 # MobileNet, VGG16
│   └── registry.py            # Central model registry
│
preprocessing/
│   ├── image.py               # Image preprocessing
│   └── audio.py               # Audio → spectrogram preprocessing
│
xai/
│   ├── gradcam.py
│   ├── lime.py
│   └── shap.py
```

How to Run the Application
--------------------------
1) Activate the virtual environment
2) Install dependencies:
   pip install -r requirements.txt

3) Launch the Streamlit app:
   streamlit run streamlit/app.py

4) Use the sidebar to navigate between pages:
   - Inference
   - XAI Comparison


Page 1 – Inference
------------------
- Upload an image or audio file
- Select a compatible model
- Select one XAI method
- View:
  - Prediction result
  - Confidence score
  - Corresponding explainability visualization


Page 2 – XAI Comparison
----------------------
- Upload one image or audio file
- Select one model
- Automatically applies:
  - Grad-CAM
  - LIME
  - SHAP
- Displays all explanations side-by-side
- Shows prediction confidence
- Automatically infers ground truth from filename when possible
  (TP / FP / FN / TN displayed for reference)


Automatic Compatibility Handling
--------------------------------
- Image inputs only show image-compatible models
- Audio inputs only show audio-compatible models
- XAI methods are filtered based on input type
- Incompatible options are hidden automatically


Notes on SHAP
-------------
SHAP explanations may take longer to compute depending on the model.
This is expected behavior due to gradient-based attribution.


## Use of Generative AI Tools

- Generative AI tools were used during the development of this project in a controlled and transparent manner, in accordance with the course rules.

### Tools used

- OpenAI ChatGPT (Large Language Model)

### Purpose of use
- Generative AI was used as a technical assistant for:
- Refactoring existing code and improving modularity and readability
- Debugging runtime errors and resolving shape, compatibility and integration issues
- Assisting in the design of the unified multi-modal application architecture (audio + image pipelines, page separation, XAI workflow)
- Helping draft parts of the documentation and clarify technical explanations

### Scope and limitations
- All architectural decisions, implementation choices, model integrations and explainability pipelines were designed, implemented and validated by the project authors.
- Generative AI was not used to automatically generate a complete solution, nor to replace understanding, design or decision-making.
- Classification models, preprocessing pipelines and XAI methods were either implemented by us or reused from existing works and are explicitly cited and referenced.
- The final codebase is the result of human-driven design, testing and validation.

### Responsibility
- The authors take full responsibility for the correctness, relevance and compliance of the final implementation with the project specifications.
- Generative AI tools were used strictly as support for productivity and clarification, not as a substitute for independent technical work.

Authors
-------
Project developed by Hugo Bonnell and Mathieu Cowan

