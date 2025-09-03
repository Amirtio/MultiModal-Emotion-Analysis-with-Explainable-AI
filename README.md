

# Multimodal Emotion Analysis System with Explainable AI

This project implements a comprehensive multimodal emotion analysis system that can analyze emotions from text, images, and audio files using state-of-the-art deep learning models. The system provides two distinct output modes: detailed analysis with Explainable AI features and simple results without detailed explanations.

---

## 🌟 Features

- **Text Emotion Analysis**: Advanced sentiment analysis using BERT-based models with keyword explanations  
- **Image Emotion Recognition**: Facial emotion detection from images with confidence scores  
- **Audio Emotion Analysis**: Speech emotion recognition from audio files  
- **Multimodal Fusion**: Intelligent combination of results from multiple modalities  
- **Explainable AI**: Detailed explanations of decision-making processes and confidence metrics  
- **Interactive Web Interface**: User-friendly Gradio interface for easy interaction  
- **Dual Output Modes**: Choose between detailed explanations or simple results  

---

## 🚀 Project Structure

```

multimodal-emotion-analysis/
│
├── notebooks/
│   ├── Untitled5.ipynb                 # Main implementation
│  
│
├── models/                             # Pre-trained models (auto-downloaded)
│   ├── text\_emotion/
│   ├── image\_emotion/
│   └── audio\_emotion/
│
├── examples/                           # Sample files for testing
│   ├── sample\_text.txt
│   ├── sample\_face.jpg
│   └── sample\_audio.wav
│
├── requirements.txt
└── README.md

````

---

## 📋 Requirements

### Core Dependencies
- **torch** (2.8.0+cu126)  
- **transformers** (4.55.4)  
- **gradio** (5.43.1)  
- **opencv-python** (4.12.0.88)  
- **pillow** (11.3.0)  
- **numpy** (2.0.2)  
- **librosa** & **soundfile** (for audio processing)  

### Explainable AI Dependencies
- **lime** (0.2.0.1)  
- **shap** (0.48.0)  
- **grad-cam** & **pytorch-gradcam** (for visual explanations)  

---

## 🛠️ Installation

1. **Clone or download the project:**
```markdown
   ```bash
   git clone https://github.com/your-username/multimodal-emotion-analysis.git
   cd multimodal-emotion-analysis
````

2. **Open in Google Colab or Jupyter:**

   * Upload the notebook files to your preferred environment
   * For Google Colab, mount your drive for model storage

3. **Install dependencies:**
   Run the first cell in any notebook to install all required packages:

   ```python
   !pip install torch torchvision torchaudio transformers opencv-python pillow numpy matplotlib seaborn
   !pip install gradio lime shap grad-cam pytorch-gradcam
   !pip install librosa soundfile datasets accelerate scikit-learn pandas plotly
   ```

---

## 💻 Usage

### Method 1: Simple Results Mode

Execute **Cells 1-6** in the main notebook to get basic emotion analysis without detailed explanations.

**Features:**

* Clean, concise emotion predictions
* Confidence scores for each modality
* Multimodal fusion results
* Fast processing

**Example Output:**

```
📝 Text: joy (87.3%)
🖼️ Image: happiness (82.1%) 
🎵 Audio: positive (79.8%)
🎯 Final Result: joy (83.1%)
```

---

### Method 2: Explainable AI Mode

Execute **Cells 1-6** + **Cell 7** (from Untitled5 (2).ipynb) to get comprehensive analysis with explanations.

**Features:**

* Detailed decision explanations
* Keyword analysis for text
* Confidence interpretation for images
* Fusion methodology explanations
* All emotion scores breakdown

**Example Output:**

```
📝 Text: joy (87.3%)
🖼️ Image: happiness (82.1%)
🎵 Audio: positive (79.8%)

🎯 Final Result: joy (83.1%)

==================================================
🧠 Explainable AI Analysis:
==================================================

🔍 Text Analysis: Positive keywords detected: "amazing", "wonderful", "excited"
📊 All Emotion Scores:
   • joy: 87.3%
   • excitement: 8.9%
   • neutral: 3.8%

✨ Image Analysis: High confidence detection (82.1%) - Clear facial expression
📊 Image Scores:
   • happiness: 82.1%
   • neutral: 12.4%
   • surprise: 5.5%

🔗 Fusion Method: Weighted combination using text, image, audio for comprehensive analysis
```

---

## 🎯 How to Use the Interface

1. **Launch the Interface**
   After running the appropriate cells, you'll receive a public URL:

   ```
   Running on public URL: https://xxxxxxxxx.gradio.live
   ```

2. **Input Your Data**

   * 📝 Text Box: Enter any text for sentiment analysis
   * 🖼️ Image Upload: Upload facial images (JPG, PNG, GIF)
   * 🎵 Audio Upload: Upload audio files (WAV, MP3, M4A)

3. **Analyze Results**
   Click "Submit" to get instant analysis. You can use any combination of inputs:

   * Text only
   * Image only
   * Audio only
   * Any combination of the above

4. **Interpret Results**

   * **Simple Mode**: Basic emotion labels with confidence
   * **Explainable AI Mode**: Detailed breakdowns and explanations

---

## 📊 Supported Emotions

**Primary Emotions:**

* 😊 Joy/Happiness
* 😢 Sadness
* 😠 Anger
* 😨 Fear
* 😮 Surprise
* 😐 Neutral

**Extended Emotions (text analysis):**

* Excitement
* Disappointment
* Love
* Optimism
* Pessimism

---

## 🔧 Configuration Options

### Text Analysis Parameters

```python
text_model = "j-hartmann/emotion-english-distilroberta-base"
max_length = 512
```

### Image Analysis Parameters

```python
image_model = "trpakov/vit-face-expression"  
confidence_threshold = 0.5
```

### Audio Analysis Parameters

```python
audio_model = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
sample_rate = 16000
```

---

## 🎨 Customization

### Adding New Models

```python
self.text_model = "your-custom-text-model"
self.image_model = "your-custom-image-model"
self.audio_model = "your-custom-audio-model"
```

### Modifying UI Theme

```python
demo = gr.Interface(
    # ... other parameters
    theme=gr.themes.Monochrome(),  # or Soft(), Glass(), etc.
)
```

### Custom CSS Styling

```python
css = """
.gradio-container {
    max-width: 1200px;
    margin: auto;
}
"""
demo = gr.Interface(css=css, ...)
```

---

## 🐛 Troubleshooting

### Common Issues

1. **Empty Interface**

   * Restart kernel and run all cells in order
   * Check if all dependencies are installed

2. **Model Loading Errors**

   * Ensure stable internet connection for model downloads
   * Check available storage space

3. **Audio Processing Issues**

   * Verify audio file format (WAV recommended)
   * Check if librosa is properly installed

4. **CUDA Errors**

   * The system works on both GPU and CPU
   * GPU acceleration is automatic if available

### Performance Tips

* **First Run**: May take longer due to model downloads
* **Large Files**: Audio/image files >50MB may process slowly
* **Memory**: Close other applications if running locally

---

## 📈 Technical Details

### Architecture Overview

```
Input Layer (Text/Image/Audio)
         ↓
   Individual Analyzers
    (BERT/ViT/Wav2Vec2)
         ↓
   Feature Extraction
         ↓
   Confidence Scoring
         ↓
   Multimodal Fusion
         ↓
   Explainable AI Layer
         ↓
    Final Predictions
```

### Fusion Strategy

```python
final_score = (text_conf * text_pred + image_conf * image_pred + audio_conf * audio_pred) / \
              (text_conf + image_conf + audio_conf)
```

---

## 🤝 Contributing

We welcome contributions! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

**Areas for Contribution:**

* Additional emotion categories
* New model integrations
* Performance optimizations
* UI/UX improvements
* Documentation enhancements

---

## 👥 Team & Contributors

- **Amirreza Navali** - *Initial Concept, Model Training & Core Logic* - [@amirtio](https://github.com/amirtio)

We collaborated on all aspects of the project, from brainstorming to final implementation.

---

**⭐ If you find this project useful, please consider giving it a star on GitHub!**

---

*This project demonstrates the power of multimodal AI and explainable machine learning for real-world emotion analysis applications. Perfect for research, education, and practical implementations in human-computer interaction, mental health monitoring, and social media analysis.*

```
