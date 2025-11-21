# 🤖 Hand Gesture Recognition using CNN | Task 4

This project focuses on building a **Convolutional Neural Network (CNN)** model to classify **10 different hand gesture classes** using the **LeapGestRecog** dataset.  
The dataset was originally structured in multiple nested subject folders, so it was **flattened, cleaned, and reorganized**, followed by an **80-20 train-test split**.

The model was trained on **Google Colab (GPU)** for faster performance and achieves strong accuracy in recognizing different hand gesture categories.

---

## 📦 Clone This Repository

```bash
git clone https://github.com/bhumi27-lab/PRODIGY_ML_04
cd PRODIGY_ML_04
```

---

## 🛠 How to Run This Project

1. Open the notebook (`notebook.ipynb`) in **Google Colab**
2. Click: **Runtime → Change runtime type → GPU**
3. Mount Google Drive inside the notebook
4. Download the dataset from Kaggle:  
   👉 **https://www.kaggle.com/datasets/gti-upm/leapgestrecog**
5. Upload the dataset ZIP into Google Drive
6. Run all cells step-by-step:
   - Dataset extraction  
   - Flattening script  
   - Train-test split  
   - ImageDataGenerator  
   - CNN model training  
   - Evaluation + predictions  
7. The trained model will be saved as:
   ```
   Task4_gesture_model.h5
   ```
8. You can download the model from Colab for further use.

---

## 📂 Dataset Used
- **LeapGestRecog Dataset from Kaggle**  
  🔗 https://www.kaggle.com/datasets/gti-upm/leapgestrecog  
- Contains images of **10 gesture classes (00–09)**
- Dataset was:
  - Flattened (merged subject folders)
  - Cleaned
  - Split into train & test sets

---

## 🧠 Model Details
- Built using **TensorFlow/Keras**
- Architecture:
  - `Conv2D` + `MaxPooling2D` layers
  - `Flatten` layer
  - Dense layers with ReLU activation
  - Output layer with Softmax (10 classes)
- Optimizer: **Adam**
- Loss: **Categorical Crossentropy**
- Trained on **GPU (T4)**

---

## 🚀 Steps Performed
1. Extracted dataset from Kaggle  
2. Flattened nested folder structure  
3. Created **train/test** split  
4. Preprocessed images (rescaling, resizing)  
5. Built a CNN model  
6. Trained for 10 epochs  
7. Evaluated model performance  
8. Saved final model (`Task4_gesture_model.h5`)

---

## 📈 Results
- High classification accuracy  
- Model successfully predicts hand gesture classes  
- Visualized:
  - Training Accuracy & Loss curves  
  - Confusion Matrix  
  - Sample prediction outputs  

---

## 🗃 Files in This Repository
- `notebook.ipynb` — Full Google Colab code   
- `requirements.txt` — All dependencies  
- `README.md` — Project overview  

*(Dataset not included due to size constraints.)*

---

## 💡 Future Improvements
- Add real-time webcam gesture recognition  
- Build a lightweight model for mobile deployment  
- Add GUI/interactive interface  

---

## 👩‍💻 Developer
**BHUMI SIRVI**  
Machine Learning Intern 🌟
