# Report / PPT Outline — Plant Disease Detection Using Leaf Images

Use this outline for both your written report and your presentation slides.
Suggested: ~10-12 slides for a PPT, or a section-per-heading for a report.

## 1. Title Slide
- Project title, your name, roll number, guide name, department, college

## 2. Abstract
- 4-6 sentence summary: problem, approach (transfer learning CNN), dataset,
  results (test accuracy), and deployment (web app).

## 3. Introduction / Problem Statement
- Why plant disease detection matters (crop loss, food security)
- Limitations of manual inspection by farmers
- Goal: automated, image-based disease classification tool

## 4. Objectives
- Build an accurate CNN classifier for 38 plant leaf classes
- Provide an easy-to-use web interface for farmers/students
- Store and review prediction history
- Deploy as a free, publicly accessible web app

## 5. Literature Survey (brief)
- Mention 2-3 prior works/papers on plant disease detection using CNNs
- Note common approaches: custom CNNs vs. transfer learning (VGG, ResNet, MobileNet)

## 6. Dataset
- PlantVillage dataset: 38 classes, ~54,000+ images, 14 crop species
- Train/Val/Test split: 80/10/10
- Example images (include a few class samples as a grid figure)

## 7. Methodology / System Architecture
- Pipeline diagram: Image Upload → Preprocessing → CNN Model → Prediction → Remedy → Database
- Model: MobileNetV2 (ImageNet pretrained) + custom classification head
- Data augmentation: rotation, flip, zoom, contrast
- Training details: optimizer (Adam), loss (categorical cross-entropy),
  callbacks (early stopping, LR reduction, checkpointing)
- Tech stack diagram: FastAPI backend, Streamlit frontend, SQLite DB

## 8. Implementation
- Brief code walkthrough (key snippets): model building, training loop,
  FastAPI `/predict` endpoint, Streamlit UI
- Tools used: TensorFlow/Keras, FastAPI, Streamlit, SQLite

## 9. Results
- Training/validation accuracy & loss curves (insert `training_history.png`)
- Final test accuracy/loss (from `test_results.json`)
- Confusion matrix or sample predictions (optional, add if time permits)
- Screenshots of the working web app (Predict page + History page)

## 10. Conclusion
- Summary of what was achieved
- Practical usefulness for farmers/students
- Note on model accuracy and reliability

## 11. Limitations & Future Scope
- Model trained on lab-condition leaf images (PlantVillage); may not
  generalize well to real field photos with complex backgrounds
- Future: expand to more crop species, mobile app version, multilingual
  remedy suggestions, real-time camera detection

## 12. References
- PlantVillage dataset citation (Kaggle/original paper)
- TensorFlow, FastAPI, Streamlit documentation links
- Any research papers cited in literature survey

---

### Tips for the PPT version
- Keep each slide to 3-5 bullet points max
- Use the architecture diagram and accuracy/loss plots as visual anchors
- Include at least 2 screenshots of the actual working app (not mockups)
- Practice explaining the "why transfer learning" choice — a common
  question from evaluators
