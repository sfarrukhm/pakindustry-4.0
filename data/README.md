---

# 📘 `data/README.md` (copy this inside `data/`)

```markdown
# Data Sources & Download Instructions

## 1. Defect Detection
- **MVTec Anomaly Detection (AD)**  
  https://www.mvtec.com/company/research/datasets/mvtec-ad  
  Contains 5,000+ images across 15 categories with pixel-level ground truth.

- **Kolektor Surface-Defect Dataset (SDD / SDD2)**  
  https://www.vicos.si/resources/kolektorsdd/  
  Surface defect images (binary classification + segmentation).

- **NEU Surface Defect Dataset**  
  http://faculty.neu.edu.cn/me/zhanghf/NEU_surface_defect_database.html  
  1,800 grayscale steel defect images.

---

## 2. Predictive Maintenance
- **NASA C-MAPSS Turbofan RUL Dataset**  
  https://data.nasa.gov/dataset/C-MAPSS-Aircraft-Engine-Simulator-Data/ff5v-kuh6  

- **FEMTO Bearing Dataset (PRONOSTIA)**  
  https://www.femto-st.fr/en/Research-departments/AS2M/Research-groups/PHM/PRONOSTIA-database  

---

## 3. Forecasting
- **M5 Forecasting – Walmart Sales**  
  https://www.kaggle.com/competitions/m5-forecasting-accuracy  

---

## Notes
- Do **not** push raw datasets into GitHub. Store them locally in `data/raw/`.
- Preprocessed versions should go in `data/processed/`.
- Document any additional preprocessing in notebooks.
```
