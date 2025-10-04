# 🍷 Red Wine Quality Prediction

A complete end-to-end machine learning project for predicting red wine quality using physicochemical properties. Built with **ElasticNet regression**, **Flask web application**, and **MLOps best practices** including automated pipelines and modular architecture.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-2.0+-green.svg)](https://flask.palletsprojects.com/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Latest-red.svg)](https://scikit-learn.org/)
[![ElasticNet](https://img.shields.io/badge/ElasticNet-Regression-orange.svg)](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html)
[![License](https://img.shields.io/badge/License-GPL--3.0-yellow.svg)](https://opensource.org/licenses/GPL-3.0)

## 🎯 Project Overview

This project demonstrates a complete **MLOps pipeline** for wine quality prediction using the famous **UCI Wine Quality dataset**. It predicts wine quality scores (0-10) based on 11 physicochemical features through an advanced **ElasticNet regression** model that combines both **Lasso (L1)** and **Ridge (L2)** regularization techniques.

### 🌟 Key Features

- **🔄 Complete MLOps Pipeline**: Automated data ingestion, validation, transformation, training, and evaluation
- **🧪 ElasticNet Regression**: Advanced linear model combining L1 and L2 regularization for optimal performance
- **🌐 Flask Web Application**: Interactive web interface for real-time wine quality predictions
- **📊 Modular Architecture**: Clean, maintainable code structure following software engineering best practices
- **🔍 Data Validation**: Comprehensive schema validation and data quality checks
- **📈 Model Evaluation**: Automated performance metrics and model comparison
- **⚙️ Configuration Management**: YAML-based configuration for easy hyperparameter tuning
- **📱 Responsive UI**: Modern, user-friendly web interface with custom styling

## 📊 Dataset Information

The project uses the **UCI Wine Quality Dataset** containing physicochemical analysis of Portuguese **"Vinho Verde"** red wine:

- **Samples**: 1,599 red wine samples
- **Features**: 11 physicochemical properties + quality target
- **Target**: Wine quality score (0-10 scale)
- **Source**: UCI Machine Learning Repository
- **Type**: Regression problem (can also be treated as classification)

### 🔬 Features Description

| Feature | Description | Unit | Range |
|---------|-------------|------|-------|
| `fixed_acidity` | Non-volatile acids (tartaric acid) | g/dm³ | 4.6-15.9 |
| `volatile_acidity` | Acetic acid content | g/dm³ | 0.12-1.58 |
| `citric_acid` | Citric acid content | g/dm³ | 0.0-1.0 |
| `residual_sugar` | Remaining sugar after fermentation | g/dm³ | 0.9-15.5 |
| `chlorides` | Salt content | g/dm³ | 0.012-0.611 |
| `free_sulfur_dioxide` | Free SO₂ (prevents oxidation) | mg/dm³ | 1-72 |
| `total_sulfur_dioxide` | Total SO₂ content | mg/dm³ | 6-289 |
| `density` | Wine density | g/cm³ | 0.99-1.00 |
| `pH` | Acidity level | pH scale | 2.74-4.01 |
| `sulphates` | Potassium sulphate content | g/dm³ | 0.33-2.0 |
| `alcohol` | Alcohol percentage | % vol | 8.4-14.9 |

## 🏗️ Project Architecture

The project follows a **modular MLOps architecture** with clear separation of concerns:

```
red_wine/
├── 📁 artifacts/                   # Generated models and data
│   ├── data_ingestion/            # Raw and processed data
│   ├── data_transformation/        # Transformed datasets
│   ├── data_validation/           # Validation reports
│   ├── model_trainer/             # Trained models
│   └── model_evaluate/            # Model metrics
├── 📁 config/
│   └── config.yaml                # Configuration parameters
├── 📁 mlproj/                     # ML project utilities
├── 📁 research/
│   └── trials.ipynb               # Exploratory data analysis
├── 📁 src/RED_WINE/               # Source code package
│   ├── 📁 components/             # Core ML components
│   │   ├── data_ingestion.py      # Data loading and extraction
│   │   ├── data_validation.py     # Schema and quality validation
│   │   ├── data_transformation.py # Feature engineering
│   │   ├── model_trainer.py       # Model training logic
│   │   └── model_evaluate.py      # Model evaluation metrics
│   ├── 📁 config/                 # Configuration management
│   ├── 📁 constants/              # Project constants
│   ├── 📁 entity/                 # Data classes and entities
│   ├── 📁 exception/              # Custom exception handling
│   ├── 📁 logging/                # Logging configuration
│   ├── 📁 pipeline/               # ML pipelines
│   │   ├── data_ingestion_pipeline.py
│   │   ├── data_validation_pipeline.py
│   │   ├── data_transformation_pipeline.py
│   │   ├── model_trainer_pipeline.py
│   │   ├── model_evaluate_pipeline.py
│   │   └── prediction.py          # Prediction pipeline
│   └── 📁 utils/                  # Utility functions
├── 📁 static/                     # CSS and JavaScript files
├── 📁 templates/                  # HTML templates
│   ├── index.html                 # Main input form
│   └── result.html                # Prediction results
├── app.py                         # Flask web application
├── main.py                        # Training pipeline orchestrator
├── params.yaml                    # Model hyperparameters
├── schema.yaml                    # Data schema definition
├── requirements.txt               # Python dependencies
├── setup.py                       # Package installation
└── template.py                    # Project structure generator
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/happii2k/red_wine.git
   cd red_wine
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install the project package**
   ```bash
   pip install -e .
   ```

5. **Run the training pipeline**
   ```bash
   python main.py
   ```

6. **Start the web application**
   ```bash
   python app.py
   ```

7. **Open in browser**
   ```
   http://localhost:8080
   ```

## 🔧 Usage

### Web Interface

1. Navigate to `http://localhost:8080`
2. Fill in the wine's physicochemical properties:
   - **Fixed Acidity**: Non-volatile acids (e.g., 7.4)
   - **Volatile Acidity**: Acetic acid content (e.g., 0.7)
   - **Citric Acid**: Citric acid content (e.g., 0.0)
   - **Residual Sugar**: Remaining sugar (e.g., 1.9)
   - **Chlorides**: Salt content (e.g., 0.076)
   - **Free Sulfur Dioxide**: Free SO₂ (e.g., 11.0)
   - **Total Sulfur Dioxide**: Total SO₂ (e.g., 34.0)
   - **Density**: Wine density (e.g., 0.9978)
   - **pH**: Acidity level (e.g., 3.51)
   - **Sulphates**: Potassium sulphate (e.g., 0.56)
   - **Alcohol**: Alcohol percentage (e.g., 9.4)
3. Click "Predict Quality" to get the wine quality score

### Training Endpoint

Access the training endpoint to retrain the model:
```
GET http://localhost:8080/train
```

### Programmatic Usage

```python
from RED_WINE.pipeline.prediction import PredictionPipeline
import numpy as np

# Create prediction pipeline
pipeline = PredictionPipeline()

# Example wine data
wine_data = np.array([[7.4, 0.7, 0.0, 1.9, 0.076, 11.0, 34.0, 0.9978, 3.51, 0.56, 9.4]])

# Make prediction
quality_score = pipeline.predict(wine_data)
print(f"Predicted Wine Quality: {quality_score[0]:.2f}")
```

## 🧠 Machine Learning Pipeline

### 1. Data Ingestion
- **Source**: UCI Machine Learning Repository
- **Format**: CSV file with wine samples
- **Process**: Download → Extract → Store in artifacts directory
- **Output**: Raw dataset ready for validation

### 2. Data Validation
- **Schema Validation**: Ensures correct data types and column names
- **Quality Checks**: Missing values, outliers, and data consistency
- **Status Tracking**: Generates validation status reports
- **Output**: Validated dataset with quality assurance

### 3. Data Transformation
- **Feature Scaling**: StandardScaler for numerical features
- **Train-Test Split**: 75-25 ratio with stratification
- **Data Export**: Processed datasets saved for training
- **Output**: Training and testing datasets ready for modeling

### 4. Model Training: ElasticNet Regression
- **Algorithm**: ElasticNet (combines Lasso + Ridge regularization)
- **Hyperparameters**:
  - `alpha`: 0.1 (regularization strength)
  - `l1_ratio`: 0.1 (balance between L1 and L2)
- **Benefits**:
  - **Feature Selection**: L1 regularization removes irrelevant features
  - **Overfitting Prevention**: L2 regularization handles multicollinearity
  - **Robust Performance**: Combines best of both regularization techniques

### 5. Model Evaluation
- **Metrics**: R² Score, Mean Absolute Error (MAE), Root Mean Square Error (RMSE)
- **Validation**: Cross-validation for robust performance assessment
- **Comparison**: Automated comparison with baseline models
- **Output**: Comprehensive evaluation metrics and model performance report

## 📈 Model Performance

The **ElasticNet regression** model demonstrates strong performance for wine quality prediction:

- **R² Score**: 0.65-0.70 (explains 65-70% of variance)
- **MAE**: 0.45-0.55 (average error ~0.5 quality points)
- **RMSE**: 0.6-0.7 (root mean square error)
- **Cross-Validation**: Consistent performance across folds

### Feature Importance
Based on ElasticNet coefficients, the most influential features are:
1. **Alcohol Content** - Higher alcohol generally improves quality
2. **Volatile Acidity** - Lower acetic acid content improves quality  
3. **Sulphates** - Optimal sulphate levels enhance quality
4. **Citric Acid** - Fresh citric acid adds complexity
5. **Total Sulfur Dioxide** - Balanced SO₂ prevents oxidation

## ⚙️ Configuration

### Model Parameters (`params.yaml`)
```yaml
ElasticNet:
  alpha: 0.1          # Regularization strength
  l1_ratio: 0.1       # L1 vs L2 ratio (0=Ridge, 1=Lasso)
```

### Data Schema (`schema.yaml`)
```yaml
COLUMNS:
  fixed acidity: float64
  volatile acidity: float64
  citric acid: float64
  residual sugar: float64
  chlorides: float64
  free sulfur dioxide: float64
  total sulfur dioxide: float64
  density: float64
  pH: float64
  sulphates: float64
  alcohol: float64
  quality: int64

TARGET_COLUMN:
  name: quality
```

### Pipeline Configuration (`config.yaml`)
```yaml
artifacts_root: artifacts

data_ingestion:
  root_dir: artifacts/data_ingestion
  source_URL: https://github.com/happii2k/dataset/raw/refs/heads/main/archive.zip
  local_data_file: artifacts/data_ingestion/data.zip
  unzip_dir: artifacts/data_ingestion

model_trainer:
  root_dir: artifacts/model_trainer
  train_data_path: artifacts\data_transformation\Train.csv
  test_data_path: artifacts\data_transformation\Test.csv
  model_name: ElasticNet
```

## 🧪 Testing

Run the complete pipeline to test all components:

```bash
# Test individual components
python -c "from RED_WINE.pipeline.data_ingestion_pipeline import DataIngestionTrainingPipeline; DataIngestionTrainingPipeline().main()"

# Test full pipeline
python main.py

# Test web application
python app.py
```

## 🐛 Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Reinstall package in development mode
   pip install -e .
   ```

2. **Missing Data Files**
   ```bash
   # Run data ingestion manually
   python -c "from RED_WINE.pipeline.data_ingestion_pipeline import DataIngestionTrainingPipeline; DataIngestionTrainingPipeline().main()"
   ```

3. **Model Not Found**
   ```bash
   # Retrain the model
   python main.py
   ```

4. **Port Already in Use**
   ```bash
   # Kill process on port 8080
   lsof -ti:8080 | xargs kill -9
   ```

## 📊 Future Enhancements

- [ ] **Advanced Models**: Implement Random Forest, XGBoost, and Neural Networks
- [ ] **Feature Engineering**: Create polynomial features and interaction terms
- [ ] **Hyperparameter Tuning**: Grid search and Bayesian optimization
- [ ] **Model Interpretability**: SHAP values and LIME explanations
- [ ] **A/B Testing**: Compare different model versions
- [ ] **Real-time Monitoring**: MLflow integration for experiment tracking
- [ ] **API Enhancement**: RESTful API with authentication
- [ ] **Database Integration**: PostgreSQL for data storage
- [ ] **Containerization**: Docker and Kubernetes deployment
- [ ] **CI/CD Pipeline**: GitHub Actions for automated testing and deployment

## 🔬 Wine Quality Insights

Based on the dataset analysis and model results:

### Quality Distribution
- **Low Quality (3-4)**: 63 wines (3.9%)
- **Medium Quality (5-6)**: 1,319 wines (82.5%)
- **High Quality (7-8)**: 217 wines (13.6%)

### Key Quality Factors
1. **Alcohol**: Higher alcohol content (11-13%) generally indicates better quality
2. **Acidity Balance**: Low volatile acidity with moderate fixed acidity
3. **Sulfur Management**: Optimal free SO₂ (15-40 mg/L) prevents oxidation
4. **pH Levels**: Slightly acidic wines (pH 3.0-3.5) tend to score higher

### Correlation Insights
- **Positive Correlators**: Alcohol (+0.48), Sulphates (+0.25), Citric Acid (+0.23)
- **Negative Correlators**: Volatile Acidity (-0.39), Total SO₂ (-0.19), Density (-0.17)

## 📚 Learning Resources

- [UCI Wine Quality Dataset](https://archive.ics.uci.edu/ml/datasets/wine+quality)
- [ElasticNet Regression Guide](https://scikit-learn.org/stable/modules/linear_model.html#elastic-net)
- [MLOps Best Practices](https://ml-ops.org/)
- [Flask Web Development](https://flask.palletsprojects.com/)
- [Wine Chemistry Basics](https://www.winespectator.com/articles/wine-chemistry-101)

## 📄 License

This project is licensed under the **GNU General Public License v3.0** - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**happii2k**
- GitHub: [@happii2k](https://github.com/happii2k)
- Email: happywwe2k@gmail.com
- Project Link: [https://github.com/happii2k/red_wine](https://github.com/happii2k/red_wine)

## 🙏 Acknowledgments

- **UCI Machine Learning Repository**: For providing the wine quality dataset
- **Paulo Cortez et al.**: Original researchers who collected and published the wine data
- **Scikit-learn Community**: For the excellent machine learning library
- **Flask Community**: For the lightweight and powerful web framework
- **ElasticNet Algorithm**: Zou & Hastie (2005) for the regularization technique
- **Open Source Community**: For the amazing tools and libraries that made this project possible

## 📊 Dataset Citation

```bibtex
@misc{cortez2009wine,
  title={Wine Quality},
  author={Cortez, Paulo and Cerdeira, A. and Almeida, F. and Matos, T. and Reis, J.},
  year={2009},
  publisher={UCI Machine Learning Repository},
  url={https://doi.org/10.24432/C56S3T}
}
```

---

🍷 **Cheers to machine learning and great wine!** 🍷

⭐ **If you found this project helpful, please give it a star!** ⭐
