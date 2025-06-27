# COVID-19 Time Series Forecasting: Statistical Models vs Deep Learning

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io)

## Project Overview

A comprehensive comparative study evaluating **statistical methods vs deep learning approaches** for multi-country COVID-19 time series forecasting. This project implements and compares Theta statistical models with Recurrent Neural Networks (RNN) to determine the most effective approach for epidemiological prediction.

### Key Results
- **RNN outperformed Theta model by 60%** (MSE: 0.98 vs 2.43)
- **Positive predictive value**: RNN achieved R² = 0.10, while Theta showed R² = -1.23
- **Multi-country analysis**: Robust performance across different geographical regions
- **Sequence-to-sequence prediction**: 14-day ahead forecasting capability

---

## Live Demo

https://mta5dmlrknz349bxcvif2q.streamlit.app/
---

## Architecture & Methodology

### Data Pipeline
```
Original_data.csv → Data Cleaning → Cleaned_data.csv → Scaling/Normalization → Scaled_data.csv → Feature Engineering → Windowing → Model Training → Evaluation
```

### Data Files
- **Original_data.csv**: Raw COVID-19 time series data with original values
- **Cleaned_data.csv**: Data after cleaning (missing values, outliers handled)  
- **Scaled_data.csv**: Normalized data ready for machine learning models

### Models Implemented
1. **Statistical Approach**: Theta Model with seasonal decomposition
2. **Deep Learning Approach**: Multi-layer RNN with LSTM cells

### Input Features
- `new_cases`: Daily new COVID-19 cases
- `tests_done`: Daily testing volume
- `positivity_rate`: Test positivity percentage
- `population`: Country population (normalization factor)

---

## Technical Implementation

### Model Architecture
```python
# RNN Configuration
- Input Dimensions: 4 features
- Hidden Dimensions: 8 units (optimized to prevent overfitting)
- Layers: 2 LSTM layers
- Output: 14-day sequence prediction
- Regularization: 0.2 dropout
```

### Performance Metrics
| Model | MSE (lower is better) | RMSE (lower is better) | MAE (lower is better) | R² (higher is better) | MAPE (lower is better) |
|-------|-------|--------|-------|------|--------|
| **RNN** | **0.98** | **0.99** | **0.60** | **0.10** | **367.68** |
| Theta | 2.43 | 1.56 | 0.81 | -1.23 | 481.51 |

---

## Installation & Setup

### Prerequisites
```bash
Python 3.8+
pip or conda package manager
```

### Installation
```bash
# Clone the repository (replace with your actual repository URL)
git clone https://github.com/AbhilashAdya/Deep-learning-LSTM-vs-Statistical-model-Theta-
cd your-repo-name

# Install required packages
pip install -r requirements.txt

# Or using conda
conda env create -f environment.yml
conda activate covid-forecasting
```

### Dependencies
```txt
torch>=2.0.0
pandas>=1.5.0
numpy>=1.24.0
matplotlib>=3.6.0
seaborn>=0.12.0
streamlit>=1.28.0
plotly>=5.15.0
statsmodels>=0.14.0
scikit-learn>=1.3.0
```

---

## Usage

### 1. Data Preprocessing & Model Training
```bash
# Run the complete ML pipeline
python main.py
```
This will:
- Load and preprocess COVID-19 data from Data/Scaled_data.csv
- Train both RNN and Theta models
- Generate performance comparisons
- Save results to `results/` directory

### 2. Launch Interactive Dashboard
```bash
# Start the Streamlit dashboard
streamlit run streamlit_app.py
```
Access the dashboard at `http://localhost:8501`

---

## Project Structure

```
covid-forecasting-project/
├── README.md                              # Project documentation
├── main.py                                # ML pipeline orchestrator
├── streamlit_app.py                       # Interactive dashboard
├── config.py                              # Configuration parameters
├── requirements.txt                       # Python dependencies
├── src/
│   ├── __init__.py
│   ├── models.py                          # RNN and Theta model implementations
│   ├── evaluation.py                      # Model comparison and metrics
│   └── utils.py                           # Data processing utilities
├── notebooks/
│   └── exploratory_data_analysis.ipynb   # EDA and data insights
├── Data/
│   ├── Original_data.csv                  # Raw COVID-19 dataset
│   ├── Cleaned_data.csv                   # Preprocessed dataset
│   └── Scaled_data.csv                    # Normalized dataset for modeling
└── results/
    ├── plots/                             # Generated visualizations
    ├── models/                            # Saved model weights
    └── reports/                           # Performance reports
```

---

## Key Features

### Advanced ML Pipeline
- **Time-aware data splitting**: Prevents data leakage in temporal analysis
- **Sequence-to-sequence modeling**: Multi-step ahead forecasting
- **Hyperparameter optimization**: Systematic approach to prevent overfitting
- **Cross-validation**: Robust model evaluation methodology

### Interactive Dashboard
- **Multi-page navigation**: Project overview, data exploration, model comparison
- **Dynamic visualizations**: Plotly-powered interactive charts
- **Real-time insights**: Instant access to model performance metrics
- **Professional presentation**: Clean, employer-ready interface

### Comprehensive Analysis
- **Statistical decomposition**: Trend, seasonal, and residual analysis
- **Multi-country comparison**: Robust evaluation across different regions
- **Performance benchmarking**: Detailed metrics comparison
- **Business insights**: Practical implications and recommendations

---

## Results & Insights

### Model Performance Summary
- **Winner**: RNN consistently outperformed statistical methods
- **Improvement**: 60% reduction in prediction error (MSE)
- **Stability**: Positive R² indicates meaningful pattern recognition
- **Scalability**: Effective across multiple countries and time periods

### Business Implications
1. **Deep learning approaches** show superior performance for complex epidemiological forecasting
2. **Multi-feature models** capture underlying dynamics better than single-variable approaches
3. **Sequence-to-sequence prediction** enables practical 2-week ahead planning
4. **Model interpretability** vs performance trade-offs clearly demonstrated

---

## Skills Demonstrated

### Technical Skills
- **Machine Learning**: Time series forecasting, model comparison, hyperparameter tuning
- **Deep Learning**: RNN/LSTM architecture, sequence modeling, PyTorch implementation
- **Statistical Analysis**: Time series decomposition, statistical testing, model evaluation
- **Data Engineering**: ETL pipelines, feature engineering, data validation
- **Software Engineering**: Modular design, clean code, documentation

### Tools & Technologies
- **Languages**: Python
- **ML Frameworks**: PyTorch, Scikit-learn, Statsmodels
- **Visualization**: Matplotlib, Seaborn, Plotly
- **Web Framework**: Streamlit
- **Data Processing**: Pandas, NumPy
- **Development**: Jupyter, Git, VS Code

---

## Future Enhancements

- [ ] **Ensemble methods**: Combine multiple models for improved accuracy
- [ ] **Real-time data integration**: Connect to live COVID-19 APIs
- [ ] **Geographic analysis**: Incorporate spatial correlation features
- [ ] **Uncertainty quantification**: Add confidence intervals to predictions
- [ ] **Model deployment**: Production-ready API with monitoring
- [ ] **A/B testing framework**: Systematic model comparison methodology

---

## License

This project is licensed under the MIT License.

---

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## Author

**AbhilashAdya**
- GitHub: (https://github.com/AbhilashAdya)
- LinkedIn: (https://www.linkedin.com/in/abhilashadya/) 
- Email: abhilashadya1303@gmail.com

---

## Acknowledgments

- COVID-19 data source: European Center for Disease Prevention and Control 
- PyTorch and Streamlit communities for excellent frameworks
- Statistical forecasting research community for baseline methodologies

*This project demonstrates end-to-end machine learning capabilities from data preprocessing through model deployment, showcasing both technical depth and practical business application.*