# Regression on the tabular data. General Machine Learning

## Installation

The code uses **Python 3.8**.

#### Create a Conda virtual environment:

```bash
conda create -n regressiontabdata python=3.8
conda activate regressiontabdata
```

#### Clone and install requirements:
```bash
git clone https://github.com/yankur/regressiontabdata.git
cd regressiontabdata
pip install -r requirements.txt
```

## Project Structure:
#### Exploratory Data Analysis
Open the Jupyter notebook in the notebooks directory to view the EDA:

```bash
jupyter notebook notebooks/exploratory_data_analysis.ipynb
```

#### Model Training
Run the train.py script to train the model with hyperparameter tuning:

```bash
python scripts/train.py --data_path data/train.csv --model_path weights/xgboost_model.pkl --scaler_path weights/scaler.pkl
```

#### Model Inference
Run the predict.py script to generate predictions:

```bash
python scripts/predict.py --data_path data/hidden_test.csv --model_path weights/xgboost_model.pkl --scaler_path weights/scaler.pkl --output_path predictions.csv
```


## Project Description:

The dataset (data/train.csv) contains 53 anonymized 
features and a target column. The task was to build a model 
that predicts a target based on the proposed features.
Target metric is RMSE.
The model chosen is XGBoost.

*Best parameters found: {'colsample_bytree': 1.0, 'learning_rate': 0.1, 'max_depth': 7, 'n_estimators': 300, 'subsample': 1.0}*
*Best cross-validation RMSE: 0.014603752742449544*

Please see comprehensive analysis and explanation of choise in Exploratory Data Analysis notebook

