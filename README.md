# Soil Engineering Properties Prediction Using ANN

## Features

- Select any numerical soil engineering property as the target
- Select input features from the Excel dataset
- Automatic ANN architecture selection
- 5-fold cross-validation
- Independent unseen testing
- Manual prediction for one new soil sample
- SHAP Explainable AI
- Optional Gemini AI interpretation

## Manual Prediction

After training and testing the model, open:

**🎯 5. Manual Prediction**

The app automatically creates one input field for every selected feature.

Enter one complete set of soil input values, then click:

**🎯 Predict [Target Property]**

The app displays the final predicted value.

## Example

If the selected inputs are:

- LL
- PL
- PI
- Specific Gravity
- Clay %

Enter one value for each parameter. The ANN then predicts one output value, such as UCS.

## Important

Predictions are generally more reliable when the new input values are within the range represented by the training dataset.

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Gemini API Key

Add the following to Streamlit Secrets:

```toml
GEMINI_API_KEY = "your_api_key_here"
```

Do not place the API key directly in `app.py`.
