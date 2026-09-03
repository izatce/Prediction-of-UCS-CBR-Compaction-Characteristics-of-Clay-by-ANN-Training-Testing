# Soil Engineering Properties Prediction Using ANN

## Features
- Select input features from the uploaded Excel dataset
- Select any available numerical soil property as the output target
- Automatic ANN architecture selection
- 5-fold cross-validation
- Final training using the complete training dataset
- Independent unseen testing using a separate Excel file
- Standardization using StandardScaler
- SHAP Explainable AI
- Optional Gemini AI interpretation

## Possible Target Properties
Examples include UCS, soaked CBR, unsoaked CBR, OMC, MDD, cohesion (c), and angle of internal friction (phi), depending on the columns in your dataset.

## Important Rule
The selected target property must not also be selected as an input feature.

## Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Gemini API Key
Add this in Streamlit Secrets:

```toml
GEMINI_API_KEY = "your_api_key_here"
```

Never place the API key directly in app.py.
