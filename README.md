# HDFC Loan Approval Predictor

Streamlit deployment package for the UCS321 HDFC Bank loan approval classification project.

## Files

- `app.py` — Streamlit web application
- `requirements.txt` — Python dependencies
- `loan_approval_dataset.csv` — put your actual dataset here before deploying

## Local run

Open a terminal in this folder:

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app will open in your browser.

## Streamlit Community Cloud

1. Create a GitHub repository.
2. Upload `app.py`, `requirements.txt`, and your `loan_approval_dataset.csv`.
3. Go to https://share.streamlit.io/
4. Sign in with GitHub.
5. Choose **Create app**.
6. Select your repository, branch (`main`) and `app.py`.
7. Click **Deploy**.

The app will receive a `streamlit.app` URL.

If you do not want to commit the dataset to GitHub, the app also supports uploading the CSV directly from the web interface.
