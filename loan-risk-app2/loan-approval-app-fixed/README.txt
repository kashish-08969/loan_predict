# Loan Approval Prediction Website

## Important
Copy your trained `loan_model.pkl` file into this folder, beside `app.py`.
The website homepage can open without the model, but prediction requires it.

Folder layout:
- app.py
- loan_model.pkl
- requirements.txt
- templates/index.html

## Run in VS Code (Windows PowerShell)

1. Open this extracted folder in VS Code: `File > Open Folder`.
2. Open `Terminal > New Terminal`.
3. If your virtual environment is not active, activate it:
   `.\.venv\Scripts\Activate.ps1`
   (If `.venv` is not in this project folder, create it first with
   `python -m venv .venv`.)
4. Install dependencies:
   `python -m pip install -r requirements.txt`
5. Start the website:
   `python app.py`
6. Open this address in your browser:
   `http://127.0.0.1:5000`

Keep the terminal open while using the website. Press `Ctrl+C` in the terminal
when you want to stop it.

## If the model fails to load
The model was trained in Google Colab. In Colab, run:
`import sklearn; print(sklearn.__version__)`
In VS Code, run:
`python -c "import sklearn; print(sklearn.__version__)"`
If the versions differ, install the Colab version locally, for example:
`python -m pip install scikit-learn==1.5.2`
Replace `1.5.2` with the exact version shown in your Colab notebook.

Only use a model file that you created/trained yourself or trust. Loading
serialized Python model files can execute code embedded in the file.
