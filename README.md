# Email Classification Chrome Extension
### Powered by DistilBERT | FastAPI | Chrome MV3

A complete end-to-end system that classifies emails into 6 Gmail-style categories
using a fine-tuned DistilBERT transformer model, served via a FastAPI backend and
surfaced as a Chrome Extension popup.

---

## Email Categories

| # | Category    | Description                                      |
|---|-------------|--------------------------------------------------|
| 0 | Promotions  | Deals, offers, marketing emails                  |
| 1 | Social      | Social media notifications, friend requests      |
| 2 | Updates     | Order updates, account alerts, app notifications |
| 3 | Forum       | Discussion threads, replies, community posts     |
| 4 | Spam        | Phishing, junk, unwanted mail                    |
| 5 | Verify_Code | OTP, 2FA, login verification codes               |

---

## Project Structure

```
6th Sem PBL Project/
|-- backend/
|   |-- app.py           (FastAPI server — POST /predict, GET /health)
|   |-- predictor.py     (Model load + inference, singleton)
|   |-- __init__.py
|-- model/
|   |-- train.py         (DistilBERT fine-tuning script)
|   |-- evaluate.py      (Accuracy, Precision, Recall, F1, confusion matrix)
|   |-- __init__.py
|   |-- saved_model/     (Created after training)
|-- dataset/
|   |-- preprocess.py    (HuggingFace dataset loader + text cleaner)
|   |-- __init__.py
|-- extension/
|   |-- manifest.json    (Chrome MV3 manifest)
|   |-- popup.html       (Extension popup UI)
|   |-- popup.js         (Popup logic — classify + render results)
|   |-- content.js       (Gmail DOM scraper)
|   |-- background.js    (Service worker / API relay)
|   |-- styles.css       (Dark-mode popup styles)
|   |-- icons/           (Extension icons: 16, 48, 128 px)
|-- run_server.py        (One-command server launcher)
|-- requirements.txt
|-- README.md
```

---

## Setup

### Step 1 - Create and activate virtual environment

```powershell
python -m venv myenv
myenv\Scripts\activate
```

### Step 2 - Install dependencies

```powershell
pip install -r requirements.txt
```

> First-time install downloads approx. 1 GB (PyTorch + Transformers). Stable internet needed.

### Step 3 - Get a Hugging Face token

1. Create a free account at https://huggingface.co
2. Go to https://huggingface.co/settings/tokens and click New Token (read access is enough)
3. Copy the token (it starts with hf_...)
4. Accept the dataset terms at:
   https://huggingface.co/datasets/jason23322/high-accuracy-email-classifier

---

## Training the Model

```powershell
# Set your HuggingFace token (replace with your actual token)
$env:HF_TOKEN = "hf_your_token_here"

# Train with 2,400 samples — approx. 10-15 min on CPU
python -m model.train --max_per_class 400 --max_length 128
```

### Training Flags

| Flag             | Default    | Description                                        |
|------------------|------------|----------------------------------------------------|
| --max_per_class  | None       | Samples per category. Use 400 for quick training   |
| --max_length     | 256        | Token length. Use 128 for 2x speed on CPU          |
| --epochs         | 3          | Number of training epochs                          |
| --batch_size     | 16         | Batch size (lower to 8 if memory issues)           |
| --lr             | 2e-5       | Learning rate                                      |

### What Happens During Training

1. Downloads the dataset from HuggingFace (~5 MB, cached after first run)
2. Downloads distilbert-base-uncased (~260 MB, first run only)
3. Fine-tunes for 3 epochs, printing progress every 10 steps
4. Saves the best checkpoint to model/saved_model/

> Note: The $env:HF_TOKEN variable must be set in the same terminal session as training.
> It resets when you close PowerShell — set it again in each new session.

---

## Evaluating the Model

Run this after training completes:

```powershell
python -m model.evaluate
```

Sample output:

```
====================================================
  EVALUATION RESULTS
====================================================
  Accuracy   :  0.8900   (89.00 %)
  Precision  :  0.8923   (macro avg)
  Recall     :  0.8887   (macro avg)
  F1-Score   :  0.8901   (macro avg)
====================================================
```

The confusion matrix is saved to model/confusion_matrix.png.

---

## Running the Backend Server

Open a dedicated PowerShell window and keep it running:

```powershell
cd "c:\Users\soni\Desktop\6th Sem PBL Project"
myenv\Scripts\activate
python run_server.py
```

Expected output:

```
Predictor ready on cpu
Server ready — listening for requests.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### Test the API

```powershell
# Health check
Invoke-WebRequest -Uri "http://localhost:8000/health" | Select-Object -ExpandProperty Content

# Classify an email
$body = '{"text": "Your OTP for login is 482910. Valid for 5 minutes."}'
Invoke-WebRequest -Uri "http://localhost:8000/predict" -Method POST `
  -ContentType "application/json" -Body $body | Select-Object -ExpandProperty Content
```

Expected response:

```json
{
  "category": "Verify_Code",
  "confidence": 0.9621,
  "all_scores": [
    {"label": "Verify_Code", "score": 0.9621},
    {"label": "Updates",     "score": 0.0187}
  ]
}
```

Interactive API docs: http://127.0.0.1:8000/docs

---

## Loading the Chrome Extension

1. Open Chrome and go to chrome://extensions
2. Enable Developer mode (toggle in the top-right corner)
3. Click Load unpacked
4. Select the extension/ folder inside this project
5. The Email Classifier icon will appear in your Chrome toolbar

### Using the Extension

**Auto-detect from Gmail:**
1. Go to mail.google.com
2. Open any email
3. Click the extension icon and then click "Detect from Gmail"
4. The predicted category and confidence score appear instantly

**Manual mode:**
1. Click the extension icon
2. Paste any email text into the textarea
3. Click "Classify Email" or press Ctrl + Enter

> The backend server (python run_server.py) must be running for the extension to work.

---

## Troubleshooting

| Problem                                 | Fix                                                          |
|-----------------------------------------|--------------------------------------------------------------|
| Backend offline shown in popup          | Run python run_server.py in a separate terminal              |
| Saved model not found                   | Run training first: python -m model.train                    |
| Failed to load dataset                  | Set $env:HF_TOKEN = "hf_..." and retrain                    |
| Gmail detect shows no email found       | Open a specific email in Gmail, not just the inbox list      |
| OOM or memory error during training     | Use --batch_size 8 or --max_length 64                        |
| Extension not updating after code change| Go to chrome://extensions and click the reload button        |
| Server shuts down immediately           | Keep the server in its own terminal — do not run other commands in the same window |

---

## Dataset

Source: jason23322/high-accuracy-email-classifier on Hugging Face
Link:   https://huggingface.co/datasets/jason23322/high-accuracy-email-classifier

| Property      | Value                                                    |
|---------------|----------------------------------------------------------|
| Total emails  | 12,000+                                                  |
| Categories    | 6 (Promotions, Social, Updates, Forum, Spam, Verify_Code)|
| Balance       | Approx. 2,000 emails per category                       |
| Format        | CSV and JSON                                             |
| License       | Apache 2.0                                               |

---

## Model Architecture

```
Input Email Text
      |
  [Text Cleaning]          Strip URLs, email addresses, special characters
      |
  [DistilBERT Tokenizer]   Max 128 tokens, padding and truncation
      |
  [DistilBERT Encoder]     6-layer transformer, 768-dimensional hidden states
      |
  [Classification Head]    Linear layer (768 -> 6)
      |
  [Softmax]                Probability score per category
      |
  Category + Confidence Score + All Category Scores
```

### Why DistilBERT?

- 40% smaller than full BERT
- 60% faster inference
- Retains 97% of BERT's accuracy
- Well suited for CPU deployment

---

## References

- DistilBERT paper: https://arxiv.org/abs/1910.01108 (Sanh et al., 2019)
- HuggingFace Transformers: https://huggingface.co/docs/transformers
- Email Classifier Dataset: https://huggingface.co/datasets/jason23322/high-accuracy-email-classifier
- Chrome Extension Manifest V3: https://developer.chrome.com/docs/extensions/mv3/intro/
- FastAPI Documentation: https://fastapi.tiangolo.com/

---

Built for 6th Semester PBL Project — Email Classification and Categorization using BERT Transformers
