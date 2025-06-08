# 📧 MailTagger

The **The MailTagger App** is an intelligent web-based tool that uses a custom-trained AI model to automatically tag emails into predefined categories such as `placement_and_internship`, `course_updates`, `events`, `research_and_opportunity`, and more. It's built for students, professionals, and institutions to efficiently sort, filter, and analyze email content.

---

## 🔍 Features

- ✨ Automatically classify emails using a pre-trained ML model
- 🧠 Uses Logistic Regression / Random Forest / Naive Bayes under the hood
- 📊 Supports multi-class classification with customizable tags
- 📁 Attachment preview and download via Gmail API
- 🔐 OAuth2 authentication with Google for secure email access
- 🧪 Integrated with vectorizer and label encoder for reliable inference

---

## 🛠️ Tech Stack

- **Frontend:** HTML, CSS, JavaScript (Vanilla / React)
- **Backend:** Python (Flask)
- **ML Libraries:** Scikit-learn, Pandas, Joblib
- **Gmail API:** Google OAuth2, Gmail REST API
- **Data:** Custom labeled dataset (500+ realistic email samples with mix with AI generated data)

---

## 📦 Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/AtharvaGoliwar/MailTagger.git
cd MailTagger
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Load model and label encoder

Ensure you have:

- `email_tag_model.pkl`
- `vectorizer.pkl`
- `label_encoder.pkl`

### 4. Setup Google API credentials

- Go to [Google Cloud Console](https://console.cloud.google.com/)
- Create OAuth 2.0 credentials
- Download `credentials.json` and place it in the root directory

### 5. Run the app

```bash
python app.py
```

Then visit `http://localhost:5000` in your browser.

---

## 📂 Folder Structure

```
email-tag-classifier/
│
├──(Trained ML model, vectorizer, label encoder)
├── templates/               # Frontend HTML templates
├── static/                  # Static files (JS, CSS)
├── app.py                   # Flask backend server
├── classify.py              # Email classification logic
├── gmail_service.py         # Gmail API logic
├── requirements.txt
└── README.md
```

---

## 🏷️ Supported Tags

The model is trained on the following categories:

- `placement_and_internship`
- `course_updates`
- `events`
- `research_and_opportunity`
- `announcement`
- `spam`
- `hackathon`
- `technical_workshop`
- `vittbi`
- `hostel`

The model currently performs well across diverse categories of emails typically received by college students, professionals, and similar users.

Once **custom label classification** is implemented on the website, it will become significantly more flexible and robust, making it suitable for a broader range of users and use cases.

---

## 🔄 Future Enhancements

- ✅ Multi-label classification support along with **custom label classification**
- 🌐 Web-based dashboard with filter & search
- 🧾 Tag suggestions using GPT-based summarization
- ✅ Smart summarization of long emails
- 🧠 Auto-training pipeline with new data

---

## 📘 License

MIT License © 2025 \[AtharvaGoliwar]
