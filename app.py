import os
import base64
import pickle
import numpy as np
from flask import Flask, redirect, url_for, session, render_template, request, jsonify
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
from google.auth.transport.requests import Request
from bs4 import BeautifulSoup
import re
from dotenv import load_dotenv

load_dotenv()



app = Flask(__name__)
app.secret_key = 'REPLACE_THIS_WITH_SECRET_KEY'

SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']
# TOKEN_FILE = 'token.json'
# CREDS_FILE = 'credentials.json'

# Model configuration
MODEL_FILE = 'email_tag_model.pkl'
VECTORIZER_FILE = 'vectorizer.pkl'
LABEL_ENCODER_FILE = 'label_mapping.pkl'

# Email Classification Model Class
class EmailClassifier:
    def __init__(self, model_path=MODEL_FILE, vectorizer_path=VECTORIZER_FILE, label_encoder_path=LABEL_ENCODER_FILE):
        """Initialize the email classifier with your trained model"""
        self.model = None
        self.vectorizer = None
        self.label_encoder = None
        self.class_names = None

        # Load your trained model
        try:
            print(f"Loading model from {model_path}")
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            # Handle different pickle file structures
            if isinstance(model_data, dict):
                # If your pickle file contains a dictionary with model, vectorizer, etc.
                self.model = model_data.get('model')
                self.vectorizer = model_data.get('vectorizer')
                self.label_encoder = model_data.get('label_encoder')
                self.class_names = model_data.get('class_names')
            else:
                # If your pickle file contains just the model
                self.model = model_data
            
            print("Model loaded successfully!")
            
            # Load vectorizer separately if provided
            if vectorizer_path and os.path.exists(vectorizer_path):
                print(f"Loading vectorizer from {vectorizer_path}")
                with open(vectorizer_path, 'rb') as f:
                    self.vectorizer = pickle.load(f)
                print("Vectorizer loaded successfully!")
            
            # Load label encoder separately if provided
            if label_encoder_path and os.path.exists(label_encoder_path):
                print(f"Loading label encoder from {label_encoder_path}")
                with open(label_encoder_path, 'rb') as f:
                    self.label_encoder = pickle.load(f)
                print("Label encoder loaded successfully!")
                print(f"Label encoder classes: {self.label_encoder.classes_}")
            
            # Set class names from label encoder
            if not self.class_names and self.label_encoder and hasattr(self.label_encoder, 'classes_'):
                self.class_names = self.label_encoder.classes_.tolist()
                
            print(f"Available classes: {self.class_names}")
            
        except FileNotFoundError:
            print(f"Error: Model file '{model_path}' not found!")
            print("Please ensure your trained model pickle file is in the same directory as this script.")
            raise
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise
    
    import re
    def preprocess_text(self, text):
        """Preprocess text for classification"""
        if not text:
            return ""
        
        # Basic text preprocessing
        text = text.lower().strip()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
        text = re.sub(r'\@w+|\#','', text)
        text = re.sub(r"[^A-Za-z\s]", '', text)
        text = text.lower()
        
        return text
    
    def classify_email(self, subject, sender, body_preview):
        """
        Classify email using your trained model and return predicted tags with confidence scores
        """
        try:
            # Combine email text
            combined_text = f"{subject} {body_preview}"
            
            # Preprocess text
            processed_text = self.preprocess_text(combined_text)
            
            if not processed_text:
                return {}
            
            # Transform text using vectorizer if available
            if self.vectorizer:
                text_features = self.vectorizer.transform([processed_text[:200]])
            else:
                text_features = [processed_text[:200]]
            
            # Get predictions
            if hasattr(self.model, 'predict_proba'):
                # For models that support probability prediction
                probabilities = self.model.predict_proba(text_features)[0]
                
                # Create predictions dictionary
                predictions = {}
                for i, prob in enumerate(probabilities):
                    if i < len(self.class_names):
                        tag = self.class_names[i]
                        confidence = float(prob)
                        if confidence > 0.1:  # Only include predictions above 10% confidence
                            predictions[tag] = round(confidence, 3)
            else:
                # For models that only provide hard predictions
                prediction = self.model.predict(text_features)[0]
                
                # Convert prediction to tag name
                if self.label_encoder:
                    tag = self.label_encoder.inverse_transform([prediction])[0]
                elif isinstance(prediction, (int, np.integer)) and prediction < len(self.class_names):
                    tag = self.class_names[prediction]
                else:
                    tag = str(prediction)
                
                predictions = {tag: 1.0}  # 100% confidence for hard predictions
            
            # Sort by confidence and return top predictions
            sorted_predictions = dict(sorted(predictions.items(), key=lambda x: x[1], reverse=True))
            
            # Return top 5 predictions above 0.1 confidence
            filtered_predictions = {k: v for k, v in sorted_predictions.items() if v > 0.1}
            return dict(list(filtered_predictions.items())[:5])
            
        except Exception as e:
            print(f"Error in email classification: {str(e)}")
            return {}
    
    def get_top_tags(self, predictions, threshold=0.3):
        """Get top predicted tags above threshold"""
        x = [tag for tag, confidence in predictions.items() if confidence >= threshold]
        # print(predictions.items())
        if len(x)==0:
            x.append(list(predictions.items())[0][0])
        return x

# Initialize classifier
try:
    email_classifier = EmailClassifier(
        model_path=MODEL_FILE,
        vectorizer_path=VECTORIZER_FILE if os.path.exists(VECTORIZER_FILE) else None,
        label_encoder_path=LABEL_ENCODER_FILE if os.path.exists(LABEL_ENCODER_FILE) else None
    )
    print("Email classifier initialized successfully!")
except Exception as e:
    print(f"Failed to initialize email classifier: {e}")
    print("Please check that your model file exists and is in the correct format.")
    email_classifier = None

def get_credentials():
    """Get valid credentials from Flask session"""
    if 'credentials' not in session:
        return None

    try:
        creds = Credentials(**session['credentials'])

        if not creds.valid:
            if creds.expired and creds.refresh_token:
                creds.refresh(Request())
                # Update session with refreshed token
                session['credentials'] = {
                    'token': creds.token,
                    'refresh_token': creds.refresh_token,
                    'token_uri': creds.token_uri,
                    'client_id': creds.client_id,
                    'client_secret': creds.client_secret,
                    'scopes': creds.scopes
                }
            else:
                return None

        return creds
    except Exception as e:
        print(f"Error loading credentials from session: {e}")
        return None

def extract_email_body(payload):
    """Extract email body from Gmail API payload"""
    body_text = ""
    body_html = ""
    
    def process_part(part):
        nonlocal body_text, body_html
        mime_type = part.get('mimeType', '')
        
        if mime_type == 'text/plain':
            data = part['body'].get('data')
            if data:
                try:
                    body_text = base64.urlsafe_b64decode(data).decode('utf-8')
                except Exception:
                    pass
        elif mime_type == 'text/html':
            data = part['body'].get('data')
            if data:
                try:
                    body_html = base64.urlsafe_b64decode(data).decode('utf-8')
                except Exception:
                    pass
        elif mime_type.startswith('multipart/'):
            if 'parts' in part:
                for subpart in part['parts']:
                    process_part(subpart)
    
    # Check if payload has parts (multipart message)
    if 'parts' in payload:
        for part in payload['parts']:
            process_part(part)
    else:
        process_part(payload)
    
    # Return HTML if available, otherwise plain text
    if body_html:
        return body_html, 'html'
    elif body_text:
        return body_text, 'text'
    else:
        return "[No readable content]", 'text'

def extract_attachments(payload):
    """Extract attachment information from Gmail API payload"""
    attachments = []
    
    def process_part_for_attachments(part):
        if part.get('filename'):
            attachment = {
                'filename': part['filename'],
                'mime_type': part.get('mimeType', 'application/octet-stream'),
                'size': part['body'].get('size', 0),
                'attachment_id': part['body'].get('attachmentId')
            }
            attachments.append(attachment)
        
        if 'parts' in part:
            for subpart in part['parts']:
                process_part_for_attachments(subpart)
    
    if 'parts' in payload:
        for part in payload['parts']:
            process_part_for_attachments(part)
    
    return attachments

@app.route('/')
def index():
    creds = get_credentials()
    if not creds:
        return redirect(url_for('login'))
    return redirect(url_for('inbox'))

@app.route('/login')
def login():
    try:
        flow = Flow.from_client_config(
            {
                "web": {
                    "client_id": os.environ["GOOGLE_CLIENT_ID"],
                    "client_secret": os.environ["GOOGLE_CLIENT_SECRET"],
                    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                    "token_uri": "https://oauth2.googleapis.com/token",
                    "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
                    "redirect_uris": [url_for('callback', _external=True)]
                }
            },
            scopes=SCOPES,
            redirect_uri=url_for('callback', _external=True)
        )
        auth_url, state = flow.authorization_url(prompt='consent')
        
        # Store the state in session for security
        session['state'] = state
        
        return redirect(auth_url)
    except Exception as e:
        return f"Error during login setup: {e}"

@app.route('/callback')
def callback():
    try:
        # Verify state parameter for security
        if 'state' not in session:
            return "Error: Invalid session state"
        
        flow = Flow.from_client_config(
            {
                "web": {
                    "client_id": os.environ["GOOGLE_CLIENT_ID"],
                    "client_secret": os.environ["GOOGLE_CLIENT_SECRET"],
                    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                    "token_uri": "https://oauth2.googleapis.com/token",
                    "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
                    "redirect_uris": [url_for('callback', _external=True)]
                }
            },
            scopes=SCOPES,
            redirect_uri=url_for('callback', _external=True),
            state=session['state']
        )
        
        # Fetch the token using the authorization response
        flow.fetch_token(authorization_response=request.url)

        creds = flow.credentials
        
        # 🔐 Save credentials to session (not file)
        session['credentials'] = {
            'token': creds.token,
            'refresh_token': creds.refresh_token,
            'token_uri': creds.token_uri,
            'client_id': creds.client_id,
            'client_secret': creds.client_secret,
            'scopes': creds.scopes
        }
        # Clear session state
        session.pop('state', None)
        
        return redirect(url_for('inbox'))
    except Exception as e:
        return f"Error during callback: {e}"

@app.route('/inbox')
def inbox():
    creds = get_credentials()
    if not creds:
        return redirect(url_for('login'))

    if not email_classifier:
        return "Error: Email classifier not initialized. Please check your model file."

    try:
        service = build('gmail', 'v1', credentials=creds)
        
        # Get list of messages
        results = service.users().messages().list(userId='me', maxResults=50).execute()
        messages = results.get('messages', [])

        email_list = []

        for msg in messages:
            try:
                # Get message details
                msg_data = service.users().messages().get(
                    userId='me', 
                    id=msg['id'], 
                    format='full'
                ).execute()

                profile = service.users().getProfile(userId='me').execute()
                user_email = profile['emailAddress']

                
                headers = msg_data['payload']['headers']
                subject = sender = date = "N/A"

                # Extract headers
                for header in headers:
                    if header['name'] == 'Subject':
                        subject = header['value']
                    elif header['name'] == 'From':
                        sender = header['value']
                    elif header['name'] == 'Date':
                        date = header['value']

                # Extract body preview
                body, body_type = extract_email_body(msg_data['payload'])
                if body_type == 'html':
                    soup = BeautifulSoup(body, 'html.parser')
                    body_preview = soup.get_text()
                else:
                    body_preview = body
                
                # Extract attachments
                attachments = extract_attachments(msg_data['payload'])

                # Classify email using your AI model
                predictions = email_classifier.classify_email(subject, sender, body_preview)
                top_tags = email_classifier.get_top_tags(predictions)

                email_list.append({
                    'subject': subject,
                    'from': sender,
                    'date': date,
                    'body_preview': body_preview[:200] + '...' if len(body_preview) > 200 else body_preview,
                    'id': msg['id'],
                    'has_attachments': len(attachments) > 0,
                    'attachment_count': len(attachments),
                    'predicted_tags': top_tags,
                    'confidence_scores': predictions
                })
            except Exception as e:
                print(f"Error processing message {msg['id']}: {e}")
                continue

        return render_template('inbox.html', emails=email_list,user_email=user_email)
    
    except Exception as e:
        return f"Error accessing Gmail: {e}. Please try logging in again."

@app.route('/api/email/<email_id>')
def get_full_email(email_id):
    """API endpoint to get full email content with classification"""
    creds = get_credentials()
    if not creds:
        return jsonify({'error': 'Not authenticated'}), 401

    if not email_classifier:
        return jsonify({'error': 'Email classifier not available'}), 500

    try:
        service = build('gmail', 'v1', credentials=creds)
        
        # Get full message
        msg_data = service.users().messages().get(
            userId='me', 
            id=email_id, 
            format='full'
        ).execute()
        
        headers = msg_data['payload']['headers']
        subject = sender = date = "N/A"

        # Extract headers
        for header in headers:
            if header['name'] == 'Subject':
                subject = header['value']
            elif header['name'] == 'From':
                sender = header['value']
            elif header['name'] == 'Date':
                date = header['value']

        # Extract full body
        body, body_type = extract_email_body(msg_data['payload'])
        # print(body)
        
        # Extract attachments
        attachments = extract_attachments(msg_data['payload'])
        
        # Generate classification in real-time
        if body_type == 'html':
            soup = BeautifulSoup(body, 'html.parser')
            body_text = soup.get_text()
        else:
            body_text = body
        
        # print(body_text)
        predictions = email_classifier.classify_email(subject, sender, body_text)
        top_tags = email_classifier.get_top_tags(predictions)

        classification = {
            'predicted_tags': top_tags,
            'confidence_scores': predictions
        }
        # print(classification)

        return jsonify({
            'subject': subject,
            'from': sender,
            'date': date,
            'body': body,
            'body_type': body_type,
            'attachments': attachments,
            'classification': classification
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/classify/<email_id>', methods=['POST'])
def reclassify_email(email_id):
    """API endpoint to reclassify a specific email in real-time"""
    creds = get_credentials()
    if not creds:
        return jsonify({'error': 'Not authenticated'}), 401

    if not email_classifier:
        return jsonify({'error': 'Email classifier not available'}), 500

    try:
        service = build('gmail', 'v1', credentials=creds)
        
        # Get message
        msg_data = service.users().messages().get(
            userId='me', 
            id=email_id, 
            format='full'
        ).execute()
        
        headers = msg_data['payload']['headers']
        subject = sender = "N/A"

        for header in headers:
            if header['name'] == 'Subject':
                subject = header['value']
            elif header['name'] == 'From':
                sender = header['value']

        # Extract body
        body, body_type = extract_email_body(msg_data['payload'])
        if body_type == 'html':
            soup = BeautifulSoup(body, 'html.parser')
            body_text = soup.get_text()
        else:
            body_text = body

        # Reclassify using your model
        predictions = email_classifier.classify_email(subject, sender, body_text)
        top_tags = email_classifier.get_top_tags(predictions)

        return jsonify({
            'predicted_tags': top_tags,
            'confidence_scores': predictions
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/attachment/<email_id>/<attachment_id>')
def download_attachment(email_id, attachment_id):
    """API endpoint to download attachments"""
    creds = get_credentials()
    if not creds:
        return jsonify({'error': 'Not authenticated'}), 401

    try:
        service = build('gmail', 'v1', credentials=creds)
        
        # Get attachment
        attachment = service.users().messages().attachments().get(
            userId='me',
            messageId=email_id,
            id=attachment_id
        ).execute()
        
        file_data = base64.urlsafe_b64decode(attachment['data'])
        
        return file_data, 200, {
            'Content-Type': 'application/octet-stream',
            'Content-Disposition': 'attachment'
        }
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/logout')
def logout():
    """Logout and clear stored credentials from session"""
    session.pop('credentials', None)
    session.clear()
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)