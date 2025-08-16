from flask import Flask, request, render_template, jsonify,session,url_for,redirect
import numpy as np
import librosa
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import os
from twilio.rest import Client
from werkzeug.utils import secure_filename
from flask import send_file
from reportlab.pdfgen import canvas

import io

import firebase_admin
from firebase_admin import credentials, firestore

cred = credentials.Certificate("fake-audio-detection-firebase-adminsdk-fbsvc-cf4f2171c6.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

# Twilio config
TWILIO_ACCOUNT_SID = 'Your_account_sid'
TWILIO_AUTH_TOKEN = 'Your_auth_token'
TWILIO_FROM_NUMBER = 'Your_number'  
client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'supersecretkey'  
#users = {'admin': 'admin123'}
# Load trained LSTM model
model = load_model('train_audio (1).h5')

# Audio processing parameters
MAX_LENGTH = 100
N_MFCC = 40

# ======================== 1. Route: Serve Frontend ========================

@app.route('/')

def home():
    if 'username' in session: 
        return render_template('index.html')
    return redirect(url_for('login'))  

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        user_ref = db.collection('users').document(username).get()
        if user_ref.exists:
            user_data = user_ref.to_dict()
            if user_data['password'] == password:
                session['username'] = username
                session['email'] = user_data['email']
                session['phone'] = user_data['phone']
                session['age'] = user_data['age']
                return redirect(url_for('home'))
            else:
                session['message'] = 'Incorrect password.'
                return redirect(url_for('login'))
        else:
            session['message'] = 'Account not found. Please sign up first.'
            return redirect(url_for('login'))
    
    message = session.pop('message', '')  # Get flash message if any
    return render_template('login.html', message=message)



def index():
    return render_template('index.html')

@app.route('/about.html')
def about():
    if 'username' in session: 
        return render_template('about.html')
    return redirect(url_for('login'))  # returns HTML content to be fetched

@app.route('/account.html')
def account():
    if 'username' in session:  # Protect route, only logged-in users can access
        return render_template('account.html')
    return redirect(url_for('login'))

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        email = request.form['email']
        phone = request.form['phone']
        age = request.form['age']

        # Store the user data in the users dictionary (or Firebase)
        db.collection('users').document(username).set({
            'username': username,
            'password': password,
            'email': email,
            'phone': phone,
            'age': age
        })

        # Redirect to login page with success message
        session['message'] = 'Account created successfully! Please log in.'
        return redirect(url_for('login'))

    return render_template('signup.html')


@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))


# ======================== 2. Route: Predict Audio ========================
@app.route('/predict', methods=['POST'])
def predict():
    if 'audio' not in request.files:
        return jsonify({'error': 'No file uploaded'})

    audio_file = request.files['audio']
    if audio_file.filename == '':
        return jsonify({'error': 'No selected file'})

    # Save the uploaded file temporarily
    temp_path = 'temp_audio.wav'
    audio_file.save(temp_path)

    # Process the audio and make prediction
    audio, sr = librosa.load(temp_path, sr=None)

    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC).T
    
    # Pad or truncate the MFCC to match model input size
    if len(mfcc) > MAX_LENGTH:
        mfcc = mfcc[:MAX_LENGTH]
    mfcc = pad_sequences([mfcc], maxlen=MAX_LENGTH, dtype='float32', padding='post', truncating='post')
     # Predict using the model
    prediction = model.predict(mfcc)[0][0]
    label = 'Real' if prediction > 0.5 else 'Fake'
    confidence = float(prediction) if label == 'Real' else 1 - float(prediction)

    # Clean up temp audio
    os.remove(temp_path)
    
    return jsonify({'label': label, 'confidence': round(confidence, 4)})
    
@app.route('/send-report', methods=['POST'])
def send_report():
    if 'audio' not in request.files:
        return jsonify({'success': False, 'error': 'Audio file missing'}), 400

    phone = request.form.get('phone')
    label = request.form.get('label')
    confidence = request.form.get('confidence')
    audio = request.files['audio']

    if not all([phone, label, confidence]):
        return jsonify({'success': False, 'error': 'Missing data'}), 400

    # Save to static/uploads
    filename = secure_filename(audio.filename)
    filepath = os.path.join('static/uploads', filename)
    os.makedirs('static/uploads', exist_ok=True)
    audio.save(filepath)

    # Create public URL for the audio
    #audio_url = f"http://127.0.0.1:5000/static/uploads/{filename}"
    ngrok_base_url = "https://5110-43-225-26-110.ngrok-free.app"  # change if ngrok restarts
    audio_url = f"{ngrok_base_url}/static/uploads/{filename}"

    #https://e83e-2406-b400-b5-ae6e-218a-afaa-41be-5cd7.ngrok-free.app

    message_body = (
    f"Audio Report\n"
    f"{label} Audio ({float(confidence) * 100:.1f}%)\n"
    f"Listen: {audio_url}"
)

    try:
        message = client.messages.create(
            from_=TWILIO_FROM_NUMBER,
            to=phone,
            body=message_body
        )
        session['report_sent'] = True
        return jsonify({'success': True, 'sid': message.sid})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500
@app.route('/download-report', methods=['POST'])
def download_report():
    if not session.get('report_sent'):  
        return jsonify({'success': False, 'error': 'Please send the report before downloading.'}), 403
    phone = request.form.get('phone', 'Unknown')  
    label = request.form.get('label', 'No Label') 
    confidence = request.form.get('confidence', 'N/A')  
    filename = request.form.get('filename', 'uploaded_audio.wav')  
    audio_url = request.form.get('audio_url', '') 
    if not all([phone, label, confidence]):  
        return jsonify({'success': False, 'error': 'Missing data'}), 400
    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer)
    pdf.setFont("Helvetica", 12)

    # Content
    pdf.drawString(100, 750, "🧠 Fake Audio Detection Report")
    pdf.drawString(100, 720, f"📁 File: {filename}")
    pdf.drawString(100, 700, f"🎙 Prediction: {label}")
    pdf.drawString(100, 680, f"📊 Confidence: {float(confidence)*100:.2f}%")
    pdf.drawString(100, 660, f"📞 Reported To: {phone}")
    if audio_url:  
        link_text = f"🎧 Listen: {audio_url}"
        pdf.drawString(100, 640, link_text)
        pdf.linkURL(audio_url, (100, 638, 500, 652), relative=0)
    else:
        pdf.drawString(100, 640, "🎧 No audio link provided.")

    pdf.save()
    session.pop('report_sent', None)
    buffer.seek(0)

    return send_file(buffer, as_attachment=True, download_name="Fake_Audio_Report.pdf", mimetype='application/pdf')
# ======================== 3. Run the Flask App ========================
if __name__ == '__main__':
    app.run(debug=True)

