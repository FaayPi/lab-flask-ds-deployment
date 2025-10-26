from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import pandas as pd
import os
from sklearn.feature_extraction.text import CountVectorizer
from gensim.models import Word2Vec
import re

app = Flask(__name__)

# ===== MODEL LADEN =====
print("Loading model...")
MODEL_PATH = 'final_xgboost_model.pkl'

try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    print(f"✓ Model successfully loaded from {MODEL_PATH}")
    print(f"Model type: {type(model)}")
except FileNotFoundError:
    print(f"ERROR: Model file not found at {MODEL_PATH}")
    print("Please make sure the model file exists!")
    model = None
except Exception as e:
    print(f"ERROR loading model: {e}")
    model = None


# ===== PREPROCESSING FUNCTIONS =====

def clean_text_comprehensive(text):
    """
    Umfassende Text-Bereinigung (gleiche Funktion wie im Training)
    """
    if pd.isna(text) or text == "":
        return ""
    
    # 1. Lowercase
    text = text.lower()
    
    # 2. Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # 3. Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text, flags=re.IGNORECASE)
    
    # 4. Remove mentions
    text = re.sub(r'@\w+', '', text)
    
    # 5. Contractions
    contractions = {
        "don't": "do not", "can't": "cannot", "won't": "will not", "n't": " not",
        "i'm": "i am", "you're": "you are", "he's": "he is", "she's": "she is",
        "it's": "it is", "we're": "we are", "they're": "they are",
        "i've": "i have", "you've": "you have", "we've": "we have",
        "i'll": "i will", "you'll": "you will", "he'll": "he will",
        "i'd": "i would", "you'd": "you would", "he'd": "he would"
    }
    for contraction, expansion in contractions.items():
        text = re.sub(r'\b' + contraction + r'\b', expansion, text)
    
    # 6. Remove numbers
    text = re.sub(r'\d+', '', text)
    
    # 7. Remove special characters
    text = re.sub(r'[^\w\s]', '', text)
    
    # 8. Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # 9. Remove stopwords (basic)
    basic_stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 
                       'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 
                       'were', 'be', 'been', 'being', 'have', 'has', 'had'}
    words = text.split()
    words = [word for word in words if word not in basic_stop_words]
    text = ' '.join(words)
    
    return text


def extract_features(title, text):
    """
    Extrahiert alle Features aus Title und Text
    """
    # Kombiniere Title und Text
    title_text_combined = f"{title} {text}"
    
    # Feature 1: Text Length
    text_length = len(text)
    
    # Feature 2: Uppercase Percentage
    total_chars = len(title_text_combined)
    uppercase_chars = sum(1 for char in title_text_combined if char.isupper())
    uppercase_percentage = (uppercase_chars / total_chars * 100) if total_chars > 0 else 0.0
    
    # Feature 3: Punctuation Count
    punctuation_count = title_text_combined.count('!') + title_text_combined.count('?')
    
    # Feature 4: URL Count
    url_pattern = re.compile(r'https?://\S+|www\.\S+', re.IGNORECASE)
    url_count = len(url_pattern.findall(title_text_combined))
    
    # Feature 5: Mention Count
    mention_pattern = re.compile(r'@\w+')
    mention_count = len(mention_pattern.findall(title_text_combined))
    
    # Feature 6: Cleaned Text
    title_text_cleaned = clean_text_comprehensive(title_text_combined)
    
    features = {
        'text_length': text_length,
        'uppercase_percentage': uppercase_percentage,
        'punctuation_count': punctuation_count,
        'url_count': url_count,
        'mention_count': mention_count,
        'title_text_cleaned': title_text_cleaned
    }
    
    return features


def create_feature_vector(features, expected_feature_count=1105):
    """
    Erstellt einen Feature-Vektor mit allen benötigten Features
    HINWEIS: Dies ist eine vereinfachte Version. Im Produktiv-System müsstest du
    den gleichen Vectorizer und Word2Vec Model wie beim Training verwenden.
    """
    # Basis-Features
    feature_vector = [
        features['text_length'],
        features['uppercase_percentage'],
        features['punctuation_count'],
        features['url_count'],
        features['mention_count']
    ]
    
    # Fülle mit Nullen auf (Platzhalter für N-grams und Word2Vec Features)
    # In einer echten Implementierung würdest du hier die gleichen Vectorizer
    # und Word2Vec Models wie beim Training verwenden
    remaining_features = expected_feature_count - len(feature_vector)
    feature_vector.extend([0] * remaining_features)
    
    return feature_vector


# ===== FLASK ROUTES =====

@app.route('/')
def home():
    """
    Home Page mit Eingabeformular
    """
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """
    Prediction Endpoint
    Erwartet JSON mit 'title' und 'text' Feldern
    """
    try:
        # Prüfe ob Model geladen wurde
        if model is None:
            return jsonify({
                'error': 'Model not loaded. Please check server logs.',
                'success': False
            }), 500
        
        # Hole Daten aus Request
        data = request.get_json()
        
        # Validierung
        if not data:
            return jsonify({
                'error': 'No data provided. Please send JSON data.',
                'success': False
            }), 400
        
        if 'title' not in data or 'text' not in data:
            return jsonify({
                'error': 'Missing required fields. Please provide both "title" and "text".',
                'success': False
            }), 400
        
        title = data['title']
        text = data['text']
        
        # Validiere Eingaben
        if not title or not text:
            return jsonify({
                'error': 'Title and text cannot be empty.',
                'success': False
            }), 400
        
        if len(title) > 1000 or len(text) > 10000:
            return jsonify({
                'error': 'Title or text too long. Maximum length: Title 1000 chars, Text 10000 chars.',
                'success': False
            }), 400
        
        # Feature Extraction
        print(f"\nProcessing prediction request...")
        print(f"Title: {title[:50]}...")
        print(f"Text: {text[:50]}...")
        
        features = extract_features(title, text)
        feature_vector = create_feature_vector(features)
        
        # Konvertiere zu numpy array und reshape für Prediction
        X = np.array([feature_vector])
        
        # Mache Prediction
        prediction = model.predict(X)[0]
        prediction_proba = model.predict_proba(X)[0]
        
        # Bestimme Confidence
        confidence = float(max(prediction_proba))
        
        # Label Mapping (anpassen an deine Klassen)
        label_map = {
            0: 'Fake News',
            1: 'Real News'
        }
        
        predicted_label = label_map.get(prediction, f'Class {prediction}')
        
        print(f"✓ Prediction: {predicted_label} (Confidence: {confidence:.2%})")
        
        # Response
        response = {
            'success': True,
            'prediction': int(prediction),
            'predicted_label': predicted_label,
            'confidence': confidence,
            'probabilities': {
                'fake': float(prediction_proba[0]),
                'real': float(prediction_proba[1])
            },
            'features': {
                'text_length': features['text_length'],
                'uppercase_percentage': round(features['uppercase_percentage'], 2),
                'punctuation_count': features['punctuation_count'],
                'url_count': features['url_count'],
                'mention_count': features['mention_count'],
                'cleaned_text_preview': features['title_text_cleaned'][:100]
            }
        }
        
        return jsonify(response), 200
        
    except ValueError as e:
        return jsonify({
            'error': f'Invalid input data: {str(e)}',
            'success': False
        }), 400
    
    except Exception as e:
        print(f"ERROR during prediction: {e}")
        return jsonify({
            'error': f'Internal server error: {str(e)}',
            'success': False
        }), 500


@app.route('/health', methods=['GET'])
def health():
    """
    Health Check Endpoint
    """
    model_status = "loaded" if model is not None else "not loaded"
    
    return jsonify({
        'status': 'healthy',
        'model_status': model_status,
        'model_type': str(type(model)) if model else None
    }), 200


@app.route('/api/info', methods=['GET'])
def api_info():
    """
    API Information Endpoint
    """
    return jsonify({
        'api_version': '1.0',
        'model': 'XGBoost Classifier',
        'endpoints': {
            '/': 'Home page',
            '/predict': 'POST - Make predictions',
            '/health': 'GET - Health check',
            '/api/info': 'GET - API information'
        },
        'required_fields': ['title', 'text'],
        'example_request': {
            'title': 'Breaking News: Example Title',
            'text': 'This is an example news article text...'
        }
    }), 200


# ===== ERROR HANDLERS =====

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'error': 'Endpoint not found',
        'success': False
    }), 404


@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({
        'error': 'Method not allowed',
        'success': False
    }), 405


@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'error': 'Internal server error',
        'success': False
    }), 500


# ===== START SERVER =====

if __name__ == '__main__':
    print("\n" + "="*50)
    print("STARTING FLASK SERVER")
    print("="*50)
    print(f"Model Status: {'✓ Loaded' if model else '✗ Not Loaded'}")
    print(f"Server starting on http://127.0.0.1:8080")
    print("="*50 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=8080)