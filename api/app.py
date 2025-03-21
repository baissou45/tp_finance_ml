from flask import Flask, request, jsonify, render_template
import os
import json
from datetime import datetime, timedelta
from tensorflow.keras.models import load_model
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf

app = Flask(__name__)

# Charger le modèle Tesla et son scaler
model = load_model('../tesla_model/lstm_model.keras')
scaler = joblib.load('../tesla_model/scaler.joblib')

# Charger le modèle Bitcoin et son scaler (s'ils existent)
try:
    btc_model = load_model('../bitcoin_model/lstm_model.keras')
    btc_scaler = joblib.load('../bitcoin_model/scaler.joblib')
    BTC_MODEL_AVAILABLE = True
    # Date de fin d'entraînement pour le modèle Bitcoin
    BTC_TRAINING_END_DATE = datetime(2025, 3, 15).date()  # À ajuster selon votre modèle
except (OSError, IOError):
    BTC_MODEL_AVAILABLE = False
    print("Modèle Bitcoin non disponible. L'endpoint /predict/btc ne sera pas fonctionnel.")

# Fichiers pour stocker l'historique des prédictions
HISTORY_FILE = 'prediction_history.json'
BTC_HISTORY_FILE = 'btc_prediction_history.json'

# Date de fin d'entraînement (à définir selon votre modèle)
# Le modèle a été entraîné jusqu'au 15/03/2025
TRAINING_END_DATE = datetime(2025, 3, 15).date()

def get_last_training_sequence():
    """
    Récupère la dernière séquence utilisée pour l'entraînement
    """
    lookback = 60
    end_date = TRAINING_END_DATE
    start_date = end_date - timedelta(days=lookback + 30)
    
    tesla = yf.Ticker("TSLA")
    data = tesla.history(start=start_date, end=end_date + timedelta(days=1))
    
    if len(data) < lookback:
        raise ValueError(f"Pas assez de données. Nécessaire: {lookback}, Obtenu: {len(data)}")
    
    sequence = data['Close'].tail(lookback).values
    last_price = data['Close'].iloc[-1]
    
    return sequence, last_price

def predict_future_price(target_date):
    """
    Prédit le prix de Tesla à une date future spécifique
    """
    # Obtenir la séquence de fin d'entraînement
    sequence, last_training_price = get_last_training_sequence()
    
    # Calculer le nombre de jours ouvrables entre la fin d'entraînement et la date cible
    target = pd.to_datetime(target_date).date()
    
    # Approximation des jours de trading (excluant weekends)
    days_diff = np.busday_count(TRAINING_END_DATE, target)
    
    if days_diff <= 0:
        return last_training_price, 0, TRAINING_END_DATE
    
    # Normaliser la séquence initiale
    scaled_sequence = scaler.transform(sequence.reshape(-1, 1)).flatten()
    
    # Faire des prédictions itératives
    current_sequence = scaled_sequence.copy()
    
    for _ in range(days_diff):
        # Préparer l'entrée pour le modèle (batch_size, time_steps, features)
        X = current_sequence[-60:].reshape(1, 60, 1)
        
        # Prédire la prochaine valeur
        next_scaled = model.predict(X, verbose=0)[0][0]
        
        # Ajouter à la séquence
        current_sequence = np.append(current_sequence, next_scaled)
    
    # Dénormaliser la dernière prédiction
    predicted_price = scaler.inverse_transform([[current_sequence[-1]]])[0][0]
    
    return predicted_price, days_diff, TRAINING_END_DATE


def get_btc_last_training_sequence():
    """
    Récupère la dernière séquence utilisée pour l'entraînement du modèle Bitcoin
    """
    lookback = 60
    end_date = BTC_TRAINING_END_DATE
    start_date = end_date - timedelta(days=lookback + 30)
    
    bitcoin = yf.Ticker("BTC-USD")
    data = bitcoin.history(start=start_date, end=end_date + timedelta(days=1))
    
    if len(data) < lookback:
        raise ValueError(f"Pas assez de données. Nécessaire: {lookback}, Obtenu: {len(data)}")
    
    sequence = data['Close'].tail(lookback).values
    last_price = data['Close'].iloc[-1]
    
    return sequence, last_price


def predict_btc_future_price(target_date):
    """
    Prédit le prix du Bitcoin à une date future spécifique
    """
    if not BTC_MODEL_AVAILABLE:
        raise ValueError("Le modèle Bitcoin n'est pas disponible.")
        
    # Obtenir la séquence de fin d'entraînement
    sequence, last_training_price = get_btc_last_training_sequence()
    
    # Calculer le nombre de jours entre la fin d'entraînement et la date cible
    # Pour les crypto-monnaies, on compte tous les jours (24/7)
    target = pd.to_datetime(target_date).date()
    days_diff = (target - BTC_TRAINING_END_DATE).days
    
    if days_diff <= 0:
        return last_training_price, 0, BTC_TRAINING_END_DATE
    
    # Normaliser la séquence initiale
    scaled_sequence = btc_scaler.transform(sequence.reshape(-1, 1)).flatten()
    
    # Faire des prédictions itératives
    current_sequence = scaled_sequence.copy()
    
    for _ in range(days_diff):
        # Préparer l'entrée pour le modèle (batch_size, time_steps, features)
        X = current_sequence[-60:].reshape(1, 60, 1)
        
        # Prédire la prochaine valeur
        next_scaled = btc_model.predict(X, verbose=0)[0][0]
        
        # Ajouter à la séquence
        current_sequence = np.append(current_sequence, next_scaled)
    
    # Dénormaliser la dernière prédiction
    predicted_price = btc_scaler.inverse_transform([[current_sequence[-1]]])[0][0]
    
    return predicted_price, days_diff, BTC_TRAINING_END_DATE

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        target_date = data.get('date')
        
        if not target_date:
            return jsonify({'error': 'Date non spécifiée'}), 400
            
        # Prédire le prix à la date cible
        predicted_price, days_ahead, training_end = predict_future_price(target_date)
        
        # Récupérer le prix actuel pour comparaison
        current_price = yf.Ticker("TSLA").history(period="1d")['Close'].iloc[-1]
        
        # Sauvegarder la prédiction dans l'historique
        save_prediction(target_date, predicted_price, current_price)
        
        return jsonify({
            'date': target_date,
            'training_end_date': training_end.strftime('%Y-%m-%d'),
            'days_from_training': int(days_ahead),
            'predicted_price': float(predicted_price),
            'current_price': float(current_price),
            'prediction_vs_current': {
                'change': float(predicted_price - current_price),
                'change_percent': float((predicted_price - current_price) / current_price * 100)
            },
            'currency': 'USD'
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'type': str(type(e).__name__),
            'traceback': str(e.__traceback__)
        }), 500

@app.route('/predict/btc', methods=['POST'])
def predict_btc():
    try:
        if not BTC_MODEL_AVAILABLE:
            return jsonify({
                'error': 'Le modèle Bitcoin n\'est pas disponible.',
                'details': 'Veuillez exécuter le notebook btc_lstm.ipynb pour créer le modèle.'
            }), 503
            
        data = request.get_json()
        target_date = data.get('date')
        
        if not target_date:
            return jsonify({'error': 'Date non spécifiée'}), 400
            
        # Prédire le prix à la date cible
        predicted_price, days_ahead, training_end = predict_btc_future_price(target_date)
        
        # Récupérer le prix actuel pour comparaison
        current_price = yf.Ticker("BTC-USD").history(period="1d")['Close'].iloc[-1]
        
        # Sauvegarder la prédiction dans l'historique
        save_btc_prediction(target_date, predicted_price, current_price)
        
        return jsonify({
            'date': target_date,
            'training_end_date': training_end.strftime('%Y-%m-%d'),
            'days_from_training': int(days_ahead),
            'predicted_price': float(predicted_price),
            'current_price': float(current_price),
            'prediction_vs_current': {
                'change': float(predicted_price - current_price),
                'change_percent': float((predicted_price - current_price) / current_price * 100)
            },
            'currency': 'USD'
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'type': str(type(e).__name__),
            'traceback': str(e.__traceback__)
        }), 500

@app.route('/model-info', methods=['GET'])
def model_info():
    info = {
        'tesla': {
            'model_type': 'LSTM',
            'training_end_date': TRAINING_END_DATE.strftime('%Y-%m-%d'),
            'lookback_period': 60,
            'performance': {
                'test_r2': 0.954,  # Selon vos résultats précédents
                'test_rmse': 17.28
            }
        }
    }
    
    if BTC_MODEL_AVAILABLE:
        info['bitcoin'] = {
            'model_type': 'LSTM',
            'training_end_date': BTC_TRAINING_END_DATE.strftime('%Y-%m-%d'),
            'lookback_period': 60,
            'performance': {
                'test_r2': 0.92,  # À ajuster selon les résultats réels
                'test_rmse': 1500.0  # À ajuster selon les résultats réels
            }
        }
    
    return jsonify(info)

def save_prediction(date, predicted_price, current_price):
    """
    Sauvegarde une prédiction dans l'historique
    """
    # Charger l'historique existant
    history = []
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, 'r') as f:
                history = json.load(f)
        except json.JSONDecodeError:
            # Si le fichier est corrompu, on repart de zéro
            history = []
    
    # Ajouter la nouvelle prédiction
    prediction = {
        'date': date,
        'predicted_price': float(predicted_price),
        'current_price': float(current_price),
        'timestamp': datetime.now().isoformat(),
        'actual_price': None  # Sera mis à jour plus tard si la date est passée
    }
    
    history.append(prediction)
    
    # Mettre à jour les prix réels pour les prédictions passées
    update_actual_prices(history)
    
    # Sauvegarder l'historique
    with open(HISTORY_FILE, 'w') as f:
        json.dump(history, f)

def update_actual_prices(history):
    """
    Met à jour les prix réels pour les prédictions passées
    """
    today = datetime.now().date()
    tesla = yf.Ticker("TSLA")
    
    for prediction in history:
        # Si la prédiction a une date passée et n'a pas encore de prix réel
        pred_date = pd.to_datetime(prediction['date']).date()
        if pred_date < today and prediction['actual_price'] is None:
            # Récupérer le prix réel
            try:
                # Récupérer les données pour cette date
                data = tesla.history(start=pred_date, end=pred_date + timedelta(days=1))
                if not data.empty:
                    prediction['actual_price'] = float(data['Close'].iloc[0])
            except Exception as e:
                print(f"Erreur lors de la récupération du prix réel: {e}")


def update_btc_actual_prices(history):
    """
    Met à jour les prix réels pour les prédictions Bitcoin passées
    """
    today = datetime.now().date()
    bitcoin = yf.Ticker("BTC-USD")
    
    for prediction in history:
        # Si la prédiction a une date passée et n'a pas encore de prix réel
        pred_date = pd.to_datetime(prediction['date']).date()
        if pred_date < today and prediction['actual_price'] is None:
            # Récupérer le prix réel
            try:
                # Récupérer les données pour cette date
                data = bitcoin.history(start=pred_date, end=pred_date + timedelta(days=1))
                if not data.empty:
                    prediction['actual_price'] = float(data['Close'].iloc[0])
            except Exception as e:
                print(f"Erreur lors de la récupération du prix réel Bitcoin: {e}")


def save_btc_prediction(date, predicted_price, current_price):
    """
    Sauvegarde une prédiction Bitcoin dans l'historique
    """
    # Charger l'historique existant
    history = []
    if os.path.exists(BTC_HISTORY_FILE):
        try:
            with open(BTC_HISTORY_FILE, 'r') as f:
                history = json.load(f)
        except json.JSONDecodeError:
            # Si le fichier est corrompu, on repart de zéro
            history = []
    
    # Ajouter la nouvelle prédiction
    prediction = {
        'date': date,
        'predicted_price': float(predicted_price),
        'current_price': float(current_price),
        'timestamp': datetime.now().isoformat(),
        'actual_price': None  # Sera mis à jour plus tard si la date est passée
    }
    
    history.append(prediction)
    
    # Mettre à jour les prix réels pour les prédictions passées
    update_btc_actual_prices(history)
    
    # Sauvegarder l'historique
    with open(BTC_HISTORY_FILE, 'w') as f:
        json.dump(history, f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/history')
def history():
    return render_template('history.html')

@app.route('/prediction-history')
def prediction_history():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, 'r') as f:
                history = json.load(f)
                # Mettre à jour les prix réels
                update_actual_prices(history)
                # Sauvegarder les mises à jour
                with open(HISTORY_FILE, 'w') as f:
                    json.dump(history, f)
                return jsonify({'predictions': history})
        except Exception as e:
            return jsonify({'error': str(e), 'predictions': []})
    else:
        return jsonify({'predictions': []})

@app.route('/prediction-history/btc')
def btc_prediction_history():
    if not BTC_MODEL_AVAILABLE:
        return jsonify({
            'error': 'Le modèle Bitcoin n\'est pas disponible.',
            'details': 'Veuillez exécuter le notebook btc_lstm.ipynb pour créer le modèle.',
            'predictions': []
        })
        
    if os.path.exists(BTC_HISTORY_FILE):
        try:
            with open(BTC_HISTORY_FILE, 'r') as f:
                history = json.load(f)
                # Mettre à jour les prix réels
                update_btc_actual_prices(history)
                # Sauvegarder les mises à jour
                with open(BTC_HISTORY_FILE, 'w') as f:
                    json.dump(history, f)
                return jsonify({'predictions': history})
        except Exception as e:
            return jsonify({'error': str(e), 'predictions': []})
    else:
        return jsonify({'predictions': []})

if __name__ == '__main__':
    app.run(debug=True, port=5001)