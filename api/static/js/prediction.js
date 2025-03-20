// Script pour l'application de prédiction Tesla

document.addEventListener('DOMContentLoaded', function() {
    // Charger les informations du modèle
    loadModelInfo();
    
    // Définir la date minimale à aujourd'hui
    const today = new Date('2025-03-15');
    const formattedDate = today.toISOString().split('T')[0];
    document.getElementById('target-date').min = formattedDate;
    
    // Gérer le formulaire de prédiction
    const form = document.getElementById('prediction-form');
    form.addEventListener('submit', handlePredictionSubmit);
});

function loadModelInfo() {
    fetch('/model-info')
        .then(response => response.json())
        .then(data => {
            document.getElementById('model-type').textContent = data.model_type;
            document.getElementById('training-end-date').textContent = data.training_end_date;
            document.getElementById('lookback-period').textContent = data.lookback_period;
            document.getElementById('r2-score').textContent = data.performance.test_r2;
            document.getElementById('rmse-score').textContent = data.performance.test_rmse;
        })
        .catch(error => {
            console.error('Erreur lors du chargement des informations du modèle:', error);
            showError('Impossible de charger les informations du modèle.');
        });
}

function handlePredictionSubmit(e) {
    e.preventDefault();
    
    const targetDate = document.getElementById('target-date').value;
    
    if (!targetDate) {
        showError('Veuillez sélectionner une date.');
        return;
    }
    
    // Afficher le chargement
    showLoading(true);
    
    // Appeler l'API
    fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ date: targetDate })
    })
    .then(response => {
        if (!response.ok) {
            return response.json().then(err => { throw err; });
        }
        return response.json();
    })
    .then(data => {
        showLoading(false);
        displayResults(data);
        createChart(data);
    })
    .catch(error => {
        showLoading(false);
        showError(error.error || 'Une erreur est survenue lors de la prédiction.');
        console.error('Erreur:', error);
    });
}

function showLoading(isLoading) {
    const loadingElement = document.getElementById('loading');
    const resultsElement = document.getElementById('prediction-results');
    const errorElement = document.getElementById('error-message');
    
    if (isLoading) {
        loadingElement.classList.remove('d-none');
        resultsElement.classList.add('d-none');
        errorElement.classList.add('d-none');
    } else {
        loadingElement.classList.add('d-none');
    }
}

function showError(message) {
    const errorElement = document.getElementById('error-message');
    errorElement.classList.remove('d-none');
    errorElement.textContent = message;
}

function displayResults(data) {
    document.getElementById('prediction-results').classList.remove('d-none');
    
    // Formater la date
    const formattedDate = new Date(data.date).toLocaleDateString('fr-FR', {
        year: 'numeric',
        month: 'long',
        day: 'numeric'
    });
    
    // Afficher les résultats
    document.getElementById('prediction-date').textContent = formattedDate;
    document.getElementById('current-price').textContent = data.current_price.toFixed(2);
    document.getElementById('predicted-price').textContent = data.predicted_price.toFixed(2);
    document.getElementById('days-from-training').textContent = data.days_from_training;
    
    const change = data.prediction_vs_current.change;
    const changePercent = data.prediction_vs_current.change_percent;
    
    const changeElement = document.getElementById('price-change');
    const changePercentElement = document.getElementById('price-change-percent');
    
    // Réinitialiser les classes
    changeElement.classList.remove('positive-change', 'negative-change');
    changePercentElement.classList.remove('positive-change', 'negative-change');
    
    // Mettre à jour le texte
    changeElement.textContent = change.toFixed(2);
    changePercentElement.textContent = changePercent.toFixed(2) + '%';
    
    // Ajouter les classes appropriées
    if (change > 0) {
        changeElement.classList.add('positive-change');
        changePercentElement.classList.add('positive-change');
    } else {
        changeElement.classList.add('negative-change');
        changePercentElement.classList.add('negative-change');
    }
}

function createChart(data) {
    const ctx = document.getElementById('prediction-chart').getContext('2d');
    
    // Détruire le graphique existant s'il y en a un
    if (window.predictionChart) {
        window.predictionChart.destroy();
    }
    
    // Créer le graphique
    window.predictionChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: ['Actuel', 'Prédit'],
            datasets: [{
                label: 'Prix Tesla ($)',
                data: [data.current_price, data.predicted_price],
                backgroundColor: 'rgba(54, 162, 235, 0.2)',
                borderColor: 'rgba(54, 162, 235, 1)',
                borderWidth: 2,
                pointRadius: 6,
                pointHoverRadius: 8
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: false
                }
            },
            plugins: {
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return `Prix: $${context.raw.toFixed(2)}`;
                        }
                    }
                }
            }
        }
    });
}
