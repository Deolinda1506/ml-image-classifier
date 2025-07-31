// ML Pipeline Dashboard JavaScript

class MLPipelineDashboard {
    constructor() {
        this.init();
        this.setupEventListeners();
        this.startPeriodicUpdates();
    }

    init() {
        this.updateModelStatus();
        this.updateStatistics();
        this.loadRecentPredictions();
        this.loadModelInfo();
    }

    setupEventListeners() {
        // Prediction form
        document.getElementById('prediction-form').addEventListener('submit', (e) => {
            e.preventDefault();
            this.handlePrediction();
        });

        // Upload form
        document.getElementById('upload-form').addEventListener('submit', (e) => {
            e.preventDefault();
            this.handleBulkUpload();
        });

        // Retrain button
        document.getElementById('retrain-btn').addEventListener('click', () => {
            this.handleRetrain();
        });

        // Image preview
        document.getElementById('image-upload').addEventListener('change', (e) => {
            this.previewImage(e.target.files[0]);
        });
    }

    async handlePrediction() {
        const fileInput = document.getElementById('image-upload');
        const file = fileInput.files[0];

        if (!file) {
            this.showError('Please select an image file');
            return;
        }

        this.showLoading('Making prediction...');

        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch('/api/predict', {
                method: 'POST',
                body: formData
            });

            const result = await response.json();

            if (response.ok) {
                this.displayPredictionResult(result);
                this.updateStatistics();
                this.loadRecentPredictions();
            } else {
                this.showError(result.error || 'Prediction failed');
            }
        } catch (error) {
            this.showError('Network error: ' + error.message);
        } finally {
            this.hideLoading();
        }
    }

    displayPredictionResult(result) {
        const resultDiv = document.getElementById('prediction-result');
        
        if (result.error) {
            resultDiv.innerHTML = `
                <div class="prediction-result prediction-error fade-in">
                    <i class="fas fa-exclamation-triangle fa-2x mb-3"></i>
                    <h5>Prediction Error</h5>
                    <p>${result.error}</p>
                </div>
            `;
        } else {
            const confidence = (result.confidence * 100).toFixed(1);
            const confidenceClass = confidence > 80 ? 'confidence-high' : 
                                  confidence > 60 ? 'confidence-medium' : 'confidence-low';
            
            resultDiv.innerHTML = `
                <div class="prediction-result prediction-success fade-in">
                    <i class="fas fa-check-circle fa-2x mb-3"></i>
                    <h5>Prediction Result</h5>
                    <h3 class="mb-3">${result.predicted_class.toUpperCase()}</h3>
                    <p><strong>Confidence:</strong> ${confidence}%</p>
                    <div class="confidence-bar">
                        <div class="confidence-fill ${confidenceClass}" style="width: ${confidence}%"></div>
                    </div>
                    <div class="mt-3">
                        <small>Class Probabilities:</small><br>
                        ${Object.entries(result.probabilities).map(([cls, prob]) => 
                            `<span class="badge bg-light text-dark me-2">${cls}: ${(prob * 100).toFixed(1)}%</span>`
                        ).join('')}
                    </div>
                </div>
            `;
        }
    }

    async handleBulkUpload() {
        const fileInput = document.getElementById('bulk-upload');
        const files = fileInput.files;

        if (files.length === 0) {
            this.showError('Please select files to upload');
            return;
        }

        this.showLoading('Uploading files...');
        this.showUploadProgress();

        const formData = new FormData();
        for (let file of files) {
            formData.append('files', file);
        }

        try {
            const response = await fetch('/api/upload', {
                method: 'POST',
                body: formData
            });

            const result = await response.json();

            if (response.ok) {
                this.showSuccess(`Successfully uploaded ${result.successful_uploads} files`);
                this.updateStatistics();
            } else {
                this.showError(result.error || 'Upload failed');
            }
        } catch (error) {
            this.showError('Network error: ' + error.message);
        } finally {
            this.hideLoading();
            this.hideUploadProgress();
        }
    }

    async handleRetrain() {
        if (confirm('Are you sure you want to retrain the model? This may take several minutes.')) {
            this.showLoading('Starting model retraining...');
            
            try {
                const response = await fetch('/api/retrain', {
                    method: 'POST'
                });

                const result = await response.json();

                if (response.ok) {
                    this.showSuccess('Model retraining started successfully!');
                    this.startTrainingMonitoring();
                } else {
                    this.showError(result.error || 'Failed to start retraining');
                }
            } catch (error) {
                this.showError('Network error: ' + error.message);
            } finally {
                this.hideLoading();
            }
        }
    }

    startTrainingMonitoring() {
        const statusDiv = document.getElementById('training-status');
        const progressBar = statusDiv.querySelector('.progress-bar');
        const messageDiv = document.getElementById('training-message');
        
        statusDiv.style.display = 'block';
        
        const checkStatus = async () => {
            try {
                const response = await fetch('/api/training-status');
                const status = await response.json();
                
                progressBar.style.width = status.progress + '%';
                messageDiv.textContent = status.message;
                
                if (status.status === 'completed') {
                    this.showSuccess('Model training completed successfully!');
                    this.updateModelStatus();
                    this.updateStatistics();
                    return;
                } else if (status.status === 'failed') {
                    this.showError('Model training failed: ' + status.message);
                    return;
                }
                
                // Continue monitoring
                setTimeout(checkStatus, 2000);
            } catch (error) {
                console.error('Error checking training status:', error);
                setTimeout(checkStatus, 5000);
            }
        };
        
        checkStatus();
    }

    async updateModelStatus() {
        try {
            const response = await fetch('/api/health');
            const health = await response.json();
            
            const statusElement = document.getElementById('model-status');
            const icon = statusElement.querySelector('i');
            
            if (health.model_loaded) {
                icon.className = 'fas fa-circle text-success';
                statusElement.innerHTML = '<i class="fas fa-circle text-success"></i> Model Ready';
            } else {
                icon.className = 'fas fa-circle text-danger';
                statusElement.innerHTML = '<i class="fas fa-circle text-danger"></i> Model Not Loaded';
            }
        } catch (error) {
            console.error('Error updating model status:', error);
        }
    }

    async updateStatistics() {
        try {
            const response = await fetch('/api/statistics');
            const stats = await response.json();
            
            if (response.ok) {
                document.getElementById('cpu-usage').textContent = stats.system.cpu_percent.toFixed(1) + '%';
                document.getElementById('memory-usage').textContent = stats.system.memory_percent.toFixed(1) + '%';
                document.getElementById('total-predictions').textContent = stats.predictions.total;
                document.getElementById('total-uploads').textContent = stats.uploads.total;
            }
        } catch (error) {
            console.error('Error updating statistics:', error);
        }
    }

    async loadRecentPredictions() {
        try {
            const response = await fetch('/api/predictions');
            const predictions = await response.json();
            
            const tbody = document.getElementById('predictions-table');
            
            if (predictions.length === 0) {
                tbody.innerHTML = '<tr><td colspan="4" class="text-center text-muted">No predictions yet</td></tr>';
                return;
            }
            
            tbody.innerHTML = predictions.slice(0, 10).map(pred => `
                <tr class="fade-in">
                    <td>${new Date(pred.timestamp).toLocaleString()}</td>
                    <td>
                        <span class="status-indicator ${pred.error ? 'status-error' : 'status-success'}"></span>
                        ${pred.predicted_class || 'Error'}
                    </td>
                    <td>${pred.confidence ? (pred.confidence * 100).toFixed(1) + '%' : 'N/A'}</td>
                    <td>
                        <span class="badge ${pred.error ? 'bg-danger' : 'bg-success'}">
                            ${pred.error ? 'Failed' : 'Success'}
                        </span>
                    </td>
                </tr>
            `).join('');
        } catch (error) {
            console.error('Error loading predictions:', error);
        }
    }

    async loadModelInfo() {
        try {
            const response = await fetch('/api/model-info');
            const info = await response.json();
            
            const infoDiv = document.getElementById('model-info');
            
            if (response.ok && !info.error) {
                infoDiv.innerHTML = `
                    <div class="fade-in">
                        <p><strong>Model Type:</strong> ${info.model_type}</p>
                        <p><strong>Image Size:</strong> ${info.img_size[0]}x${info.img_size[1]}</p>
                        <p><strong>Classes:</strong> ${info.class_names.join(', ')}</p>
                        <p><strong>Loaded:</strong> ${new Date(info.loaded_at).toLocaleString()}</p>
                    </div>
                `;
            } else {
                infoDiv.innerHTML = '<p class="text-muted">Model information not available</p>';
            }
        } catch (error) {
            console.error('Error loading model info:', error);
        }
    }

    previewImage(file) {
        if (file) {
            const reader = new FileReader();
            reader.onload = function(e) {
                const resultDiv = document.getElementById('prediction-result');
                resultDiv.innerHTML = `
                    <div class="text-center">
                        <img src="${e.target.result}" class="upload-preview" alt="Preview">
                        <p class="text-muted mt-2">Image preview</p>
                    </div>
                `;
            };
            reader.readAsDataURL(file);
        }
    }

    startPeriodicUpdates() {
        // Update statistics every 30 seconds
        setInterval(() => {
            this.updateStatistics();
        }, 30000);

        // Update model status every 60 seconds
        setInterval(() => {
            this.updateModelStatus();
        }, 60000);
    }

    showLoading(message = 'Loading...') {
        document.getElementById('loading-message').textContent = message;
        const modal = new bootstrap.Modal(document.getElementById('loadingModal'));
        modal.show();
    }

    hideLoading() {
        const modal = bootstrap.Modal.getInstance(document.getElementById('loadingModal'));
        if (modal) {
            modal.hide();
        }
    }

    showUploadProgress() {
        const progressDiv = document.getElementById('upload-progress');
        progressDiv.style.display = 'block';
    }

    hideUploadProgress() {
        const progressDiv = document.getElementById('upload-progress');
        progressDiv.style.display = 'none';
    }

    showSuccess(message) {
        document.getElementById('success-message').textContent = message;
        const modal = new bootstrap.Modal(document.getElementById('successModal'));
        modal.show();
    }

    showError(message) {
        document.getElementById('error-message').textContent = message;
        const modal = new bootstrap.Modal(document.getElementById('errorModal'));
        modal.show();
    }
}

// Initialize dashboard when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new MLPipelineDashboard();
}); 