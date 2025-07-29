import dash
from dash import dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import numpy as np
import base64
import io
import json
import requests
import time
from datetime import datetime, timedelta
import threading

# Initialize Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "ML Image Classifier Dashboard"

# Configuration
API_BASE_URL = "http://localhost:5000"

# Global variables for storing data
model_metrics = {}
training_history = []
prediction_history = []

def get_model_status():
    """Get current model status from API"""
    try:
        response = requests.get(f"{API_BASE_URL}/status")
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": "Failed to get model status"}
    except Exception as e:
        return {"error": f"Connection error: {str(e)}"}

def get_training_status():
    """Get current training status from API"""
    try:
        response = requests.get(f"{API_BASE_URL}/training-status")
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": "Failed to get training status"}
    except Exception as e:
        return {"error": f"Connection error: {str(e)}"}

# Layout
app.layout = dbc.Container([
    # Header
    dbc.Row([
        dbc.Col([
            html.H1("🐱🐶 ML Image Classifier Dashboard", 
                   className="text-center mb-4 text-primary"),
            html.Hr()
        ])
    ]),
    
    # Main content
    dbc.Row([
        # Left column - Model Status and Controls
        dbc.Col([
            # Model Status Card
            dbc.Card([
                dbc.CardHeader("📊 Model Status"),
                dbc.CardBody([
                    html.Div(id="model-status-content"),
                    dbc.Button("🔄 Refresh Status", id="refresh-status-btn", 
                              color="primary", className="mt-2")
                ])
            ], className="mb-4"),
            
            # Training Controls Card
            dbc.Card([
                dbc.CardHeader("🚀 Training Controls"),
                dbc.CardBody([
                    dbc.Button("🎯 Start Training", id="start-training-btn", 
                              color="success", className="me-2"),
                    dbc.Button("⏹️ Stop Training", id="stop-training-btn", 
                              color="danger", className="me-2"),
                    html.Div(id="training-status-content", className="mt-3")
                ])
            ], className="mb-4"),
            
            # Upload Data Card
            dbc.Card([
                dbc.CardHeader("📁 Upload New Data"),
                dbc.CardBody([
                    dcc.Upload(
                        id='upload-data',
                        children=html.Div([
                            'Drag and Drop or ',
                            html.A('Select Files')
                        ]),
                        style={
                            'width': '100%',
                            'height': '60px',
                            'lineHeight': '60px',
                            'borderWidth': '1px',
                            'borderStyle': 'dashed',
                            'borderRadius': '5px',
                            'textAlign': 'center',
                            'margin': '10px'
                        },
                        multiple=True
                    ),
                    html.Div(id='upload-output'),
                    dbc.Button("🔄 Retrain Model", id="retrain-btn", 
                              color="warning", className="mt-2")
                ])
            ])
        ], width=4),
        
        # Right column - Visualizations and Predictions
        dbc.Col([
            # Tabs for different sections
            dbc.Tabs([
                # Predictions Tab
                dbc.Tab([
                    dbc.Card([
                        dbc.CardHeader("🔮 Make Predictions"),
                        dbc.CardBody([
                            dcc.Upload(
                                id='upload-prediction',
                                children=html.Div([
                                    'Upload an image for prediction'
                                ]),
                                style={
                                    'width': '100%',
                                    'height': '60px',
                                    'lineHeight': '60px',
                                    'borderWidth': '1px',
                                    'borderStyle': 'dashed',
                                    'borderRadius': '5px',
                                    'textAlign': 'center',
                                    'margin': '10px'
                                },
                                multiple=False
                            ),
                            html.Div(id='prediction-output'),
                            html.Div(id='prediction-result')
                        ])
                    ])
                ], label="Predictions", tab_id="predictions"),
                
                # Metrics Tab
                dbc.Tab([
                    dbc.Card([
                        dbc.CardHeader("📈 Model Metrics"),
                        dbc.CardBody([
                            dcc.Graph(id='metrics-chart'),
                            dcc.Graph(id='confusion-matrix'),
                            dcc.Graph(id='training-history')
                        ])
                    ])
                ], label="Metrics", tab_id="metrics"),
                
                # Monitoring Tab
                dbc.Tab([
                    dbc.Card([
                        dbc.CardHeader("📊 System Monitoring"),
                        dbc.CardBody([
                            dcc.Graph(id='uptime-chart'),
                            dcc.Graph(id='prediction-latency'),
                            html.Div(id='system-stats')
                        ])
                    ])
                ], label="Monitoring", tab_id="monitoring")
            ])
        ], width=8)
    ])
], fluid=True)

# Callbacks
@app.callback(
    Output('model-status-content', 'children'),
    Input('refresh-status-btn', 'n_clicks'),
    Input('interval-component', 'n_intervals')
)
def update_model_status(n_clicks, n_intervals):
    """Update model status"""
    status = get_model_status()
    
    if "error" in status:
        return html.Div([
            html.P(f"❌ {status['error']}", className="text-danger")
        ])
    
    return html.Div([
        html.P(f"✅ Model Loaded: {status['model_loaded']}"),
        html.P(f"📊 Total Parameters: {status['model_info']['total_params']:,}"),
        html.P(f"🎯 Classes: {', '.join(status['model_info']['class_names'])}"),
        html.P(f"⏰ Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    ])

@app.callback(
    Output('training-status-content', 'children'),
    Input('interval-component', 'n_intervals')
)
def update_training_status(n_intervals):
    """Update training status"""
    status = get_training_status()
    
    if "error" in status:
        return html.Div([
            html.P(f"❌ {status['error']}", className="text-danger")
        ])
    
    status_color = {
        'idle': 'secondary',
        'training': 'warning',
        'completed': 'success',
        'failed': 'danger'
    }
    
    return html.Div([
        dbc.Badge(status['status'], color=status_color.get(status['status'], 'secondary')),
        html.P(f"📝 {status['message']}"),
        dbc.Progress(value=status['progress'], className="mt-2") if 'progress' in status else None
    ])

@app.callback(
    Output('start-training-btn', 'disabled'),
    Output('stop-training-btn', 'disabled'),
    Input('interval-component', 'n_intervals')
)
def update_training_buttons(n_intervals):
    """Update training button states"""
    status = get_training_status()
    
    if "error" in status:
        return True, True
    
    is_training = status['status'] == 'training'
    return is_training, not is_training

@app.callback(
    Output('start-training-btn', 'children'),
    Input('start-training-btn', 'n_clicks')
)
def start_training(n_clicks):
    """Start model training"""
    if n_clicks:
        try:
            response = requests.post(f"{API_BASE_URL}/train")
            if response.status_code == 200:
                return "🎯 Training Started"
            else:
                return "❌ Training Failed"
        except Exception as e:
            return f"❌ Error: {str(e)}"
    return "🎯 Start Training"

@app.callback(
    Output('prediction-output', 'children'),
    Input('upload-prediction', 'contents'),
    State('upload-prediction', 'filename')
)
def handle_prediction_upload(contents, filename):
    """Handle prediction image upload"""
    if contents is None:
        return ""
    
    # Decode image
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    
    try:
        # Send prediction request
        files = {'image': (filename, decoded, 'image/jpeg')}
        response = requests.post(f"{API_BASE_URL}/predict", files=files)
        
        if response.status_code == 200:
            result = response.json()
            
            # Store prediction history
            prediction_history.append({
                'timestamp': datetime.now(),
                'filename': filename,
                'prediction': result['predicted_class'],
                'confidence': result['confidence'],
                'latency': result['prediction_time']
            })
            
            return html.Div([
                html.H5(f"Prediction Result: {result['predicted_class'].upper()}"),
                html.P(f"Confidence: {result['confidence']:.2%}"),
                html.P(f"Prediction Time: {result['prediction_time']:.3f}s"),
                html.Img(src=contents, style={'maxWidth': '300px', 'maxHeight': '300px'})
            ])
        else:
            return html.Div(f"❌ Prediction failed: {response.text}")
            
    except Exception as e:
        return html.Div(f"❌ Error: {str(e)}")

@app.callback(
    Output('metrics-chart', 'figure'),
    Input('interval-component', 'n_intervals')
)
def update_metrics_chart(n_intervals):
    """Update metrics visualization"""
    status = get_model_status()
    
    if "error" in status or 'metrics' not in status:
        # Return empty chart
        return go.Figure().add_annotation(
            text="No metrics available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
    
    metrics = status['metrics']
    
    # Create metrics bar chart
    fig = go.Figure(data=[
        go.Bar(
            x=['Accuracy', 'Precision', 'Recall', 'F1-Score'],
            y=[
                metrics.get('test_accuracy', 0),
                metrics.get('test_precision', 0),
                metrics.get('test_recall', 0),
                metrics.get('test_f1', 0)
            ],
            marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        )
    ])
    
    fig.update_layout(
        title="Model Performance Metrics",
        yaxis_title="Score",
        yaxis_range=[0, 1],
        showlegend=False
    )
    
    return fig

@app.callback(
    Output('confusion-matrix', 'figure'),
    Input('interval-component', 'n_intervals')
)
def update_confusion_matrix(n_intervals):
    """Update confusion matrix"""
    status = get_model_status()
    
    if "error" in status or 'metrics' not in status:
        return go.Figure().add_annotation(
            text="No confusion matrix available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
    
    metrics = status['metrics']
    conf_matrix = metrics.get('confusion_matrix', [[0, 0], [0, 0]])
    
    fig = go.Figure(data=go.Heatmap(
        z=conf_matrix,
        x=['Predicted Cat', 'Predicted Dog'],
        y=['Actual Cat', 'Actual Dog'],
        colorscale='Blues',
        text=conf_matrix,
        texttemplate="%{text}",
        textfont={"size": 16}
    ))
    
    fig.update_layout(
        title="Confusion Matrix",
        xaxis_title="Predicted",
        yaxis_title="Actual"
    )
    
    return fig

@app.callback(
    Output('uptime-chart', 'figure'),
    Input('interval-component', 'n_intervals')
)
def update_uptime_chart(n_intervals):
    """Update uptime chart"""
    # Simulate uptime data (in real implementation, this would come from monitoring)
    now = datetime.now()
    times = [now - timedelta(minutes=i) for i in range(60, 0, -1)]
    uptime_values = [99.5 + np.random.normal(0, 0.1) for _ in range(60)]
    
    fig = go.Figure(data=[
        go.Scatter(
            x=times,
            y=uptime_values,
            mode='lines+markers',
            name='Uptime %',
            line=dict(color='#28a745', width=2)
        )
    ])
    
    fig.update_layout(
        title="System Uptime (Last Hour)",
        xaxis_title="Time",
        yaxis_title="Uptime %",
        yaxis_range=[95, 100]
    )
    
    return fig

@app.callback(
    Output('prediction-latency', 'figure'),
    Input('interval-component', 'n_intervals')
)
def update_latency_chart(n_intervals):
    """Update prediction latency chart"""
    if not prediction_history:
        return go.Figure().add_annotation(
            text="No prediction data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
    
    # Get last 20 predictions
    recent_predictions = prediction_history[-20:]
    
    fig = go.Figure(data=[
        go.Scatter(
            x=[p['timestamp'] for p in recent_predictions],
            y=[p['latency'] for p in recent_predictions],
            mode='lines+markers',
            name='Latency (s)',
            line=dict(color='#dc3545', width=2)
        )
    ])
    
    fig.update_layout(
        title="Prediction Latency (Last 20 Predictions)",
        xaxis_title="Time",
        yaxis_title="Latency (seconds)"
    )
    
    return fig

@app.callback(
    Output('system-stats', 'children'),
    Input('interval-component', 'n_intervals')
)
def update_system_stats(n_intervals):
    """Update system statistics"""
    if not prediction_history:
        return html.Div("No prediction data available")
    
    # Calculate statistics
    latencies = [p['latency'] for p in prediction_history]
    avg_latency = np.mean(latencies)
    max_latency = np.max(latencies)
    min_latency = np.min(latencies)
    
    total_predictions = len(prediction_history)
    
    return html.Div([
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4(f"{total_predictions}", className="text-primary"),
                        html.P("Total Predictions", className="text-muted")
                    ])
                ])
            ], width=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4(f"{avg_latency:.3f}s", className="text-success"),
                        html.P("Avg Latency", className="text-muted")
                    ])
                ])
            ], width=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4(f"{max_latency:.3f}s", className="text-warning"),
                        html.P("Max Latency", className="text-muted")
                    ])
                ])
            ], width=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4(f"{min_latency:.3f}s", className="text-info"),
                        html.P("Min Latency", className="text-muted")
                    ])
                ])
            ], width=3)
        ])
    ])

# Add interval component for auto-refresh
app.layout.children.append(
    dcc.Interval(
        id='interval-component',
        interval=5*1000,  # 5 seconds
        n_intervals=0
    )
)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8050, help='Port to run the dashboard on')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to run the dashboard on')
    args = parser.parse_args()
    
    app.run_server(debug=True, host=args.host, port=args.port) 