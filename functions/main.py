"""
Firebase Cloud Functions for OncoGuardian API
Wraps the Flask application for deployment to Firebase Cloud Functions

Uses lazy-loading to avoid initialization timeout issues with large ML models.
"""

import os
import sys
from pathlib import Path
import logging
import threading

from firebase_functions import https_fn
from firebase_functions.options import set_global_options
from firebase_admin import initialize_app
from flask import jsonify

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Firebase
initialize_app()

# Set global options for Cloud Functions
set_global_options(
    max_instances=50,  # Allow up to 50 concurrent instances
    memory=2048,       # 2GB memory for model loading
    timeout_sec=540    # 9 minutes timeout for predictions
)

# Global app instance (lazy loaded)
_app = None
_init_lock = threading.Lock()

def get_app():
    """Lazy-load Flask app on first request to avoid initialization timeout"""
    global _app
    with _init_lock:
        if _app is None:
            logger.info("Initializing Flask app and loading ML models...")
            try:
                # Import Flask app from local src directory
                from src.app import app as flask_app
                _app = flask_app
                logger.info("✅ Flask app initialized successfully with predictor loaded")
            except Exception as e:
                logger.error(f"❌ Failed to initialize Flask app: {e}", exc_info=True)
                raise
    return _app


@https_fn.on_request(max_instances=50)
def api(req: https_fn.Request) -> https_fn.Response:
    """
    OncoGuardian API Cloud Function - Main entry point
    
    Endpoints:
    - GET /health - Health check
    - GET /model-info - Model metadata  
    - POST /predict - Make cancer risk predictions
    
    Args:
        req: HTTP request object
        
    Returns:
        HTTP response from Flask app
    """
    try:
        # Handle CORS preflight requests
        if req.method == 'OPTIONS':
            headers = {
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Methods': 'GET, HEAD, OPTIONS, POST',
                'Access-Control-Allow-Headers': 'Content-Type, Authorization',
                'Access-Control-Max-Age': '3600'
            }
            return https_fn.Response('', status=204, headers=headers)
        
        # Get Flask app (lazy loaded on first request)
        app = get_app()
        
        # Create Flask test request context from HTTP request
        with app.test_request_context(
            path=req.path,
            method=req.method,
            data=req.get_data(),
            headers=dict(req.headers),
            query_string=req.query_string or b''
        ):
            # Dispatch the request to Flask
            response = app.full_dispatch_request()
            
            # Add CORS headers if response is a Flask Response object
            if hasattr(response, 'headers'):
                response.headers['Access-Control-Allow-Origin'] = '*'
                response.headers['Access-Control-Allow-Methods'] = 'GET, HEAD, OPTIONS, POST'
                response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
            
            return response
            
    except Exception as e:
        logger.error(f"Error processing request: {e}", exc_info=True)
        return https_fn.Response(
            jsonify({'error': str(e), 'success': False}).get_data(as_text=True),
            status=500,
            mimetype='application/json'
        )
