from flask import Flask, jsonify, request, send_from_directory
import requests
import os
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder='static', static_url_path='')

CONVOSO_API_TOKEN = os.getenv('CONVOSO_API_TOKEN')
CONVOSO_API_BASE_URL = os.getenv('CONVOSO_API_BASE_URL', 'https://api.convoso.com/v1')
CONVOSO_TIMEOUT = int(os.getenv('CONVOSO_TIMEOUT', '30'))

if not CONVOSO_API_TOKEN:
    logger.error("CONVOSO_API_TOKEN environment variable is not set")
    raise ValueError("CONVOSO_API_TOKEN must be set in environment variables")

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/api/convoso/call-logs', methods=['POST'])
def get_call_logs():
    try:
        data = request.json
        url = f"{CONVOSO_API_BASE_URL}/log/retrieve"
        
        params = {
            'auth_token': CONVOSO_API_TOKEN,
            'limit': data.get('limit', 100),
            'start_time': data.get('start_time'),
            'end_time': data.get('end_time')
        }
        
        logger.info(f"Fetching call logs from Convoso API")
        response = requests.post(url, json=params, timeout=CONVOSO_TIMEOUT)
        
        if response.status_code == 200:
            logger.info("Successfully retrieved call logs")
            return jsonify(response.json())
        else:
            logger.warning(f"Convoso API returned status {response.status_code}")
            return jsonify({'error': f'API returned status {response.status_code}'}), response.status_code
            
    except requests.exceptions.Timeout:
        logger.error("Convoso API timeout")
        return jsonify({'error': 'API request timeout'}), 504
    except Exception as e:
        logger.error(f"Error fetching call logs: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/convoso/lead-search', methods=['POST'])
def search_leads():
    try:
        data = request.json
        url = f"{CONVOSO_API_BASE_URL}/leads/search"
        
        params = {
            'auth_token': CONVOSO_API_TOKEN,
            'phone_number': data.get('phone_number'),
            'lead_id': data.get('lead_id')
        }
        
        logger.info(f"Searching leads in Convoso API")
        response = requests.post(url, json=params, timeout=CONVOSO_TIMEOUT)
        
        if response.status_code == 200:
            logger.info("Successfully retrieved lead data")
            return jsonify(response.json())
        else:
            logger.warning(f"Convoso API returned status {response.status_code}")
            return jsonify({'error': f'API returned status {response.status_code}'}), response.status_code
            
    except Exception as e:
        logger.error(f"Error searching leads: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/convoso/recordings', methods=['POST'])
def get_recordings():
    try:
        data = request.json
        url = f"{CONVOSO_API_BASE_URL}/leads/get-recordings"
        
        params = {
            'auth_token': CONVOSO_API_TOKEN,
            'lead_id': data.get('lead_id')
        }
        
        logger.info(f"Fetching recordings from Convoso API")
        response = requests.post(url, json=params, timeout=CONVOSO_TIMEOUT)
        
        if response.status_code == 200:
            logger.info("Successfully retrieved recordings")
            return jsonify(response.json())
        else:
            logger.warning(f"Convoso API returned status {response.status_code}")
            return jsonify({'error': f'API returned status {response.status_code}'}), response.status_code
            
    except Exception as e:
        logger.error(f"Error fetching recordings: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/health')
def health():
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
