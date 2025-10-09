from flask import Flask, jsonify, request, send_from_directory
import requests
import os
from datetime import datetime
import logging
import sqlite3
import json
from contextlib import contextmanager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder='static', static_url_path='')

CONVOSO_API_TOKEN = os.getenv('CONVOSO_API_TOKEN')
CONVOSO_API_BASE_URL = os.getenv('CONVOSO_API_BASE_URL', 'https://api.convoso.com/v1')
CONVOSO_TIMEOUT = int(os.getenv('CONVOSO_TIMEOUT', '30'))
DB_PATH = os.getenv('DB_PATH', 'call_qa_production.db')

if not CONVOSO_API_TOKEN:
    logger.error("CONVOSO_API_TOKEN environment variable is not set")
    raise ValueError("CONVOSO_API_TOKEN must be set in environment variables")

@contextmanager
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()

def init_database():
    with get_db() as conn:
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS calls (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                call_id TEXT UNIQUE NOT NULL,
                lead_id TEXT,
                agent TEXT,
                campaign TEXT,
                status TEXT,
                duration INTEGER,
                datetime TEXT,
                call_type TEXT,
                disposition TEXT,
                recording_url TEXT,
                processing_status TEXT DEFAULT 'pending',
                transcript TEXT,
                qa_score REAL,
                sentiment_score REAL,
                clarity_score REAL,
                professionalism_score REAL,
                rapport_score REAL,
                fact_gathering_score REAL,
                script_adherence_score REAL,
                missed_opportunities TEXT,
                coaching_notes TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS agents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_name TEXT UNIQUE NOT NULL,
                total_calls INTEGER DEFAULT 0,
                avg_qa_score REAL DEFAULT 0,
                conversion_rate REAL DEFAULT 0,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS scripts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                script_name TEXT NOT NULL,
                script_content TEXT,
                active BOOLEAN DEFAULT 0,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS scoring_weights (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sentiment_weight REAL DEFAULT 0.2,
                clarity_weight REAL DEFAULT 0.15,
                professionalism_weight REAL DEFAULT 0.15,
                rapport_weight REAL DEFAULT 0.15,
                fact_gathering_weight REAL DEFAULT 0.15,
                script_adherence_weight REAL DEFAULT 0.2,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('SELECT COUNT(*) FROM scoring_weights')
        if cursor.fetchone()[0] == 0:
            cursor.execute('''
                INSERT INTO scoring_weights (sentiment_weight, clarity_weight, professionalism_weight,
                                            rapport_weight, fact_gathering_weight, script_adherence_weight)
                VALUES (0.2, 0.15, 0.15, 0.15, 0.15, 0.2)
            ''')
        
        conn.commit()
        logger.info("Database initialized successfully")

init_database()

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
        
        logger.info(f"Fetching call logs from Convoso API with params: {params}")
        response = requests.post(url, params=params, timeout=CONVOSO_TIMEOUT)
        logger.info(f"Convoso API response status: {response.status_code}, body: {response.text[:500]}")
        
        if response.status_code == 200:
            convoso_data = response.json()
            
            if not convoso_data.get('success', False) and convoso_data.get('code') == 400:
                logger.error(f"Convoso API error: {convoso_data.get('text', 'Unknown error')}")
                return jsonify({'error': convoso_data.get('text', 'API request failed'), 'calls': []}), 400
            
            data_obj = convoso_data.get('data', {})
            calls_data = data_obj.get('results', []) if isinstance(data_obj, dict) else []
            
            if calls_data:
                with get_db() as conn:
                    cursor = conn.cursor()
                    for call in calls_data:
                        call_id = call.get('id')
                        cursor.execute('''
                            INSERT OR REPLACE INTO calls 
                            (call_id, lead_id, agent, campaign, status, duration, datetime, 
                             call_type, disposition, recording_url, processing_status)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 
                                   COALESCE((SELECT processing_status FROM calls WHERE call_id = ?), 'pending'))
                        ''', (
                            call_id,
                            call.get('lead_id'),
                            call.get('user'),
                            call.get('campaign'),
                            call.get('status'),
                            call.get('call_length'),
                            call.get('call_date'),
                            call.get('call_type'),
                            call.get('status_name'),
                            call.get('recording_url'),
                            call_id
                        ))
                    conn.commit()
                logger.info(f"Successfully retrieved and stored {len(calls_data)} call logs")
            else:
                logger.warning("No call data found in Convoso API response")
            
            transformed_calls = []
            for call in calls_data:
                transformed_calls.append({
                    'call_id': call.get('id'),
                    'lead_id': call.get('lead_id'),
                    'agent': call.get('user'),
                    'campaign': call.get('campaign'),
                    'status': call.get('status'),
                    'duration': call.get('call_length'),
                    'datetime': call.get('call_date'),
                    'call_type': call.get('call_type'),
                    'disposition': call.get('status_name'),
                    'recording_url': call.get('recording_url'),
                    'processing_status': 'pending'
                })
            
            return jsonify({'calls': transformed_calls, 'success': True, 'count': len(transformed_calls)})
        else:
            logger.warning(f"Convoso API returned status {response.status_code}")
            return jsonify({'error': f'API returned status {response.status_code}'}), response.status_code
            
    except requests.exceptions.Timeout:
        logger.error("Convoso API timeout")
        return jsonify({'error': 'API request timeout'}), 504
    except Exception as e:
        logger.error(f"Error fetching call logs: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/calls/list', methods=['GET'])
def list_calls():
    try:
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT call_id, lead_id, agent, campaign, status, duration, 
                       datetime, call_type, disposition, processing_status,
                       qa_score, created_at
                FROM calls
                ORDER BY datetime DESC
                LIMIT 1000
            ''')
            
            calls = []
            for row in cursor.fetchall():
                calls.append(dict(row))
            
            return jsonify({'calls': calls, 'count': len(calls)})
            
    except Exception as e:
        logger.error(f"Error listing calls: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/calls/stats', methods=['GET'])
def get_call_stats():
    try:
        with get_db() as conn:
            cursor = conn.cursor()
            
            cursor.execute('SELECT COUNT(*) as total FROM calls')
            total_calls = cursor.fetchone()['total']
            
            cursor.execute('SELECT AVG(qa_score) as avg_score FROM calls WHERE qa_score IS NOT NULL')
            avg_qa_score = cursor.fetchone()['avg_score'] or 0
            
            cursor.execute('SELECT COUNT(*) as processed FROM calls WHERE processing_status = "completed"')
            processed_calls = cursor.fetchone()['processed']
            
            cursor.execute('SELECT COUNT(*) as pending FROM calls WHERE processing_status = "pending"')
            pending_calls = cursor.fetchone()['pending']
            
            return jsonify({
                'total_calls': total_calls,
                'avg_qa_score': round(avg_qa_score, 2),
                'processed_calls': processed_calls,
                'pending_calls': pending_calls
            })
            
    except Exception as e:
        logger.error(f"Error getting call stats: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/agents/list', methods=['GET'])
def list_agents():
    try:
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT agent, COUNT(*) as call_count, AVG(qa_score) as avg_score
                FROM calls
                WHERE agent IS NOT NULL
                GROUP BY agent
                ORDER BY call_count DESC
            ''')
            
            agents = []
            for row in cursor.fetchall():
                agents.append({
                    'name': row['agent'],
                    'call_count': row['call_count'],
                    'avg_score': round(row['avg_score'], 2) if row['avg_score'] else 0
                })
            
            return jsonify({'agents': agents})
            
    except Exception as e:
        logger.error(f"Error listing agents: {e}")
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
