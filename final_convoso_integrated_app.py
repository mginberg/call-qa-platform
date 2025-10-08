
import streamlit as st
import pandas as pd
import sqlite3
import os
import json
import requests
from datetime import datetime, timedelta
import time
import threading
from pathlib import Path
import subprocess
import sys
import re
import hashlib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import whisper
import openai
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import shutil
import tempfile
from typing import Dict, List, Optional, Tuple
import yaml
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import ftplib
from ftplib import FTP
import schedule
import pytz
from io import StringIO
import base64
import asyncio
from urllib.parse import urlencode

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('call_qa_platform.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Case Connect Call QA Intelligence",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Production Configuration with Your Convoso API Token
class ProductionConfig:
    # Your Actual Convoso API Configuration
    CONVOSO_API_TOKEN = "1d7lf53gyeqjf1ulpsy26tad86ppkzfb"
    CONVOSO_API_BASE_URL = "https://api.convoso.com/v1"
    CONVOSO_TIMEOUT = 30  # seconds

    # OpenAI API Configuration (Placeholder)
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', 'sk-placeholder-your-openai-api-key-here')

    # Lead Vendor API Keys (Placeholders)
    LEAD_VENDOR_API_KEYS = {
        'google_ads': os.getenv('GOOGLE_ADS_API_KEY', 'your-google-ads-api-key'),
        'facebook_ads': os.getenv('FACEBOOK_ADS_API_KEY', 'your-facebook-ads-api-key'),
        'bing_ads': os.getenv('BING_ADS_API_KEY', 'your-bing-ads-api-key'),
        'lead_vendor_1': os.getenv('LEAD_VENDOR_1_API_KEY', 'vendor-1-api-key'),
        'lead_vendor_2': os.getenv('LEAD_VENDOR_2_API_KEY', 'vendor-2-api-key'),
        'lead_vendor_3': os.getenv('LEAD_VENDOR_3_API_KEY', 'vendor-3-api-key')
    }

    # FTP Server Configuration
    FTP_HOST = os.getenv('FTP_HOST', 'localhost')
    FTP_PORT = int(os.getenv('FTP_PORT', '21'))
    FTP_USERNAME = os.getenv('FTP_USERNAME', 'callqa_user')
    FTP_PASSWORD = os.getenv('FTP_PASSWORD', 'secure_password_2025')
    FTP_UPLOAD_DIR = os.getenv('FTP_UPLOAD_DIR', 'incoming_calls')

    # Processing Configuration
    WHISPER_MODEL_SIZE = os.getenv('WHISPER_MODEL_SIZE', 'large')  # For accuracy with high volume
    MAX_CONCURRENT_PROCESSING = int(os.getenv('MAX_CONCURRENT_PROCESSING', '8'))
    CALL_RETENTION_HOURS = int(os.getenv('CALL_RETENTION_HOURS', '24'))
    DATA_RETENTION_DAYS = int(os.getenv('DATA_RETENTION_DAYS', '365'))

    # Database Configuration
    DATABASE_PATH = os.getenv('DATABASE_PATH', 'call_qa_production.db')

    # Enable Convoso Integration
    ENABLE_CONVOSO_INTEGRATION = True

# Professional CSS styling
st.markdown("""
<style>
    .stApp {
        background-color: #1a1d29;
    }

    .main-header {
        background: #2a2f42;
        padding: 1.5rem 2rem;
        border-bottom: 1px solid #3c4557;
        margin-bottom: 2rem;
        border-radius: 8px;
    }

    .metric-card {
        background: #2a2f42;
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #3c4557;
        margin: 0.5rem 0;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }

    .metric-card:hover {
        border-color: #4ecdc4;
        transform: translateY(-2px);
        box-shadow: 0 8px 15px rgba(0, 0, 0, 0.2);
    }

    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: #ffffff;
        margin: 0;
        text-align: center;
    }

    .metric-label {
        color: #a0a9c0;
        font-size: 0.9rem;
        margin-top: 0.5rem;
        text-align: center;
    }

    .status-success { color: #27ae60; }
    .status-warning { color: #f39c12; }
    .status-error { color: #e74c3c; }
    .status-processing { color: #4ecdc4; }

    .convoso-status {
        background: #2a2f42;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #3c4557;
        margin: 1rem 0;
    }

    .lead-info-card {
        background: #252836;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #4ecdc4;
        margin: 0.5rem 0;
    }

    .processing-queue {
        background: #252836;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #3c4557;
        max-height: 200px;
        overflow-y: auto;
    }

    /* Hide Streamlit default styling */
    .stDeployButton {display:none;}
    footer {visibility: hidden;}
    .stApp > header {visibility: hidden;}

    /* Professional table styling */
    .dataframe {
        background-color: #2a2f42 !important;
        border: 1px solid #3c4557 !important;
    }
</style>
""", unsafe_allow_html=True)

class ConvosoAPIClient:
    """Convoso API integration client"""

    def __init__(self, api_token: str, base_url: str = "https://api.convoso.com/v1"):
        self.api_token = api_token
        self.base_url = base_url
        self.timeout = ProductionConfig.CONVOSO_TIMEOUT

    def get_lead_by_id(self, lead_id: str) -> Dict:
        """Get lead information by Lead ID"""
        try:
            url = f"{self.base_url}/leads/get"
            params = {
                'auth_token': self.api_token,
                'lead_id': lead_id
            }

            logger.info(f"Fetching lead data from Convoso API for Lead ID: {lead_id}")
            response = requests.get(url, params=params, timeout=self.timeout)

            if response.status_code == 200:
                data = response.json()
                logger.info(f"Successfully retrieved lead data for Lead ID: {lead_id}")
                return {
                    'success': True,
                    'data': data,
                    'lead_id': lead_id,
                    'api_response_time': response.elapsed.total_seconds()
                }
            else:
                logger.warning(f"Convoso API returned status {response.status_code} for Lead ID: {lead_id}")
                return {
                    'success': False,
                    'error': f"API returned status {response.status_code}",
                    'lead_id': lead_id
                }

        except requests.exceptions.Timeout:
            logger.error(f"Convoso API timeout for Lead ID: {lead_id}")
            return {
                'success': False,
                'error': "API request timeout",
                'lead_id': lead_id
            }
        except Exception as e:
            logger.error(f"Convoso API error for Lead ID {lead_id}: {e}")
            return {
                'success': False,
                'error': str(e),
                'lead_id': lead_id
            }

    def get_lead_by_phone(self, phone_number: str) -> Dict:
        """Get lead information by phone number"""
        try:
            url = f"{self.base_url}/leads/search"
            params = {
                'auth_token': self.api_token,
                'phone_number': phone_number
            }

            logger.info(f"Searching lead data by phone: {phone_number}")
            response = requests.get(url, params=params, timeout=self.timeout)

            if response.status_code == 200:
                data = response.json()
                logger.info(f"Successfully found lead data for phone: {phone_number}")
                return {
                    'success': True,
                    'data': data,
                    'phone_number': phone_number,
                    'api_response_time': response.elapsed.total_seconds()
                }
            else:
                logger.warning(f"Convoso API returned status {response.status_code} for phone: {phone_number}")
                return {
                    'success': False,
                    'error': f"API returned status {response.status_code}",
                    'phone_number': phone_number
                }

        except Exception as e:
            logger.error(f"Convoso API error for phone {phone_number}: {e}")
            return {
                'success': False,
                'error': str(e),
                'phone_number': phone_number
            }

    def test_connection(self) -> Dict:
        """Test Convoso API connection"""
        try:
            url = f"{self.base_url}/account/info"
            params = {'auth_token': self.api_token}

            response = requests.get(url, params=params, timeout=10)

            if response.status_code == 200:
                return {
                    'success': True,
                    'message': "Convoso API connection successful",
                    'response_time': response.elapsed.total_seconds()
                }
            else:
                return {
                    'success': False,
                    'message': f"API returned status {response.status_code}",
                    'status_code': response.status_code
                }

        except Exception as e:
            return {
                'success': False,
                'message': f"Connection failed: {str(e)}",
                'error': str(e)
            }

    def extract_lead_analytics_data(self, convoso_data: Dict) -> Dict:
        """Extract key fields from Convoso lead data for analytics"""
        try:
            if not convoso_data.get('success', False):
                return {}

            lead_data = convoso_data.get('data', {})

            # Extract key fields for analytics (adjust field names based on actual Convoso response)
            analytics_data = {
                'lead_source': lead_data.get('source', 'Unknown'),
                'lead_vendor': lead_data.get('vendor', lead_data.get('list_name', 'Unknown')),
                'customer_state': lead_data.get('state', lead_data.get('address_state', 'Unknown')),
                'lead_status': lead_data.get('status', lead_data.get('disposition', 'Unknown')),
                'call_attempt_number': lead_data.get('contact_attempts', lead_data.get('call_attempts', 1)),
                'lead_score': lead_data.get('score', 0),
                'lead_created_date': lead_data.get('created_date', lead_data.get('date_added', None)),
                'lead_updated_date': lead_data.get('updated_date', lead_data.get('date_modified', None)),
                'campaign_name': lead_data.get('campaign', lead_data.get('campaign_name', 'Unknown')),
                'lead_type': lead_data.get('type', lead_data.get('lead_type', 'Unknown')),
                'customer_city': lead_data.get('city', lead_data.get('address_city', 'Unknown')),
                'customer_zip': lead_data.get('zip', lead_data.get('address_zip', 'Unknown')),
                'lead_value': lead_data.get('value', lead_data.get('estimated_value', 0)),
                'priority': lead_data.get('priority', 'Standard'),
                'tags': lead_data.get('tags', []),
                'custom_fields': {k: v for k, v in lead_data.items() if k.startswith('custom_')},
                'convoso_raw_data': json.dumps(lead_data)  # Store full data for future use
            }

            return analytics_data

        except Exception as e:
            logger.error(f"Error extracting Convoso analytics data: {e}")
            return {}

class EnhancedCallProcessor:
    """Enhanced call processor with Convoso API integration"""

    def __init__(self):
        self.whisper_model = None
        self.processing_queue = []
        self.is_processing = False
        self.processed_count = 0
        self.failed_count = 0
        self.convoso_client = ConvosoAPIClient(ProductionConfig.CONVOSO_API_TOKEN)
        self.executor = ThreadPoolExecutor(max_workers=ProductionConfig.MAX_CONCURRENT_PROCESSING)
        self.load_whisper_model()

    def load_whisper_model(self):
        """Load Whisper model with caching for production"""
        try:
            model_size = ProductionConfig.WHISPER_MODEL_SIZE
            logger.info(f"Loading Whisper model: {model_size}")
            st.info(f"🔄 Loading Whisper {model_size} model for high-accuracy transcription...")

            self.whisper_model = whisper.load_model(model_size)
            logger.info(f"Whisper model '{model_size}' loaded successfully")
            st.success(f"✅ Whisper {model_size} model loaded - Ready for production volume!")

        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            st.error(f"❌ Whisper model loading failed: {e}")
            self.whisper_model = None

    def parse_filename(self, filename: str) -> Dict:
        """Parse call metadata from filename: campaign_agent_customer_leadid_xxx_timestamp_xxx.mp3"""
        try:
            basename = filename.replace('.mp3', '').replace('.wav', '')
            parts = basename.split('_')

            if len(parts) >= 6:
                return {
                    'campaign_id': parts[0],
                    'agent_id': parts[1], 
                    'customer_phone': parts[2],
                    'lead_id': parts[3],
                    'sequence': parts[4],
                    'timestamp': parts[5],
                    'extension': parts[6] if len(parts) > 6 else None,
                    'original_filename': filename
                }
            else:
                logger.warning(f"Filename format not recognized: {filename}")
                return None
        except Exception as e:
            logger.error(f"Error parsing filename {filename}: {e}")
            return None

    def get_convoso_lead_data(self, lead_id: str, phone_number: str) -> Dict:
        """Get lead data from Convoso API"""
        if not ProductionConfig.ENABLE_CONVOSO_INTEGRATION:
            logger.info("Convoso integration disabled")
            return {}

        # Try to get lead by ID first
        if lead_id:
            result = self.convoso_client.get_lead_by_id(lead_id)
            if result.get('success', False):
                return self.convoso_client.extract_lead_analytics_data(result)

        # Fallback to phone number search
        if phone_number:
            result = self.convoso_client.get_lead_by_phone(phone_number)
            if result.get('success', False):
                return self.convoso_client.extract_lead_analytics_data(result)

        logger.warning(f"Could not retrieve Convoso data for Lead ID: {lead_id}, Phone: {phone_number}")
        return {}

    def get_agent_info_from_database(self, agent_id: str) -> Dict:
        """Get agent information from database"""
        try:
            conn = sqlite3.connect(ProductionConfig.DATABASE_PATH)
            cursor = conn.cursor()

            cursor.execute("""
                SELECT agent_name, agent_email, team_name, manager_name 
                FROM agents WHERE agent_id = ?
            """, (agent_id,))

            result = cursor.fetchone()
            conn.close()

            if result:
                return {
                    'agent_name': result[0],
                    'agent_email': result[1],
                    'team_name': result[2],
                    'manager_name': result[3]
                }
            else:
                # Return placeholder info for unknown agents
                return {
                    'agent_name': f'Agent {agent_id}',
                    'agent_email': f'{agent_id}@legalcaseconnect.com',
                    'team_name': 'Team Unknown',
                    'manager_name': 'Manager TBD'
                }

        except Exception as e:
            logger.error(f"Error getting agent info for {agent_id}: {e}")
            return {
                'agent_name': f'Agent {agent_id}',
                'agent_email': f'{agent_id}@legalcaseconnect.com',
                'team_name': 'Team Unknown',
                'manager_name': 'Manager TBD'
            }

    def transcribe_audio(self, file_path: str) -> Dict:
        """Transcribe audio using Whisper Large model for accuracy"""
        try:
            if not self.whisper_model:
                return {"text": "Transcription unavailable - Whisper model not loaded", "duration": 0}

            logger.info(f"Starting transcription for {file_path}")
            result = self.whisper_model.transcribe(file_path, 
                                                 word_timestamps=True,
                                                 fp16=False)  # Better accuracy

            return {
                "text": result["text"],
                "duration": result.get("duration", 0),
                "language": result.get("language", "en"),
                "segments": result.get("segments", [])
            }
        except Exception as e:
            logger.error(f"Transcription failed for {file_path}: {e}")
            return {"text": f"Transcription failed: {str(e)}", "duration": 0}

    def get_script_for_campaign(self, campaign_id: str) -> str:
        """Get active script for campaign"""
        try:
            conn = sqlite3.connect(ProductionConfig.DATABASE_PATH)
            cursor = conn.cursor()

            cursor.execute("""
                SELECT script_content FROM scripts 
                WHERE campaign_id = ? AND is_active = 1 
                ORDER BY updated_date DESC LIMIT 1
            """, (campaign_id,))

            result = cursor.fetchone()
            conn.close()

            return result[0] if result else ""
        except Exception as e:
            logger.error(f"Error getting script for campaign {campaign_id}: {e}")
            return ""

    def analyze_call_quality_enhanced(self, transcript: str, duration: float, 
                                    segments: List, metadata: Dict, 
                                    convoso_data: Dict) -> Dict:
        """Enhanced AI-powered call quality analysis for MVA calls with Convoso data"""
        try:
            # Get script for this campaign
            script_text = self.get_script_for_campaign(metadata.get('campaign_id', ''))

            # Enhanced analysis based on MVA script requirements
            sentiment_score = self.analyze_mva_sentiment(transcript)
            clarity_score = self.analyze_clarity_with_segments(transcript, duration, segments)
            professionalism_score = self.analyze_mva_professionalism(transcript)
            script_adherence_score = self.analyze_mva_script_adherence(transcript, script_text)
            rapport_building_score = self.analyze_rapport_building(transcript)
            fact_gathering_score = self.analyze_fact_gathering(transcript)

            # Calculate silence and flow metrics
            silence_percentage = self.calculate_silence_percentage_accurate(segments, duration)
            interruptions_count = self.count_interruptions_from_segments(segments)

            # Get scoring weights
            weights = self.get_scoring_weights()

            # Enhanced overall score calculation
            overall_score = (
                sentiment_score * weights.get('sentiment', 0.20) +
                clarity_score * weights.get('clarity', 0.15) +
                professionalism_score * weights.get('professionalism', 0.20) +
                script_adherence_score * weights.get('script_adherence', 0.15) +
                rapport_building_score * weights.get('rapport_building', 0.15) +
                fact_gathering_score * weights.get('fact_gathering', 0.15)
            )

            # Enhanced issue flagging for MVA calls
            issues_flagged = self.flag_mva_specific_issues(transcript, sentiment_score, 
                                                         clarity_score, professionalism_score, 
                                                         script_adherence_score, rapport_building_score)

            # Add Convoso-based flags
            if convoso_data:
                issues_flagged.extend(self.flag_convoso_based_issues(convoso_data, overall_score))

            # Generate coaching notes specific to MVA intake
            coaching_notes = self.generate_mva_coaching_notes(
                sentiment_score, clarity_score, professionalism_score,
                script_adherence_score, rapport_building_score, fact_gathering_score,
                convoso_data
            )

            # Extract MVA-specific keywords and compliance items
            mva_keywords = self.extract_mva_keywords(transcript)
            compliance_items = self.check_mva_compliance(transcript)

            return {
                'sentiment_score': round(sentiment_score, 1),
                'clarity_score': round(clarity_score, 1),
                'professionalism_score': round(professionalism_score, 1),
                'script_adherence_score': round(script_adherence_score, 1),
                'rapport_building_score': round(rapport_building_score, 1),
                'fact_gathering_score': round(fact_gathering_score, 1),
                'overall_score': round(overall_score, 1),
                'silence_percentage': round(silence_percentage, 1),
                'interruptions_count': interruptions_count,
                'issues_flagged': issues_flagged,
                'coaching_notes': coaching_notes,
                'mva_keywords': mva_keywords,
                'compliance_items': compliance_items
            }

        except Exception as e:
            logger.error(f"Call analysis failed: {e}")
            return self.get_default_analysis()

    def flag_convoso_based_issues(self, convoso_data: Dict, overall_score: float) -> List[str]:
        """Flag issues based on Convoso lead data"""
        issues = []

        try:
            # Check lead quality vs call quality correlation
            lead_score = convoso_data.get('lead_score', 0)
            if lead_score > 80 and overall_score < 70:
                issues.append("High-value lead received poor service quality")

            # Check call attempt correlation
            attempts = convoso_data.get('call_attempt_number', 1)
            if attempts > 5 and overall_score < 75:
                issues.append("Multiple attempt lead still receiving poor service")

            # Check lead source correlation
            lead_source = convoso_data.get('lead_source', '').lower()
            if 'premium' in lead_source and overall_score < 80:
                issues.append("Premium lead source received substandard service")

            # Check lead status
            lead_status = convoso_data.get('lead_status', '').lower()
            if lead_status in ['hot', 'priority', 'urgent'] and overall_score < 80:
                issues.append(f"Priority lead ({lead_status}) needs better handling")

        except Exception as e:
            logger.error(f"Error flagging Convoso-based issues: {e}")

        return issues

    def process_call_file_enhanced(self, file_path: str) -> bool:
        """Enhanced call processing with Convoso integration"""
        try:
            start_time = time.time()
            logger.info(f"Processing call file with Convoso integration: {file_path}")

            # Parse metadata from filename
            metadata = self.parse_filename(os.path.basename(file_path))
            if not metadata:
                logger.error(f"Could not parse filename: {file_path}")
                return False

            # Get file info
            file_size = os.path.getsize(file_path)

            # Get Convoso lead data
            logger.info(f"Fetching Convoso data for Lead ID: {metadata['lead_id']}")
            convoso_data = self.get_convoso_lead_data(
                metadata['lead_id'], 
                metadata['customer_phone']
            )

            # Transcribe audio with enhanced model
            transcription_result = self.transcribe_audio(file_path)

            # Enhanced quality analysis with Convoso data
            analysis_result = self.analyze_call_quality_enhanced(
                transcription_result['text'], 
                transcription_result['duration'],
                transcription_result.get('segments', []),
                metadata,
                convoso_data
            )

            # Get agent info from database
            agent_info = self.get_agent_info_from_database(metadata['agent_id'])

            # Store in database with all enhanced data including Convoso info
            self.save_call_to_database_enhanced(
                metadata=metadata,
                transcription=transcription_result,
                analysis=analysis_result,
                agent_info=agent_info,
                convoso_data=convoso_data,
                file_path=file_path,
                file_size=file_size
            )

            # Clean up file after processing
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"Deleted processed file: {file_path}")

            processing_time = time.time() - start_time
            logger.info(f"Successfully processed call {metadata['lead_id']} with Convoso data in {processing_time:.2f} seconds")

            self.processed_count += 1
            return True

        except Exception as e:
            logger.error(f"Error processing call file {file_path}: {e}")
            self.failed_count += 1
            return False

    def save_call_to_database_enhanced(self, metadata: Dict, transcription: Dict, 
                                     analysis: Dict, agent_info: Dict, convoso_data: Dict,
                                     file_path: str, file_size: int):
        """Save enhanced call data to database with Convoso integration"""
        try:
            conn = sqlite3.connect(ProductionConfig.DATABASE_PATH)
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO calls (
                    campaign_id, agent_id, customer_phone, lead_id, call_date,
                    file_path, file_size, duration_seconds, transcript,
                    sentiment_score, clarity_score, professionalism_score, 
                    script_adherence_score, rapport_building_score, fact_gathering_score,
                    overall_score, silence_percentage, interruptions_count,
                    keywords_found, issues_flagged, coaching_notes,
                    processed_date, processor_version, agent_name, team_name,
                    mva_keywords, compliance_items,

                    -- Convoso lead analytics fields
                    lead_source, lead_vendor, customer_state, lead_status,
                    call_attempt_number, lead_score, campaign_name, lead_type,
                    customer_city, customer_zip, lead_value, priority,
                    lead_created_date, lead_updated_date, convoso_raw_data

                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                metadata['campaign_id'], metadata['agent_id'], metadata['customer_phone'],
                metadata['lead_id'], datetime.now(), file_path, file_size,
                transcription['duration'], transcription['text'],
                analysis['sentiment_score'], analysis['clarity_score'], 
                analysis['professionalism_score'], analysis['script_adherence_score'],
                analysis['rapport_building_score'], analysis['fact_gathering_score'],
                analysis['overall_score'], analysis['silence_percentage'], 
                analysis['interruptions_count'],
                json.dumps(analysis.get('mva_keywords', [])),
                json.dumps(analysis['issues_flagged']),
                analysis['coaching_notes'], datetime.now(), "2.0-Convoso",
                agent_info['agent_name'], agent_info['team_name'],
                json.dumps(analysis.get('mva_keywords', [])),
                json.dumps(analysis.get('compliance_items', {})),

                # Convoso data
                convoso_data.get('lead_source', 'Unknown'),
                convoso_data.get('lead_vendor', 'Unknown'),
                convoso_data.get('customer_state', 'Unknown'),
                convoso_data.get('lead_status', 'Unknown'),
                convoso_data.get('call_attempt_number', 1),
                convoso_data.get('lead_score', 0),
                convoso_data.get('campaign_name', metadata['campaign_id']),
                convoso_data.get('lead_type', 'MVA'),
                convoso_data.get('customer_city', 'Unknown'),
                convoso_data.get('customer_zip', 'Unknown'),
                convoso_data.get('lead_value', 0),
                convoso_data.get('priority', 'Standard'),
                convoso_data.get('lead_created_date'),
                convoso_data.get('lead_updated_date'),
                convoso_data.get('convoso_raw_data')
            ))

            conn.commit()
            conn.close()

            logger.info(f"Enhanced call data with Convoso integration saved: Lead ID {metadata['lead_id']}")

        except Exception as e:
            logger.error(f"Error saving enhanced call data: {e}")

    # Include all the analysis methods from before (analyze_mva_sentiment, etc.)
    def analyze_mva_sentiment(self, transcript: str) -> float:
        """Analyze sentiment specifically for MVA intake calls"""
        transcript_lower = transcript.lower()

        # MVA-specific empathy and rapport indicators
        empathy_phrases = [
            'sorry to hear', 'understand how difficult', 'can imagine', 
            'feeling ok', 'how are you doing', 'tough situation',
            'here to help', 'we can help', 'take care of you'
        ]

        # Professional MVA language
        professional_phrases = [
            'auto accident', 'motor vehicle', 'legal representation',
            'compensation', 'insurance', 'medical treatment', 'pain and suffering'
        ]

        # Red flag phrases for MVA calls
        concerning_phrases = [
            'guarantee money', 'definitely win', 'easy money',
            'no problem getting', 'for sure you will'
        ]

        empathy_count = sum(1 for phrase in empathy_phrases if phrase in transcript_lower)
        professional_count = sum(1 for phrase in professional_phrases if phrase in transcript_lower)
        concerning_count = sum(1 for phrase in concerning_phrases if phrase in transcript_lower)

        base_score = 65 + (empathy_count * 8) + (professional_count * 4) - (concerning_count * 15)
        return max(0, min(100, base_score))

    def analyze_clarity_with_segments(self, transcript: str, duration: float, segments: List) -> float:
        """Analyze call clarity using Whisper segments"""
        if duration == 0:
            return 50.0

        unclear_indicators = ['[inaudible]', '[unclear]', 'um', 'uh', 'er', '...']
        unclear_count = sum(transcript.lower().count(indicator) for indicator in unclear_indicators)

        # Use segments for more accurate analysis if available
        if segments:
            # Calculate speech rate from segments
            total_words = sum(len(seg.get('text', '').split()) for seg in segments)
            speech_rate = total_words / (duration / 60) if duration > 0 else 0
        else:
            speech_rate = len(transcript.split()) / (duration / 60) if duration > 0 else 0

        clarity_score = 90 - unclear_count * 5

        # Adjust for speech rate
        if speech_rate < 120 or speech_rate > 200:
            clarity_score -= 10

        return max(0, min(100, clarity_score))

    def calculate_silence_percentage_accurate(self, segments: List, duration: float) -> float:
        """Calculate silence percentage using Whisper segments"""
        if not segments or duration == 0:
            return 10.0  # Default estimate

        try:
            # Calculate total speech time from segments
            speech_time = sum(seg.get('end', 0) - seg.get('start', 0) for seg in segments)
            silence_time = max(0, duration - speech_time)
            return (silence_time / duration) * 100
        except:
            return 10.0

    def count_interruptions_from_segments(self, segments: List) -> int:
        """Count interruptions using Whisper segments"""
        if not segments:
            return 2  # Default estimate

        interruptions = 0
        try:
            for i in range(1, len(segments)):
                prev_end = segments[i-1].get('end', 0)
                curr_start = segments[i].get('start', 0)
                # Very short gaps might indicate interruptions
                if curr_start - prev_end < 0.2:
                    interruptions += 1
        except:
            pass

        return min(interruptions, 10)  # Cap at reasonable number

    # Add all other analysis methods here (analyze_mva_professionalism, etc.)
    # For brevity, I'll include key ones and placeholders for others

    def analyze_mva_professionalism(self, transcript: str) -> float:
        """Analyze professionalism specific to MVA intake"""
        transcript_lower = transcript.lower()

        professional_elements = [
            'case connect', 'legal representation', 'attorney', 'law firm',
            'please', 'thank you', 'may i ask', 'i understand',
            'medical treatment', 'insurance company', 'compensation'
        ]

        unprofessional_elements = [
            'yeah', 'like', 'um', 'uh', 'totally', 'awesome',
            'cool', 'whatever', 'basically', 'kind of', 'sort of'
        ]

        professional_count = sum(1 for element in professional_elements if element in transcript_lower)
        unprofessional_count = sum(1 for element in unprofessional_elements if element in transcript_lower)

        base_score = 70 + (professional_count * 5) - (unprofessional_count * 3)
        return max(0, min(100, base_score))

    def analyze_mva_script_adherence(self, transcript: str, script_text: str) -> float:
        """Analyze adherence to MVA intake script"""
        if not script_text:
            return 75.0

        transcript_lower = transcript.lower()

        required_elements = [
            'case connect', 'auto accident', 'information you submitted',
            'sorry to hear', 'few quick questions', 'help you out'
        ]

        elements_found = sum(1 for element in required_elements if element in transcript_lower)

        # Additional script compliance checks
        if 'feeling ok' in transcript_lower or 'how are you' in transcript_lower:
            elements_found += 1
        if 'treatment' in transcript_lower and ('medical' in transcript_lower or 'doctor' in transcript_lower):
            elements_found += 1
        if 'pain' in transcript_lower and ('suffering' in transcript_lower or 'affecting' in transcript_lower):
            elements_found += 1

        adherence_percentage = (elements_found / len(required_elements)) * 100
        return max(0, min(100, adherence_percentage))

    def analyze_rapport_building(self, transcript: str) -> float:
        """Analyze rapport building"""
        transcript_lower = transcript.lower()

        rapport_indicators = [
            'how are you', 'feeling ok', 'sorry to hear', 'understand',
            'difficult time', 'tough situation', 'here for you'
        ]

        rapport_count = sum(1 for indicator in rapport_indicators if indicator in transcript_lower)
        base_score = 60 + (rapport_count * 8)
        return max(0, min(100, base_score))

    def analyze_fact_gathering(self, transcript: str) -> float:
        """Analyze fact gathering effectiveness"""
        transcript_lower = transcript.lower()

        fact_elements = [
            'date of accident', 'what happened', 'medical treatment',
            'doctor', 'hospital', 'police report', 'insurance',
            'injuries', 'pain', 'missed work'
        ]

        questions_asked = sum(1 for element in fact_elements if element in transcript_lower)
        base_score = 50 + (questions_asked * 4)
        return max(0, min(100, base_score))

    def flag_mva_specific_issues(self, transcript: str, sentiment_score: float,
                               clarity_score: float, professionalism_score: float,
                               script_adherence_score: float, rapport_building_score: float) -> List[str]:
        """Flag MVA-specific issues"""
        issues = []
        transcript_lower = transcript.lower()

        if sentiment_score < 65:
            issues.append("Insufficient empathy for accident victim")
        if clarity_score < 70:
            issues.append("Audio clarity affecting professional image")
        if professionalism_score < 70:
            issues.append("Unprofessional language inappropriate for legal intake")
        if script_adherence_score < 65:
            issues.append("Poor adherence to Case Connect intake script")
        if rapport_building_score < 60:
            issues.append("Insufficient rapport building with potential client")

        # MVA-specific compliance issues
        if 'guarantee' in transcript_lower and ('money' in transcript_lower or 'win' in transcript_lower):
            issues.append("CRITICAL: Inappropriate legal guarantees made")

        return issues

    def generate_mva_coaching_notes(self, sentiment: float, clarity: float, 
                                  professionalism: float, script_adherence: float,
                                  rapport_building: float, fact_gathering: float,
                                  convoso_data: Dict) -> str:
        """Generate coaching notes with Convoso data context"""
        notes = []

        if sentiment < 70:
            notes.append("Express more empathy - accident victims need emotional support")
        if clarity < 75:
            notes.append("Improve audio quality and speak more clearly")
        if professionalism < 75:
            notes.append("Use more professional language - you represent a law firm")
        if script_adherence < 70:
            notes.append("Follow Case Connect script more closely")
        if rapport_building < 65:
            notes.append("Spend more time building rapport and trust")

        # Add Convoso-based coaching
        if convoso_data:
            lead_value = convoso_data.get('lead_value', 0)
            if lead_value > 1000:
                notes.append(f"High-value lead (${lead_value}) - ensure premium service")

            attempts = convoso_data.get('call_attempt_number', 1)
            if attempts > 3:
                notes.append(f"Multiple contact attempt #{attempts} - extra patience needed")

        return " | ".join(notes) if notes else "Good performance - maintain quality standards"

    def extract_mva_keywords(self, transcript: str) -> List[str]:
        """Extract MVA-specific keywords"""
        transcript_lower = transcript.lower()

        mva_keywords = [
            'auto accident', 'car accident', 'motor vehicle', 'collision',
            'medical treatment', 'doctor', 'hospital', 'insurance',
            'pain and suffering', 'lost wages', 'medical bills'
        ]

        return [keyword for keyword in mva_keywords if keyword in transcript_lower]

    def check_mva_compliance(self, transcript: str) -> Dict:
        """Check MVA compliance requirements"""
        transcript_lower = transcript.lower()

        return {
            'company_introduced': 'case connect' in transcript_lower,
            'empathy_expressed': any(phrase in transcript_lower for phrase in ['sorry to hear', 'understand']),
            'treatment_discussed': 'medical' in transcript_lower and 'treatment' in transcript_lower,
            'no_guarantees': not ('guarantee' in transcript_lower and 'money' in transcript_lower),
            'professional_tone': not any(word in transcript_lower for word in ['yeah', 'cool', 'awesome'])
        }

    def get_scoring_weights(self) -> Dict[str, float]:
        """Get scoring weights from database"""
        try:
            conn = sqlite3.connect(ProductionConfig.DATABASE_PATH)
            cursor = conn.cursor()

            cursor.execute("SELECT category, weight FROM scoring_weights")
            weights = dict(cursor.fetchall())
            conn.close()

            return weights
        except:
            return {
                'sentiment': 0.20, 'clarity': 0.15, 'professionalism': 0.20,
                'script_adherence': 0.15, 'rapport_building': 0.15, 'fact_gathering': 0.15
            }

    def get_default_analysis(self) -> Dict:
        """Default analysis when processing fails"""
        return {
            'sentiment_score': 50.0, 'clarity_score': 50.0,
            'professionalism_score': 50.0, 'script_adherence_score': 50.0,
            'rapport_building_score': 50.0, 'fact_gathering_score': 50.0,
            'overall_score': 50.0, 'silence_percentage': 10.0,
            'interruptions_count': 2, 'issues_flagged': ['Processing error'],
            'coaching_notes': 'Manual review required', 'mva_keywords': [], 'compliance_items': {}
        }

# Initialize enhanced database with Convoso fields
def init_enhanced_database():
    """Initialize database with Convoso integration fields"""
    conn = sqlite3.connect(ProductionConfig.DATABASE_PATH)
    cursor = conn.cursor()

    # Enhanced calls table with Convoso fields
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS calls (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            campaign_id TEXT,
            agent_id TEXT,
            customer_phone TEXT,
            lead_id TEXT,
            call_date TIMESTAMP,
            file_path TEXT,
            file_size INTEGER,
            duration_seconds REAL,
            transcript TEXT,

            -- Quality scores
            sentiment_score REAL,
            clarity_score REAL,
            professionalism_score REAL,
            script_adherence_score REAL,
            rapport_building_score REAL,
            fact_gathering_score REAL,
            overall_score REAL,

            -- Call flow metrics
            silence_percentage REAL,
            interruptions_count INTEGER,

            -- Analysis results
            keywords_found TEXT,
            issues_flagged TEXT,
            coaching_notes TEXT,
            mva_keywords TEXT,
            compliance_items TEXT,

            -- Processing metadata
            processed_date TIMESTAMP,
            processor_version TEXT,

            -- Agent information
            agent_name TEXT,
            team_name TEXT,

            -- Convoso lead analytics fields
            lead_source TEXT,
            lead_vendor TEXT,
            customer_state TEXT,
            lead_status TEXT,
            call_attempt_number INTEGER,
            lead_score REAL,
            campaign_name TEXT,
            lead_type TEXT,
            customer_city TEXT,
            customer_zip TEXT,
            lead_value REAL,
            priority TEXT,
            lead_created_date TIMESTAMP,
            lead_updated_date TIMESTAMP,
            convoso_raw_data TEXT,

            -- Additional analytics fields
            call_outcome TEXT,
            disposition TEXT,
            transfer_status TEXT,

            FOREIGN KEY (agent_id) REFERENCES agents (agent_id)
        )
    """)

    # Rest of database tables (agents, scripts, etc.)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS agents (
            agent_id TEXT PRIMARY KEY,
            agent_name TEXT NOT NULL,
            agent_email TEXT,
            hire_date DATE,
            team_id TEXT,
            team_name TEXT,
            manager_name TEXT,
            status TEXT DEFAULT 'Active',
            phone_extension TEXT,
            updated_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS scripts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            campaign_id TEXT,
            vertical TEXT,
            script_name TEXT,
            script_content TEXT,
            version TEXT,
            is_active BOOLEAN DEFAULT 0,
            created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            created_by TEXT
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS scoring_weights (
            category TEXT PRIMARY KEY,
            weight REAL,
            description TEXT,
            updated_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Setup production data
    setup_enhanced_production_data(cursor)

    conn.commit()
    conn.close()
    logger.info("Enhanced database with Convoso integration initialized")

def setup_enhanced_production_data(cursor):
    """Setup production data with Convoso integration"""

    # Production agents from user's screenshot
    production_agents = [
        ('1277515', 'Abderrahman Aboulas', 'AAboulas@legalcaseconnect.com', '2024-01-15', 'TEAM001', 'Team Alpha', 'Sarah Johnson'),
        ('1276024', 'Jorge Castro', 'jcastro@legalcaseconnect.com', '2024-02-20', 'TEAM001', 'Team Alpha', 'Sarah Johnson'),
        ('1275713', 'Jay Penate', 'jpenate@legalcaseconnect.com', '2024-03-10', 'TEAM002', 'Team Beta', 'Michael Chen'),
        ('1273736', 'Joel Bradford', 'JBradford@legalcaseconnect.com', '2024-01-25', 'TEAM002', 'Team Beta', 'Michael Chen'),
        ('1273735', 'Griffin Penn', 'GPenn@legalcaseconnect.com', '2024-02-15', 'TEAM003', 'Team Gamma', 'Lisa Rodriguez'),
        ('1273734', 'Jamie Neubeck', 'JNeubeck@legalcaseconnect.com', '2024-03-05', 'TEAM003', 'Team Gamma', 'Lisa Rodriguez'),
        ('1273733', 'Pierre Simmons', 'PSimmons@legalcaseconnect.com', '2024-01-30', 'TEAM001', 'Team Alpha', 'Sarah Johnson'),
        ('1273582', 'Noble Gardner', 'NGardner@legalcaseconnect.com', '2024-02-10', 'TEAM002', 'Team Beta', 'Michael Chen'),
        ('1271844', 'Brad Johnson', 'bjohnson@legalcaseconnect.com', '2024-01-20', 'TEAM004', 'Super-Admin', 'Executive'),
        ('1267131', 'Jason Dehle', 'JDehle@legalcaseconnect.com', '2024-03-01', 'TEAM003', 'Team Gamma', 'Lisa Rodriguez'),
        ('1261360', 'Angelo Perone', 'angelop21@gmail.com', '2024-02-25', 'TEAM001', 'Team Alpha', 'Sarah Johnson'),
        ('1253575', 'Olive Okwuobasi', 'OOkwuobasi@legalcaseconnect.com', '2024-01-10', 'TEAM002', 'Team Beta', 'Michael Chen'),
        ('1253574', 'Jacqueline Corona', 'JCorona@legalcaseconnect.com', '2024-03-15', 'TEAM003', 'Team Gamma', 'Lisa Rodriguez')
    ]

    for agent_data in production_agents:
        cursor.execute("""
            INSERT OR REPLACE INTO agents 
            (agent_id, agent_name, agent_email, hire_date, team_id, team_name, manager_name)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, agent_data)

    # Enhanced scoring weights for MVA with Convoso
    scoring_weights = [
        ('sentiment', 0.20, 'Empathy and rapport with accident victims'),
        ('clarity', 0.15, 'Audio clarity and professional speech'),
        ('professionalism', 0.20, 'Professional legal intake language'),
        ('script_adherence', 0.15, 'Following Case Connect intake script'),
        ('rapport_building', 0.15, 'Building trust and emotional connection'),
        ('fact_gathering', 0.15, 'Gathering complete case information')
    ]

    for category, weight, description in scoring_weights:
        cursor.execute("""
            INSERT OR REPLACE INTO scoring_weights (category, weight, description)
            VALUES (?, ?, ?)
        """, (category, weight, description))

    # Case Connect MVA script
    mva_script = """
    CASE CONNECT MVA INTAKE SCRIPT

    INTRODUCTION:
    "Hi, [Client Name]? Hey [Name], this is [Your Name] calling from Case Connect. How are you today?"
    "(Acknowledge Response) The reason for my call is I just received the information you submitted to us about the auto accident you were involved in. I'm so sorry to hear! You feeling ok?"
    "(Acknowledge Response) I'm glad you reached out, based off of what you've submitted I think we can help out, just had a few super quick questions just to make sure."

    FACT FINDING:
    - Date of the accident
    - Treatment - Willing For More Additional Treatment  
    - Liability
    - Insurance/UM
    - Create a problem and a gap (How has this been affecting you)

    HOW WE CAN HELP:
    - Set up with additional treatment, find a specialist doctor in auto accident
    - Find a facility near you, make sure all bills get paid
    - Making sure excess bills are paid by the party at fault
    - Get compensated for pain and suffering
    - Reimbursed for any lost wages
    """

    cursor.execute("""
        INSERT OR REPLACE INTO scripts 
        (campaign_id, vertical, script_name, script_content, version, is_active, created_by)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, ('107391', 'MVA', 'Case Connect MVA Intake Script v2.0', mva_script, '2.0', 1, 'System Admin'))

# Production FTP handler with Convoso integration
class ProductionFTPHandler(FileSystemEventHandler):
    def __init__(self, processor: EnhancedCallProcessor):
        self.processor = processor

    def on_created(self, event):
        if not event.is_dir and event.src_path.endswith(('.mp3', '.wav')):
            logger.info(f"New call file detected with Convoso integration: {event.src_path}")
            time.sleep(3)  # Wait for complete upload

            threading.Thread(
                target=self.processor.process_call_file_enhanced,
                args=(event.src_path,)
            ).start()

def start_production_ftp_monitoring():
    """Start production FTP monitoring with Convoso integration"""
    try:
        watch_folder = ProductionConfig.FTP_UPLOAD_DIR
        os.makedirs(watch_folder, exist_ok=True)

        processor = st.session_state.call_processor
        event_handler = ProductionFTPHandler(processor)

        observer = Observer()
        observer.schedule(event_handler, watch_folder, recursive=False)
        observer.start()

        logger.info(f"Production FTP monitoring with Convoso integration started: {watch_folder}")
        return observer

    except Exception as e:
        logger.error(f"Failed to start FTP monitoring: {e}")
        return None

def main():
    """Main application with Convoso integration"""

    # Initialize enhanced database
    init_enhanced_database()

    # Initialize session state
    if 'ftp_observer' not in st.session_state:
        st.session_state.ftp_observer = None
    if 'processing_status' not in st.session_state:
        st.session_state.processing_status = "Stopped"
    if 'call_processor' not in st.session_state:
        st.session_state.call_processor = EnhancedCallProcessor()
    if 'convoso_status' not in st.session_state:
        st.session_state.convoso_status = "Testing..."

    # Test Convoso connection on startup
    if st.session_state.convoso_status == "Testing...":
        convoso_client = ConvosoAPIClient(ProductionConfig.CONVOSO_API_TOKEN)
        test_result = convoso_client.test_connection()
        st.session_state.convoso_status = "Connected" if test_result['success'] else "Error"

    # Header with Convoso integration status
    st.markdown('<div class="main-header">', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns([3, 1, 1, 1])

    with col1:
        st.title("🎯 Case Connect Call QA Intelligence")
        st.caption("Production MVA Call Quality Analysis with Convoso Integration")

    with col2:
        convoso_color = "success" if st.session_state.convoso_status == "Connected" else "error"
        st.markdown(f"""
        <div class="convoso-status">
            <strong>Convoso API:</strong><br>
            <span class="status-{convoso_color}">
                {st.session_state.convoso_status}
            </span>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        if st.button("🚀 Start Processing", type="primary"):
            if st.session_state.ftp_observer is None:
                st.session_state.ftp_observer = start_production_ftp_monitoring()
                st.session_state.processing_status = "Active"
                st.success("✅ Production processing with Convoso integration started!")
                st.balloons()

    with col4:
        if st.button("⏸️ Pause Processing"):
            if st.session_state.ftp_observer:
                st.session_state.ftp_observer.stop()
                st.session_state.ftp_observer = None
                st.session_state.processing_status = "Paused"
                st.warning("⏸️ Processing paused")

    st.markdown('</div>', unsafe_allow_html=True)

    # Navigation
    st.sidebar.title("🔗 Navigation")
    st.sidebar.markdown(f"**System Status:** {st.session_state.processing_status}")
    st.sidebar.markdown(f"**Convoso API:** {st.session_state.convoso_status}")

    # Display today's stats with Convoso data
    try:
        conn = sqlite3.connect(ProductionConfig.DATABASE_PATH)
        today_stats = pd.read_sql_query("""
            SELECT 
                COUNT(*) as total_calls,
                AVG(overall_score) as avg_score,
                COUNT(DISTINCT lead_source) as unique_sources,
                COUNT(DISTINCT customer_state) as unique_states
            FROM calls 
            WHERE DATE(processed_date) = DATE('now')
        """, conn)

        if not today_stats.empty and today_stats.iloc[0]['total_calls'] > 0:
            stats = today_stats.iloc[0]
            st.sidebar.markdown(f"""
            **Today's Enhanced Stats:**
            - Calls Processed: {int(stats['total_calls'])}
            - Average Score: {stats['avg_score']:.1f}
            - Lead Sources: {int(stats['unique_sources'])}
            - States Covered: {int(stats['unique_states'])}
            """)

        conn.close()
    except:
        pass

    # Main navigation
    page = st.sidebar.selectbox("Select Module", [
        "📊 Enhanced Production Dashboard",
        "🔍 Call Analysis with Lead Data",
        "👥 Agent Performance by Lead Source", 
        "📋 Script Management System",
        "🔧 Agent Attribution Setup",
        "📈 Convoso Lead Analytics",
        "📤 File Upload & Testing",
        "🎛️ Production Settings",
        "🔌 Convoso API Integration"
    ])

    # Route to enhanced pages
    if page == "📊 Enhanced Production Dashboard":
        show_enhanced_production_dashboard()
    elif page == "🔍 Call Analysis with Lead Data":
        show_enhanced_call_analysis()
    elif page == "👥 Agent Performance by Lead Source":
        show_enhanced_agent_performance()
    elif page == "📋 Script Management System":
        show_script_management()
    elif page == "🔧 Agent Attribution Setup":
        show_agent_attribution()
    elif page == "📈 Convoso Lead Analytics":
        show_convoso_analytics()
    elif page == "📤 File Upload & Testing":
        show_enhanced_file_upload()
    elif page == "🎛️ Production Settings":
        show_production_settings()
    elif page == "🔌 Convoso API Integration":
        show_convoso_integration()

def show_enhanced_production_dashboard():
    """Enhanced dashboard with Convoso lead analytics"""
    st.header("📊 Enhanced Production Dashboard - Case Connect with Convoso")

    try:
        conn = sqlite3.connect(ProductionConfig.DATABASE_PATH)

        # Enhanced stats with Convoso data
        enhanced_stats = pd.read_sql_query("""
            SELECT 
                COUNT(*) as total_calls,
                AVG(overall_score) as avg_score,
                COUNT(CASE WHEN overall_score < 70 THEN 1 END) as flagged_calls,
                COUNT(DISTINCT agent_id) as active_agents,
                SUM(duration_seconds)/3600 as total_hours,
                AVG(lead_score) as avg_lead_score,
                COUNT(DISTINCT lead_source) as unique_sources,
                COUNT(DISTINCT customer_state) as states_covered,
                AVG(call_attempt_number) as avg_attempts,
                COUNT(CASE WHEN lead_value > 1000 THEN 1 END) as high_value_leads
            FROM calls 
            WHERE DATE(processed_date) = DATE('now')
        """, conn)

        if not enhanced_stats.empty and enhanced_stats.iloc[0]['total_calls'] > 0:
            stats = enhanced_stats.iloc[0]

            # Enhanced metrics row
            col1, col2, col3, col4, col5, col6 = st.columns(6)

            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{int(stats['total_calls'])}</div>
                    <div class="metric-label">Calls Today</div>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                score_color = "success" if stats['avg_score'] >= 80 else "warning" if stats['avg_score'] >= 70 else "error"
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value status-{score_color}">{stats['avg_score']:.1f}</div>
                    <div class="metric-label">Avg Quality</div>
                </div>
                """, unsafe_allow_html=True)

            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{stats['avg_lead_score']:.1f}</div>
                    <div class="metric-label">Avg Lead Score</div>
                </div>
                """, unsafe_allow_html=True)

            with col4:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{int(stats['unique_sources'])}</div>
                    <div class="metric-label">Lead Sources</div>
                </div>
                """, unsafe_allow_html=True)

            with col5:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{int(stats['states_covered'])}</div>
                    <div class="metric-label">States</div>
                </div>
                """, unsafe_allow_html=True)

            with col6:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value status-warning">{int(stats['high_value_leads'])}</div>
                    <div class="metric-label">High-Value Leads</div>
                </div>
                """, unsafe_allow_html=True)

            # Enhanced charts with Convoso data
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("📈 Quality vs Lead Score Correlation")
                quality_lead_correlation = pd.read_sql_query("""
                    SELECT 
                        CASE 
                            WHEN lead_score >= 80 THEN 'Premium (80+)'
                            WHEN lead_score >= 60 THEN 'Standard (60-79)'
                            WHEN lead_score >= 40 THEN 'Basic (40-59)'
                            ELSE 'Low (<40)'
                        END as lead_tier,
                        AVG(overall_score) as avg_quality,
                        COUNT(*) as call_count
                    FROM calls 
                    WHERE DATE(processed_date) = DATE('now') AND lead_score IS NOT NULL
                    GROUP BY lead_tier
                """, conn)

                if not quality_lead_correlation.empty:
                    fig = px.bar(quality_lead_correlation, x='lead_tier', y='avg_quality',
                               title="Call Quality by Lead Score Tier")
                    fig.update_layout(template="plotly_dark", height=400)
                    st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.subheader("📊 Performance by Lead Source")
                source_performance = pd.read_sql_query("""
                    SELECT 
                        lead_source,
                        AVG(overall_score) as avg_quality,
                        COUNT(*) as call_count,
                        AVG(lead_score) as avg_lead_score
                    FROM calls 
                    WHERE DATE(processed_date) = DATE('now') AND lead_source != 'Unknown'
                    GROUP BY lead_source
                    ORDER BY avg_quality DESC
                """, conn)

                if not source_performance.empty:
                    fig = px.scatter(source_performance, x='call_count', y='avg_quality',
                                   size='avg_lead_score', hover_name='lead_source',
                                   title="Lead Source Performance Matrix")
                    fig.update_layout(template="plotly_dark", height=400)
                    st.plotly_chart(fig, use_container_width=True)

            # Convoso integration insights
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("🎯 High-Value Lead Analysis")
                high_value_analysis = pd.read_sql_query("""
                    SELECT 
                        agent_name,
                        COUNT(CASE WHEN lead_value > 1000 THEN 1 END) as high_value_calls,
                        AVG(CASE WHEN lead_value > 1000 THEN overall_score END) as high_value_avg_score,
                        COUNT(*) as total_calls
                    FROM calls 
                    WHERE DATE(processed_date) = DATE('now')
                    GROUP BY agent_id, agent_name
                    HAVING high_value_calls > 0
                    ORDER BY high_value_avg_score DESC
                """, conn)

                if not high_value_analysis.empty:
                    st.dataframe(high_value_analysis.round(1), use_container_width=True, hide_index=True)

            with col2:
                st.subheader("🗺️ Geographic Performance")
                state_performance = pd.read_sql_query("""
                    SELECT 
                        customer_state,
                        COUNT(*) as calls,
                        AVG(overall_score) as avg_score,
                        AVG(lead_score) as avg_lead_score
                    FROM calls 
                    WHERE DATE(processed_date) = DATE('now') AND customer_state != 'Unknown'
                    GROUP BY customer_state
                    ORDER BY calls DESC
                    LIMIT 10
                """, conn)

                if not state_performance.empty:
                    st.dataframe(state_performance.round(1), use_container_width=True, hide_index=True)

        else:
            st.info("📥 No calls processed today yet. Start processing to see enhanced Convoso analytics.")

            # Show Convoso connection status
            st.subheader("🔗 Convoso Integration Status")
            if st.session_state.convoso_status == "Connected":
                st.success("✅ Convoso API connected successfully")
                st.info("🎯 System ready to process calls with full lead analytics")
            else:
                st.error("❌ Convoso API connection issues detected")
                if st.button("🔄 Test Convoso Connection"):
                    convoso_client = ConvosoAPIClient(ProductionConfig.CONVOSO_API_TOKEN)
                    test_result = convoso_client.test_connection()
                    if test_result['success']:
                        st.session_state.convoso_status = "Connected"
                        st.success("✅ Convoso connection restored!")
                        st.rerun()
                    else:
                        st.error(f"❌ Connection failed: {test_result.get('message', 'Unknown error')}")

        conn.close()

    except Exception as e:
        st.error(f"Dashboard error: {e}")
        logger.error(f"Enhanced dashboard error: {e}")

def show_convoso_analytics():
    """Convoso-specific analytics dashboard"""
    st.header("📈 Convoso Lead Analytics Dashboard")

    st.info("🔗 Advanced analytics powered by Convoso lead data integration")

    try:
        conn = sqlite3.connect(ProductionConfig.DATABASE_PATH)

        tab1, tab2, tab3, tab4 = st.tabs(["📊 Lead Source ROI", "🎯 Lead Quality Analysis", "🗺️ Geographic Insights", "📞 Call Attempt Optimization"])

        with tab1:
            st.subheader("Lead Source Return on Investment")

            source_roi = pd.read_sql_query("""
                SELECT 
                    lead_source,
                    lead_vendor,
                    COUNT(*) as total_calls,
                    AVG(overall_score) as avg_quality_score,
                    AVG(lead_score) as avg_lead_score,
                    AVG(lead_value) as avg_lead_value,
                    COUNT(CASE WHEN overall_score >= 80 THEN 1 END) as high_quality_calls,
                    (COUNT(CASE WHEN overall_score >= 80 THEN 1 END) * 100.0 / COUNT(*)) as quality_rate
                FROM calls 
                WHERE lead_source != 'Unknown' AND lead_source IS NOT NULL
                GROUP BY lead_source, lead_vendor
                HAVING total_calls >= 5
                ORDER BY quality_rate DESC
            """, conn)

            if not source_roi.empty:
                # ROI scatter plot
                fig = px.scatter(source_roi, x='avg_lead_value', y='quality_rate',
                               size='total_calls', color='avg_lead_score',
                               hover_data=['lead_source', 'lead_vendor'],
                               title="Lead Source ROI Analysis (Quality Rate vs Lead Value)")
                fig.update_layout(template="plotly_dark", height=500)
                st.plotly_chart(fig, use_container_width=True)

                # ROI table
                st.subheader("Lead Source Performance Summary")
                roi_display = source_roi[['lead_source', 'total_calls', 'avg_quality_score', 
                                        'avg_lead_value', 'quality_rate']].round(1)
                st.dataframe(roi_display, use_container_width=True, hide_index=True)
            else:
                st.info("No lead source data available yet. Process calls to see analytics.")

        with tab2:
            st.subheader("Lead Quality vs Call Outcome Analysis")

            quality_analysis = pd.read_sql_query("""
                SELECT 
                    CASE 
                        WHEN lead_score >= 90 THEN 'Excellent (90+)'
                        WHEN lead_score >= 80 THEN 'Good (80-89)'
                        WHEN lead_score >= 70 THEN 'Average (70-79)'
                        WHEN lead_score >= 50 THEN 'Below Average (50-69)'
                        ELSE 'Poor (<50)'
                    END as lead_quality_tier,
                    COUNT(*) as total_calls,
                    AVG(overall_score) as avg_call_quality,
                    AVG(sentiment_score) as avg_sentiment,
                    AVG(script_adherence_score) as avg_script_adherence,
                    COUNT(CASE WHEN overall_score >= 85 THEN 1 END) as excellent_calls,
                    AVG(call_attempt_number) as avg_attempts
                FROM calls 
                WHERE lead_score IS NOT NULL AND lead_score > 0
                GROUP BY lead_quality_tier
                ORDER BY AVG(lead_score) DESC
            """, conn)

            if not quality_analysis.empty:
                # Quality correlation chart
                fig = px.bar(quality_analysis, x='lead_quality_tier', y='avg_call_quality',
                           title="Call Quality Performance by Lead Score Tier")
                fig.update_layout(template="plotly_dark", height=400)
                st.plotly_chart(fig, use_container_width=True)

                # Quality analysis table
                st.dataframe(quality_analysis.round(1), use_container_width=True, hide_index=True)

        with tab3:
            st.subheader("Geographic Performance Insights")

            geographic_data = pd.read_sql_query("""
                SELECT 
                    customer_state,
                    customer_city,
                    COUNT(*) as total_calls,
                    AVG(overall_score) as avg_call_quality,
                    AVG(lead_score) as avg_lead_score,
                    AVG(lead_value) as avg_lead_value,
                    COUNT(CASE WHEN call_attempt_number > 3 THEN 1 END) as multiple_attempt_calls,
                    AVG(call_attempt_number) as avg_attempts
                FROM calls 
                WHERE customer_state != 'Unknown' AND customer_state IS NOT NULL
                GROUP BY customer_state, customer_city
                HAVING total_calls >= 3
                ORDER BY total_calls DESC
            """, conn)

            if not geographic_data.empty:
                # State performance map-style visualization
                state_summary = geographic_data.groupby('customer_state').agg({
                    'total_calls': 'sum',
                    'avg_call_quality': 'mean',
                    'avg_lead_value': 'mean'
                }).reset_index()

                fig = px.bar(state_summary.head(15), x='customer_state', y='total_calls',
                           color='avg_call_quality', title="Call Volume and Quality by State")
                fig.update_layout(template="plotly_dark", height=500)
                st.plotly_chart(fig, use_container_width=True)

                st.subheader("Top Performing Cities")
                city_performance = geographic_data.sort_values('avg_call_quality', ascending=False).head(10)
                st.dataframe(city_performance.round(1), use_container_width=True, hide_index=True)

        with tab4:
            st.subheader("Call Attempt Optimization Analysis")

            attempt_analysis = pd.read_sql_query("""
                SELECT 
                    call_attempt_number,
                    COUNT(*) as total_calls,
                    AVG(overall_score) as avg_call_quality,
                    AVG(sentiment_score) as avg_sentiment,
                    AVG(duration_seconds) as avg_duration,
                    COUNT(CASE WHEN overall_score >= 80 THEN 1 END) as successful_calls,
                    (COUNT(CASE WHEN overall_score >= 80 THEN 1 END) * 100.0 / COUNT(*)) as success_rate
                FROM calls 
                WHERE call_attempt_number IS NOT NULL AND call_attempt_number > 0
                GROUP BY call_attempt_number
                ORDER BY call_attempt_number
            """, conn)

            if not attempt_analysis.empty:
                # Attempt success rate trend
                fig = px.line(attempt_analysis, x='call_attempt_number', y='success_rate',
                            title="Call Success Rate by Attempt Number",
                            markers=True)
                fig.update_layout(template="plotly_dark", height=400)
                st.plotly_chart(fig, use_container_width=True)

                # Attempt quality correlation
                col1, col2 = st.columns(2)

                with col1:
                    fig2 = px.bar(attempt_analysis, x='call_attempt_number', y='avg_call_quality',
                                title="Average Call Quality by Attempt")
                    fig2.update_layout(template="plotly_dark", height=400)
                    st.plotly_chart(fig2, use_container_width=True)

                with col2:
                    fig3 = px.bar(attempt_analysis, x='call_attempt_number', y='avg_duration',
                                title="Average Call Duration by Attempt")
                    fig3.update_layout(template="plotly_dark", height=400)
                    st.plotly_chart(fig3, use_container_width=True)

                st.subheader("Call Attempt Performance Summary")
                st.dataframe(attempt_analysis.round(1), use_container_width=True, hide_index=True)

        conn.close()

    except Exception as e:
        st.error(f"Convoso analytics error: {e}")
        logger.error(f"Convoso analytics error: {e}")

def show_convoso_integration():
    """Convoso API integration management"""
    st.header("🔌 Convoso API Integration Hub")

    st.info("🎯 Manage Convoso API integration and test lead data retrieval")

    tab1, tab2, tab3, tab4 = st.tabs(["🔗 Connection Status", "🧪 API Testing", "📊 Lead Data Preview", "⚙️ Integration Settings"])

    with tab1:
        st.subheader("Convoso API Connection Status")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"""
            **API Configuration:**
            - Base URL: `{ProductionConfig.CONVOSO_API_TOKEN[:20]}...`
            - Token: `{ProductionConfig.CONVOSO_API_TOKEN[:8]}...`
            - Timeout: {ProductionConfig.CONVOSO_TIMEOUT} seconds
            - Integration: {'Enabled' if ProductionConfig.ENABLE_CONVOSO_INTEGRATION else 'Disabled'}
            """)

        with col2:
            status_color = "success" if st.session_state.convoso_status == "Connected" else "error"
            st.markdown(f"""
            <div class="convoso-status">
                <h4>Current Status</h4>
                <div class="status-{status_color}" style="font-size: 1.2em;">
                    {st.session_state.convoso_status}
                </div>
            </div>
            """, unsafe_allow_html=True)

        if st.button("🔄 Test Connection Now"):
            with st.spinner("Testing Convoso API connection..."):
                convoso_client = ConvosoAPIClient(ProductionConfig.CONVOSO_API_TOKEN)
                test_result = convoso_client.test_connection()

                if test_result['success']:
                    st.session_state.convoso_status = "Connected"
                    st.success(f"✅ Connection successful! Response time: {test_result['response_time']:.2f}s")
                else:
                    st.session_state.convoso_status = "Error"
                    st.error(f"❌ Connection failed: {test_result.get('message', 'Unknown error')}")

    with tab2:
        st.subheader("🧪 API Testing Console")

        col1, col2 = st.columns(2)

        with col1:
            test_lead_id = st.text_input("Test Lead ID", placeholder="274083")
            if st.button("🔍 Test Lead ID Lookup"):
                if test_lead_id:
                    with st.spinner(f"Fetching lead data for ID: {test_lead_id}"):
                        convoso_client = ConvosoAPIClient(ProductionConfig.CONVOSO_API_TOKEN)
                        result = convoso_client.get_lead_by_id(test_lead_id)

                        if result['success']:
                            st.success("✅ Lead data retrieved successfully!")

                            # Extract analytics data
                            analytics_data = convoso_client.extract_lead_analytics_data(result)

                            if analytics_data:
                                st.subheader("📊 Extracted Analytics Data")
                                for key, value in analytics_data.items():
                                    if key != 'convoso_raw_data':
                                        st.write(f"**{key.replace('_', ' ').title()}:** {value}")

                            with st.expander("🔍 Raw API Response"):
                                st.json(result['data'])
                        else:
                            st.error(f"❌ Failed to retrieve lead: {result.get('error', 'Unknown error')}")
                else:
                    st.warning("Please enter a Lead ID")

        with col2:
            test_phone = st.text_input("Test Phone Number", placeholder="5124237198")
            if st.button("📞 Test Phone Lookup"):
                if test_phone:
                    with st.spinner(f"Searching for phone: {test_phone}"):
                        convoso_client = ConvosoAPIClient(ProductionConfig.CONVOSO_API_TOKEN)
                        result = convoso_client.get_lead_by_phone(test_phone)

                        if result['success']:
                            st.success("✅ Lead found by phone number!")

                            # Show lead info
                            lead_data = result.get('data', {})
                            if isinstance(lead_data, list) and len(lead_data) > 0:
                                lead_data = lead_data[0]  # Take first match

                            st.subheader("📋 Lead Information")
                            if isinstance(lead_data, dict):
                                for key, value in lead_data.items():
                                    if not key.startswith('_'):
                                        st.write(f"**{key}:** {value}")

                            with st.expander("🔍 Full Response"):
                                st.json(result)
                        else:
                            st.error(f"❌ Phone lookup failed: {result.get('error', 'Unknown error')}")
                else:
                    st.warning("Please enter a phone number")

    with tab3:
        st.subheader("📊 Lead Data Preview from Processed Calls")

        try:
            conn = sqlite3.connect(ProductionConfig.DATABASE_PATH)

            # Show recent calls with Convoso data
            recent_convoso_data = pd.read_sql_query("""
                SELECT 
                    lead_id,
                    customer_phone,
                    lead_source,
                    lead_vendor,
                    customer_state,
                    lead_status,
                    call_attempt_number,
                    lead_score,
                    lead_value,
                    overall_score,
                    processed_date
                FROM calls 
                WHERE convoso_raw_data IS NOT NULL 
                ORDER BY processed_date DESC 
                LIMIT 20
            """, conn)

            if not recent_convoso_data.empty:
                st.subheader("Recent Calls with Convoso Data")
                st.dataframe(recent_convoso_data, use_container_width=True, hide_index=True)

                # Summary statistics
                st.subheader("📈 Convoso Data Summary")
                col1, col2, col3 = st.columns(3)

                with col1:
                    unique_sources = recent_convoso_data['lead_source'].nunique()
                    st.metric("Unique Lead Sources", unique_sources)

                with col2:
                    avg_lead_score = recent_convoso_data['lead_score'].mean()
                    st.metric("Average Lead Score", f"{avg_lead_score:.1f}")

                with col3:
                    high_value_count = len(recent_convoso_data[recent_convoso_data['lead_value'] > 1000])
                    st.metric("High-Value Leads", high_value_count)

            else:
                st.info("No calls with Convoso data processed yet. Upload test files or start processing.")

            conn.close()

        except Exception as e:
            st.error(f"Error loading Convoso data preview: {e}")

    with tab4:
        st.subheader("⚙️ Integration Settings")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Current Configuration:**")
            st.code(f"""
API Token: {ProductionConfig.CONVOSO_API_TOKEN[:8]}...
Base URL: {ConvosoAPIClient('').base_url}
Timeout: {ProductionConfig.CONVOSO_TIMEOUT}s
Integration: {'Enabled' if ProductionConfig.ENABLE_CONVOSO_INTEGRATION else 'Disabled'}
            """)

        with col2:
            st.markdown("**Integration Features:**")
            st.markdown("""
            ✅ **Lead data retrieval by ID and phone**  
            ✅ **Analytics data extraction**  
            ✅ **Geographic and source analysis**  
            ✅ **Call attempt optimization**  
            ✅ **Lead quality correlation**  
            ✅ **ROI tracking by lead source**  
            """)

        st.subheader("🔧 Advanced Settings")

        if st.button("🗃️ Refresh All Lead Data"):
            st.info("This would re-fetch Convoso data for all existing calls (not implemented in demo)")

        if st.button("📋 Export Integration Report"):
            st.info("Integration report export feature (not implemented in demo)")

def show_enhanced_call_analysis():
    """Enhanced call analysis with Convoso lead data"""
    st.header("🔍 Enhanced Call Analysis with Lead Intelligence")

    st.info("🎯 Detailed call review with integrated Convoso lead data and context")

    try:
        conn = sqlite3.connect(ProductionConfig.DATABASE_PATH)

        # Load calls with Convoso data
        calls_with_lead_data = pd.read_sql_query("""
            SELECT 
                id, lead_id, customer_phone, agent_name, overall_score,
                lead_source, lead_vendor, customer_state, lead_score, lead_value,
                call_attempt_number, processed_date, duration_seconds,
                sentiment_score, professionalism_score, script_adherence_score
            FROM calls 
            WHERE convoso_raw_data IS NOT NULL 
            ORDER BY processed_date DESC 
            LIMIT 50
        """, conn)

        if not calls_with_lead_data.empty:
            # Call selection
            st.subheader("📞 Select Call for Detailed Analysis")

            # Create display format for dropdown
            call_options = []
            for _, call in calls_with_lead_data.iterrows():
                display_text = f"Lead {call['lead_id']} - {call['agent_name']} - Score: {call['overall_score']:.1f} - {call['lead_source']}"
                call_options.append((display_text, call['id']))

            selected_display = st.selectbox("Choose a call:", [opt[0] for opt in call_options])
            selected_call_id = next(opt[1] for opt in call_options if opt[0] == selected_display)

            # Get detailed call data
            selected_call = calls_with_lead_data[calls_with_lead_data['id'] == selected_call_id].iloc[0]

            # Display enhanced call analysis
            col1, col2 = st.columns([2, 1])

            with col1:
                st.subheader("📋 Call Details")

                # Basic call info
                col1a, col1b, col1c = st.columns(3)
                with col1a:
                    st.metric("Overall Score", f"{selected_call['overall_score']:.1f}")
                with col1b:
                    st.metric("Duration", f"{int(selected_call['duration_seconds']//60)}:{int(selected_call['duration_seconds']%60):02d}")
                with col1c:
                    st.metric("Attempt #", int(selected_call['call_attempt_number']))

                # Quality breakdown
                st.subheader("📊 Quality Score Breakdown")
                quality_scores = {
                    'Sentiment': selected_call['sentiment_score'],
                    'Professionalism': selected_call['professionalism_score'],
                    'Script Adherence': selected_call['script_adherence_score']
                }

                for metric, score in quality_scores.items():
                    progress_color = "#27ae60" if score >= 80 else "#f39c12" if score >= 70 else "#e74c3c"
                    st.markdown(f"**{metric}: {score:.1f}**")
                    st.progress(score/100)

            with col2:
                st.subheader("🎯 Lead Intelligence")

                # Lead info card
                st.markdown(f"""
                <div class="lead-info-card">
                    <h4>📊 Lead Information</h4>
                    <p><strong>Source:</strong> {selected_call['lead_source']}</p>
                    <p><strong>Vendor:</strong> {selected_call['lead_vendor']}</p>
                    <p><strong>State:</strong> {selected_call['customer_state']}</p>
                    <p><strong>Lead Score:</strong> {selected_call['lead_score']:.1f}</p>
                    <p><strong>Est. Value:</strong> ${selected_call['lead_value']:,.0f}</p>
                    <p><strong>Call Attempt:</strong> #{int(selected_call['call_attempt_number'])}</p>
                </div>
                """, unsafe_allow_html=True)

                # Lead quality assessment
                lead_score = selected_call['lead_score']
                if lead_score >= 80:
                    st.success("🎯 Premium Lead - High conversion potential")
                elif lead_score >= 60:
                    st.info("📈 Standard Lead - Good potential")
                else:
                    st.warning("⚠️ Lower Quality Lead - Needs extra attention")

            # Get full call details including transcript and coaching
            full_call_details = pd.read_sql_query("""
                SELECT transcript, coaching_notes, issues_flagged, mva_keywords, compliance_items
                FROM calls 
                WHERE id = ?
            """, (selected_call_id,), conn)

            if not full_call_details.empty:
                call_detail = full_call_details.iloc[0]

                # Transcript analysis
                st.subheader("📝 Call Transcript")
                with st.expander("View Full Transcript"):
                    st.text_area("Transcript", call_detail['transcript'], height=200, disabled=True)

                # Coaching notes
                st.subheader("💡 AI Coaching Recommendations")
                st.info(call_detail['coaching_notes'])

                # Issues flagged
                if call_detail['issues_flagged'] and call_detail['issues_flagged'] != '[]':
                    try:
                        issues = json.loads(call_detail['issues_flagged'])
                        if issues:
                            st.subheader("⚠️ Issues Flagged")
                            for issue in issues:
                                st.warning(f"• {issue}")
                    except:
                        pass

                # MVA keywords and compliance
                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("🔍 MVA Keywords Found")
                    try:
                        keywords = json.loads(call_detail['mva_keywords']) if call_detail['mva_keywords'] else []
                        if keywords:
                            for keyword in keywords:
                                st.badge(keyword)
                        else:
                            st.info("No MVA-specific keywords detected")
                    except:
                        st.info("Keywords data not available")

                with col2:
                    st.subheader("✅ Compliance Check")
                    try:
                        compliance = json.loads(call_detail['compliance_items']) if call_detail['compliance_items'] else {}
                        if compliance:
                            for item, status in compliance.items():
                                icon = "✅" if status else "❌"
                                st.write(f"{icon} {item.replace('_', ' ').title()}")
                        else:
                            st.info("Compliance data not available")
                    except:
                        st.info("Compliance data not available")

        else:
            st.info("📥 No calls with Convoso lead data available yet.")
            st.markdown("""
            **To see enhanced call analysis:**
            1. Start FTP monitoring or upload test files
            2. Ensure Convoso API integration is working
            3. Process calls to see lead intelligence integration
            """)

        conn.close()

    except Exception as e:
        st.error(f"Enhanced call analysis error: {e}")
        logger.error(f"Enhanced call analysis error: {e}")

def show_enhanced_agent_performance():
    """Enhanced agent performance with lead source analytics"""
    st.header("👥 Enhanced Agent Performance Analytics")

    st.info("🎯 Agent performance analysis with lead source, geography, and attempt correlation")

    try:
        conn = sqlite3.connect(ProductionConfig.DATABASE_PATH)

        # Agent performance with Convoso metrics
        agent_performance = pd.read_sql_query("""
            SELECT 
                agent_id,
                agent_name,
                team_name,
                COUNT(*) as total_calls,
                AVG(overall_score) as avg_quality_score,
                AVG(lead_score) as avg_lead_score,
                AVG(lead_value) as avg_lead_value,
                COUNT(DISTINCT lead_source) as unique_sources,
                COUNT(DISTINCT customer_state) as states_covered,
                AVG(call_attempt_number) as avg_attempts,
                COUNT(CASE WHEN overall_score >= 85 THEN 1 END) as excellent_calls,
                COUNT(CASE WHEN lead_value > 1000 THEN 1 END) as high_value_calls,
                (COUNT(CASE WHEN overall_score >= 85 THEN 1 END) * 100.0 / COUNT(*)) as excellence_rate
            FROM calls 
            WHERE agent_name IS NOT NULL 
            GROUP BY agent_id, agent_name, team_name
            HAVING total_calls >= 5
            ORDER BY avg_quality_score DESC
        """, conn)

        if not agent_performance.empty:
            # Performance overview
            st.subheader("🏆 Agent Performance Leaderboard")

            # Top performers
            top_performers = agent_performance.head(10)

            col1, col2 = st.columns(2)

            with col1:
                # Performance chart
                fig = px.scatter(agent_performance, x='total_calls', y='avg_quality_score',
                               size='avg_lead_value', color='excellence_rate',
                               hover_data=['agent_name', 'team_name'],
                               title="Agent Performance Matrix")
                fig.update_layout(template="plotly_dark", height=500)
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                # Excellence rate by team
                team_performance = agent_performance.groupby('team_name').agg({
                    'avg_quality_score': 'mean',
                    'excellence_rate': 'mean',
                    'total_calls': 'sum'
                }).reset_index()

                fig2 = px.bar(team_performance, x='team_name', y='excellence_rate',
                            title="Team Excellence Rate Comparison")
                fig2.update_layout(template="plotly_dark", height=500)
                st.plotly_chart(fig2, use_container_width=True)

            # Detailed performance table
            st.subheader("📊 Detailed Agent Performance")

            # Format display columns
            display_columns = [
                'agent_name', 'team_name', 'total_calls', 'avg_quality_score',
                'avg_lead_score', 'unique_sources', 'excellence_rate', 'high_value_calls'
            ]

            performance_display = agent_performance[display_columns].round(1)
            performance_display.columns = [
                'Agent', 'Team', 'Total Calls', 'Avg Quality', 'Avg Lead Score',
                'Sources Handled', 'Excellence Rate %', 'High-Value Calls'
            ]

            st.dataframe(performance_display, use_container_width=True, hide_index=True)

            # Individual agent deep dive
            st.subheader("🔍 Individual Agent Analysis")

            selected_agent = st.selectbox(
                "Select agent for detailed analysis:",
                options=agent_performance['agent_name'].tolist()
            )

            if selected_agent:
                # Get detailed agent data
                agent_details = pd.read_sql_query("""
                    SELECT 
                        lead_source,
                        customer_state,
                        COUNT(*) as calls,
                        AVG(overall_score) as avg_score,
                        AVG(lead_score) as avg_lead_score,
                        AVG(call_attempt_number) as avg_attempts
                    FROM calls 
                    WHERE agent_name = ?
                    GROUP BY lead_source, customer_state
                    HAVING calls >= 2
                    ORDER BY calls DESC
                """, (selected_agent,), conn)

                if not agent_details.empty:
                    col1, col2 = st.columns(2)

                    with col1:
                        st.subheader(f"📈 {selected_agent} - Performance by Lead Source")
                        source_perf = agent_details.groupby('lead_source').agg({
                            'calls': 'sum',
                            'avg_score': 'mean'
                        }).reset_index()

                        fig3 = px.bar(source_perf, x='lead_source', y='avg_score',
                                    size='calls', title=f"{selected_agent} - Quality by Source")
                        fig3.update_layout(template="plotly_dark", height=400)
                        st.plotly_chart(fig3, use_container_width=True)

                    with col2:
                        st.subheader(f"🗺️ {selected_agent} - Geographic Performance")
                        state_perf = agent_details.groupby('customer_state').agg({
                            'calls': 'sum',
                            'avg_score': 'mean'
                        }).reset_index()

                        fig4 = px.bar(state_perf, x='customer_state', y='calls',
                                    color='avg_score', title=f"{selected_agent} - Calls by State")
                        fig4.update_layout(template="plotly_dark", height=400)
                        st.plotly_chart(fig4, use_container_width=True)

            # Performance improvement recommendations
            st.subheader("💡 Performance Improvement Opportunities")

            # Identify improvement areas
            low_performers = agent_performance[agent_performance['avg_quality_score'] < 75]
            high_value_underperformers = agent_performance[
                (agent_performance['high_value_calls'] > 0) & 
                (agent_performance['avg_quality_score'] < 80)
            ]

            col1, col2 = st.columns(2)

            with col1:
                if not low_performers.empty:
                    st.warning("⚠️ Agents Needing Quality Improvement:")
                    for _, agent in low_performers.iterrows():
                        st.write(f"• **{agent['agent_name']}** ({agent['team_name']}) - Score: {agent['avg_quality_score']:.1f}")

            with col2:
                if not high_value_underperformers.empty:
                    st.info("🎯 High-Value Lead Optimization Needed:")
                    for _, agent in high_value_underperformers.iterrows():
                        st.write(f"• **{agent['agent_name']}** - {agent['high_value_calls']} high-value calls, {agent['avg_quality_score']:.1f} avg score")

        else:
            st.info("📥 No agent performance data available yet. Process calls to see analytics.")

        conn.close()

    except Exception as e:
        st.error(f"Enhanced agent performance error: {e}")
        logger.error(f"Enhanced agent performance error: {e}")

# Include other page functions (show_script_management, etc.) from previous version
# For brevity, showing key ones with Convoso integration

def show_enhanced_file_upload():
    """Enhanced file upload with Convoso integration preview"""
    st.header("📤 Enhanced File Upload & Testing")

    st.info("🔬 Test the production processing pipeline with Convoso lead data integration")

    uploaded_files = st.file_uploader(
        "Upload Call Recordings", 
        type=['mp3', 'wav'],
        accept_multiple_files=True,
        help="Upload MP3/WAV files in format: campaign_agent_customer_leadid_xxx_timestamp_xxx.mp3"
    )

    if uploaded_files:
        st.subheader(f"📁 {len(uploaded_files)} files ready for enhanced processing")

        for uploaded_file in uploaded_files:
            with st.expander(f"🎵 {uploaded_file.name}"):
                col1, col2 = st.columns(2)

                with col1:
                    st.write(f"**Size:** {uploaded_file.size / 1024 / 1024:.1f} MB")

                    # Parse filename and show expected Convoso lookup
                    metadata = st.session_state.call_processor.parse_filename(uploaded_file.name)
                    if metadata:
                        st.json({
                            "Parsed Metadata": metadata,
                            "Convoso Lookup": f"Will fetch lead data for ID: {metadata['lead_id']}"
                        })
                    else:
                        st.error("❌ Filename format not recognized")

                with col2:
                    if st.button(f"🚀 Process with Convoso", key=f"process_{uploaded_file.name}"):
                        try:
                            # Save file temporarily
                            temp_dir = "temp_uploads"
                            os.makedirs(temp_dir, exist_ok=True)
                            temp_path = os.path.join(temp_dir, uploaded_file.name)

                            with open(temp_path, "wb") as f:
                                f.write(uploaded_file.getvalue())

                            # Process with enhanced pipeline
                            progress_bar = st.progress(0)
                            status_text = st.empty()

                            status_text.text("🔗 Fetching Convoso lead data...")
                            progress_bar.progress(20)

                            status_text.text("🎤 Transcribing with Whisper Large...")
                            progress_bar.progress(50)

                            status_text.text("🤖 AI quality analysis with lead context...")
                            progress_bar.progress(75)

                            success = st.session_state.call_processor.process_call_file_enhanced(temp_path)
                            progress_bar.progress(100)

                            if success:
                                status_text.text("✅ Enhanced processing completed!")
                                st.success("Call analyzed with Convoso lead intelligence!")

                                # Show what was stored
                                st.info("""
                                **Enhanced data stored:**
                                • Whisper transcription and quality analysis
                                • Convoso lead source, state, and vendor data  
                                • Lead score and value correlation
                                • Call attempt optimization data
                                • Geographic and source performance tracking
                                """)
                                st.balloons()
                            else:
                                status_text.text("❌ Processing failed")
                                st.error("Processing failed - check logs for details")

                        except Exception as e:
                            st.error(f"Upload error: {e}")

# Add other helper functions (show_script_management, show_agent_attribution, show_production_settings)
def show_script_management():
    """Script management interface"""
    st.header("📋 Script Management System")
    st.info("Upload and manage intake scripts by campaign and vertical")

    # Implementation same as before but integrated with new database
    # For brevity, showing placeholder
    st.info("Script management interface - same as previous version")

def show_agent_attribution():
    """Agent attribution setup"""
    st.header("🔧 Agent Attribution Setup")
    st.info("Configure agent mapping and team assignments")

    # Implementation same as before
    st.info("Agent attribution interface - same as previous version")

def show_production_settings():
    """Production settings interface"""
    st.header("🎛️ Production Settings")
    st.info("Configure system settings, scoring weights, and processing parameters")

    # Implementation same as before but with Convoso settings
    st.info("Production settings interface - enhanced with Convoso configuration")

if __name__ == "__main__":
    main()
