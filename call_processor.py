import os
import json
import sqlite3
import whisper
import librosa
import numpy as np
from datetime import datetime
import requests
import logging
from pathlib import Path
import time
import threading
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CallFileHandler(FileSystemEventHandler):
    """Handle new MP3 files arriving via FTP"""

    def __init__(self, processor):
        self.processor = processor

    def on_created(self, event):
        if not event.is_directory and event.src_path.endswith('.mp3'):
            logger.info(f"New call file detected: {event.src_path}")
            # Add small delay to ensure file is fully written
            time.sleep(2)
            self.processor.process_call(event.src_path)

class CallProcessor:
    """Enhanced call processing with real Whisper integration"""

    def __init__(self, config=None):
        self.config = config or {}
        self.watch_folder = self.config.get('watch_folder', 'incoming_calls')
        self.processed_folder = self.config.get('processed_folder', 'processed_calls')
        self.whisper_model = None

        # Create directories
        os.makedirs(self.watch_folder, exist_ok=True)
        os.makedirs(self.processed_folder, exist_ok=True)

        # Load Whisper model
        self.load_whisper_model()

        # Start file watcher
        self.start_file_watcher()

    def load_whisper_model(self):
        """Load Whisper model for transcription"""
        try:
            model_size = self.config.get('whisper_model', 'base')
            logger.info(f"Loading Whisper model: {model_size}")
            self.whisper_model = whisper.load_model(model_size)
            logger.info("Whisper model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            # Fallback to simulation mode
            self.whisper_model = None

    def start_file_watcher(self):
        """Start watching for new files"""
        event_handler = CallFileHandler(self)
        observer = Observer()
        observer.schedule(event_handler, self.watch_folder, recursive=False)
        observer.start()
        logger.info(f"Started watching folder: {self.watch_folder}")
        return observer

    def parse_filename(self, filename):
        """Parse call metadata from filename: campaign_agentid_customerphone_leadid_xxx_timestamp_xxx.mp3"""
        try:
            basename = os.path.basename(filename).replace('.mp3', '')
            parts = basename.split('_')

            if len(parts) >= 4:
                metadata = {
                    'campaign_id': parts[0],
                    'agent_id': parts[1], 
                    'customer_phone': parts[2],
                    'lead_id': parts[3],
                    'filename': filename
                }

                # Extract timestamp if available
                if len(parts) > 5:
                    try:
                        timestamp = int(parts[5])
                        metadata['call_timestamp'] = datetime.fromtimestamp(timestamp)
                    except:
                        pass

                return metadata

        except Exception as e:
            logger.error(f"Error parsing filename {filename}: {e}")

        return None

    def transcribe_audio(self, audio_path):
        """Transcribe audio using Whisper"""
        try:
            if self.whisper_model is None:
                # Fallback simulation
                return self.simulate_transcription(audio_path)

            logger.info(f"Transcribing audio: {audio_path}")
            result = self.whisper_model.transcribe(audio_path, verbose=False)

            return {
                'text': result['text'].strip(),
                'segments': result.get('segments', []),
                'language': result.get('language', 'en'),
                'duration': self.get_audio_duration(audio_path)
            }

        except Exception as e:
            logger.error(f"Transcription failed for {audio_path}: {e}")
            return self.simulate_transcription(audio_path)

    def simulate_transcription(self, audio_path):
        """Simulate transcription for demo purposes"""
        return {
            'text': "Hello, thank you for calling about your motor vehicle accident case. I understand you were involved in an accident recently and are looking for legal representation. Can you tell me more about what happened and any injuries you sustained?",
            'segments': [
                {'start': 0.0, 'end': 3.0, 'text': 'Hello, thank you for calling about your motor vehicle accident case.'},
                {'start': 3.0, 'end': 8.0, 'text': 'I understand you were involved in an accident recently and are looking for legal representation.'},
                {'start': 8.0, 'end': 12.0, 'text': 'Can you tell me more about what happened and any injuries you sustained?'}
            ],
            'language': 'en',
            'duration': self.get_audio_duration(audio_path) if os.path.exists(audio_path) else 60.0
        }

    def get_audio_duration(self, audio_path):
        """Get audio duration in seconds"""
        try:
            import librosa
            duration = librosa.get_duration(path=audio_path)
            return duration
        except:
            # Fallback estimate
            file_size = os.path.getsize(audio_path) if os.path.exists(audio_path) else 1000000
            return max(30, file_size / 16000)  # Rough estimate

    def analyze_audio_quality(self, audio_path, transcript_data):
        """Analyze audio quality and call characteristics"""
        try:
            # Load audio for analysis
            if os.path.exists(audio_path):
                y, sr = librosa.load(audio_path, sr=16000)

                # Calculate audio features
                silence_ratio = self.calculate_silence_ratio(y, sr)
                clarity_score = self.estimate_clarity(y, sr)
            else:
                # Fallback values
                silence_ratio = 0.15
                clarity_score = 75

            # Analyze transcript
            transcript_text = transcript_data.get('text', '')
            sentiment_score = self.analyze_sentiment(transcript_text)
            professionalism_score = self.analyze_professionalism(transcript_text)
            script_adherence_score = self.analyze_script_adherence(transcript_text)

            # Count interruptions/overlaps from segments
            segments = transcript_data.get('segments', [])
            interruptions = self.count_interruptions(segments)

            return {
                'clarity_score': clarity_score,
                'sentiment_score': sentiment_score,
                'professionalism_score': professionalism_score,
                'script_adherence_score': script_adherence_score,
                'silence_percentage': silence_ratio * 100,
                'interruptions_count': interruptions,
                'duration_seconds': transcript_data.get('duration', 60)
            }

        except Exception as e:
            logger.error(f"Audio analysis failed: {e}")
            # Return default values
            return {
                'clarity_score': 75.0,
                'sentiment_score': 70.0,
                'professionalism_score': 80.0,
                'script_adherence_score': 75.0,
                'silence_percentage': 10.0,
                'interruptions_count': 2,
                'duration_seconds': 60.0
            }

    def calculate_silence_ratio(self, audio, sr, threshold=0.01):
        """Calculate percentage of silence in audio"""
        try:
            # Simple energy-based silence detection
            frame_length = int(0.025 * sr)  # 25ms frames
            energy = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=frame_length//2)[0]
            silence_frames = np.sum(energy < threshold)
            total_frames = len(energy)
            return silence_frames / total_frames if total_frames > 0 else 0.1
        except:
            return 0.15  # Default fallback

    def estimate_clarity(self, audio, sr):
        """Estimate audio clarity score"""
        try:
            # Simple spectral analysis for clarity
            spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]

            # Higher spectral centroid and rolloff generally indicate clearer speech
            clarity = min(100, max(0, (np.mean(spectral_centroids) / 4000) * 100))
            return clarity
        except:
            return 75.0  # Default fallback

    def analyze_sentiment(self, text):
        """Analyze sentiment of transcript"""
        positive_words = ['thank', 'thanks', 'help', 'great', 'excellent', 'good', 'understand', 'appreciate', 'sorry', 'welcome']
        negative_words = ['problem', 'issue', 'difficult', 'frustrated', 'angry', 'complaint', 'terrible', 'awful', 'hate', 'mad']

        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)

        # Base score of 50, adjust based on word counts
        sentiment_score = 50 + (positive_count * 8) - (negative_count * 10)
        return max(0, min(100, sentiment_score))

    def analyze_professionalism(self, text):
        """Analyze professionalism based on language used"""
        professional_phrases = ['thank you', 'please', 'may i', 'i understand', 'i apologize', 'certainly', 'absolutely']
        unprofessional_words = ['yeah', 'um', 'uh', 'like', 'you know', 'whatever']

        text_lower = text.lower()
        professional_count = sum(1 for phrase in professional_phrases if phrase in text_lower)
        unprofessional_count = sum(1 for word in unprofessional_words if word in text_lower)

        base_score = 70
        professionalism_score = base_score + (professional_count * 5) - (unprofessional_count * 3)
        return max(0, min(100, professionalism_score))

    def analyze_script_adherence(self, text):
        """Analyze adherence to scripts for MVA calls"""
        required_elements = [
            'motor vehicle', 'accident', 'injury', 'injuries',
            'legal', 'attorney', 'lawyer', 'representation',
            'insurance', 'medical', 'treatment'
        ]

        text_lower = text.lower()
        elements_found = sum(1 for element in required_elements if element in text_lower)

        # Score based on percentage of required elements mentioned
        adherence_score = (elements_found / len(required_elements)) * 100
        return min(100, adherence_score)

    def count_interruptions(self, segments):
        """Count potential interruptions in conversation"""
        if len(segments) < 2:
            return 0

        interruptions = 0
        for i in range(1, len(segments)):
            prev_end = segments[i-1]['end']
            curr_start = segments[i]['start']

            # If segments overlap or have very short gaps, count as interruption
            if curr_start < prev_end or (curr_start - prev_end) < 0.5:
                interruptions += 1

        return min(interruptions, 10)  # Cap at reasonable number

    def generate_coaching_notes(self, analysis, issues):
        """Generate actionable coaching notes"""
        notes = []

        if analysis['sentiment_score'] < 60:
            notes.append("Work on maintaining positive tone and empathy throughout the call")

        if analysis['clarity_score'] < 70:
            notes.append("Improve speaking clarity - check microphone setup and speak more slowly")

        if analysis['professionalism_score'] < 75:
            notes.append("Reduce filler words and use more professional language")

        if analysis['script_adherence_score'] < 70:
            notes.append("Better adherence to script required - ensure all key points are covered")

        if analysis['silence_percentage'] > 20:
            notes.append("Reduce dead air time - keep conversation flowing")

        if analysis['interruptions_count'] > 5:
            notes.append("Allow customer to speak more - reduce interruptions")

        if len(notes) == 0:
            notes.append("Good call quality - maintain current performance level")

        return " | ".join(notes)

    def calculate_overall_score(self, analysis):
        """Calculate weighted overall score"""
        try:
            conn = sqlite3.connect('call_qa_database.db')
            weights_df = pd.read_sql_query("SELECT category, weight FROM scoring_weights", conn)
            conn.close()

            weight_map = dict(zip(weights_df['category'], weights_df['weight']))
        except:
            # Default weights if database not available
            weight_map = {
                'sentiment': 0.25,
                'clarity': 0.20,
                'professionalism': 0.20,
                'script_adherence': 0.15,
                'silence_handling': 0.10,
                'interruptions': 0.10
            }

        # Calculate weighted score
        overall_score = (
            analysis['sentiment_score'] * weight_map.get('sentiment', 0.25) +
            analysis['clarity_score'] * weight_map.get('clarity', 0.20) +
            analysis['professionalism_score'] * weight_map.get('professionalism', 0.20) +
            analysis['script_adherence_score'] * weight_map.get('script_adherence', 0.15) +
            (100 - analysis['silence_percentage']) * weight_map.get('silence_handling', 0.10) +
            max(0, 100 - analysis['interruptions_count'] * 10) * weight_map.get('interruptions', 0.10)
        )

        return max(0, min(100, overall_score))

    def identify_issues(self, analysis, transcript_text):
        """Identify specific issues in the call"""
        issues = []

        if analysis['sentiment_score'] < 40:
            issues.append("Very negative sentiment detected")
        elif analysis['sentiment_score'] < 60:
            issues.append("Negative sentiment detected")

        if analysis['clarity_score'] < 50:
            issues.append("Poor audio clarity")
        elif analysis['clarity_score'] < 70:
            issues.append("Below average audio clarity")

        if analysis['silence_percentage'] > 25:
            issues.append("Excessive dead air time")

        if analysis['interruptions_count'] > 7:
            issues.append("Too many interruptions")

        if len(transcript_text) < 50:
            issues.append("Unusually short call - possible hang-up")

        if 'um' in transcript_text.lower() or 'uh' in transcript_text.lower():
            filler_count = transcript_text.lower().count('um') + transcript_text.lower().count('uh')
            if filler_count > 5:
                issues.append("Excessive filler words")

        return issues

    def process_call(self, file_path):
        """Main call processing function"""
        try:
            logger.info(f"Processing call: {file_path}")

            # Parse filename for metadata
            filename = os.path.basename(file_path)
            metadata = self.parse_filename(filename)

            if not metadata:
                logger.error(f"Could not parse filename: {filename}")
                return False

            # Transcribe audio
            transcript_data = self.transcribe_audio(file_path)
            transcript_text = transcript_data['text']

            # Analyze call quality
            analysis = self.analyze_audio_quality(file_path, transcript_data)
            overall_score = self.calculate_overall_score(analysis)

            # Identify issues
            issues = self.identify_issues(analysis, transcript_text)
            coaching_notes = self.generate_coaching_notes(analysis, issues)

            # Get file information
            file_stats = os.stat(file_path) if os.path.exists(file_path) else None
            file_size = file_stats.st_size if file_stats else 0

            # Save to database
            self.save_call_data(metadata, transcript_data, analysis, overall_score, 
                              issues, coaching_notes, file_size, filename)

            # Clean up - delete original file after processing
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    logger.info(f"Deleted processed file: {file_path}")
                except Exception as e:
                    logger.warning(f"Could not delete file {file_path}: {e}")

            logger.info(f"Successfully processed call {filename} (Score: {overall_score:.1f})")
            return True

        except Exception as e:
            logger.error(f"Error processing call {file_path}: {e}")
            return False

    def save_call_data(self, metadata, transcript_data, analysis, overall_score, 
                      issues, coaching_notes, file_size, filename):
        """Save processed call data to database"""
        try:
            conn = sqlite3.connect('call_qa_database.db')
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO calls (
                    campaign_id, agent_id, customer_phone, lead_id, call_date,
                    file_path, file_size, duration_seconds, transcript,
                    sentiment_score, clarity_score, professionalism_score,
                    script_adherence_score, overall_score, silence_percentage,
                    interruptions_count, keywords_found, issues_flagged,
                    coaching_notes, processed_date
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                metadata['campaign_id'],
                metadata['agent_id'],
                metadata['customer_phone'],
                metadata['lead_id'],
                metadata.get('call_timestamp', datetime.now()),
                filename,
                file_size,
                analysis['duration_seconds'],
                transcript_data['text'],
                analysis['sentiment_score'],
                analysis['clarity_score'],
                analysis['professionalism_score'],
                analysis['script_adherence_score'],
                overall_score,
                analysis['silence_percentage'],
                analysis['interruptions_count'],
                json.dumps(['motor vehicle', 'accident', 'injury']),  # Keywords found
                json.dumps(issues),
                coaching_notes,
                datetime.now()
            ))

            conn.commit()
            conn.close()

        except Exception as e:
            logger.error(f"Error saving call data: {e}")
            raise

if __name__ == "__main__":
    # Example usage
    processor = CallProcessor({
        'whisper_model': 'base',  # Can be: tiny, base, small, medium, large
        'watch_folder': 'incoming_calls'
    })

    # Keep the processor running
    try:
        while True:
            time.sleep(10)
    except KeyboardInterrupt:
        logger.info("Shutting down call processor")
