# Call QA Intelligence Platform - Setup Guide

## Overview
This platform provides AI-powered call quality analysis for Motor Vehicle Accident referral services using OpenAI Whisper for transcription and advanced analytics for coaching insights.

## Features
- ✅ Automated MP3 call transcription via Whisper AI
- ✅ Real-time quality scoring (sentiment, clarity, professionalism, script adherence)
- ✅ Configurable scoring weights
- ✅ Agent performance analytics
- ✅ Issue flagging and coaching recommendations
- ✅ Export capabilities (CSV/PDF)
- ✅ Web-based dashboard (Streamlit)
- ✅ File cleanup (processes then deletes to save storage)

## Architecture
```
FTP Push → Incoming Calls Folder → Whisper Transcription → AI Analysis → Dashboard → File Cleanup
```

## Quick Deploy to Render

### Option 1: GitHub Deploy (Recommended)
1. Create new GitHub repository
2. Upload all files to repository:
   - app.py
   - call_processor.py
   - requirements.txt
   - Dockerfile
   - render.yaml

3. Connect repository to Render:
   - Go to render.com dashboard
   - Click "New +" → "Web Service"
   - Connect your GitHub repository
   - Render will auto-detect the configuration

### Option 2: Direct Deploy
1. Clone this repository
2. Install Render CLI: `npm install -g render-cli`
3. Deploy: `render deploy`

## Local Development Setup

### Prerequisites
- Python 3.9+
- FFmpeg (for audio processing)
- Git

### Installation
```bash
# Clone repository
git clone <your-repo-url>
cd call-qa-platform

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

## Configuration

### File Structure
```
call-qa-platform/
├── app.py                  # Main Streamlit application
├── call_processor.py       # Call processing and AI analysis
├── requirements.txt        # Python dependencies
├── Dockerfile             # Docker configuration
├── render.yaml            # Render deployment config
├── incoming_calls/        # FTP destination folder
├── processed_calls/       # Temporary processed files
└── call_qa_database.db    # SQLite database (auto-created)
```

### Audio File Format
Expected filename format: `campaign_agentid_customerphone_leadid_xxx_timestamp_xxx.mp3`

Example: `107391_1276024_5124237198_274083_809_1759884049_4611.mp3`
- Campaign: 107391
- Agent ID: 1276024 
- Customer Phone: 5124237198
- Lead ID: 274083

### Scoring Categories
Default weighted scoring:
- **Sentiment (25%)**: Overall call tone and positivity
- **Clarity (20%)**: Audio quality and communication clarity
- **Professionalism (20%)**: Professional language and courtesy
- **Script Adherence (15%)**: Following MVA script requirements
- **Silence Handling (10%)**: Managing dead air and pauses
- **Interruptions (10%)**: Conversation flow management

## FTP Setup for Render

### Getting FTP Access
Render doesn't provide built-in FTP, but you can:

1. **Use Render + External FTP**: Set up FTP on external service, sync to Render via cron job
2. **Use File Upload API**: Build endpoint to receive files via HTTP POST
3. **Use Cloud Storage**: Configure S3/GCP bucket, monitor for new files

### Recommended: HTTP Upload Endpoint
Add this to your app.py for direct file uploads:

```python
import streamlit as st

uploaded_file = st.file_uploader("Upload MP3 Call", type=['mp3'])
if uploaded_file:
    # Save and process immediately
    with open(f"incoming_calls/{uploaded_file.name}", "wb") as f:
        f.write(uploaded_file.getbuffer())
    processor.process_call(f"incoming_calls/{uploaded_file.name}")
```

## Environment Variables

Set these in Render dashboard or locally:

```bash
# Optional: Whisper model size (tiny, base, small, medium, large)
WHISPER_MODEL=base

# Optional: Processing folder paths  
WATCH_FOLDER=incoming_calls
PROCESSED_FOLDER=processed_calls

# Optional: Database path
DATABASE_PATH=call_qa_database.db
```

## Monitoring & Maintenance

### Logs
- Application logs: Check Render dashboard logs
- Processing status: Available in Streamlit app sidebar

### Database Backup
- SQLite database auto-created as `call_qa_database.db`
- Export data regularly via Settings page
- For production: Consider PostgreSQL upgrade

### Storage Management
- Files are automatically deleted after processing
- Only metadata and transcripts stored in database
- Can re-fetch original audio via Convoso API if needed

## Scaling & Performance

### Whisper Model Options
- **tiny**: Fastest, least accurate (~39 MB)
- **base**: Good balance (~142 MB) - **Recommended**
- **small**: Better accuracy (~466 MB)
- **medium**: High accuracy (~1.5 GB)
- **large**: Best accuracy (~3 GB)

### Resource Requirements
- **Minimum**: 1 GB RAM, 1 CPU core (tiny/base model)
- **Recommended**: 2 GB RAM, 2 CPU cores (base/small model)
- **High Volume**: 4+ GB RAM, GPU support (medium/large model)

## Troubleshooting

### Common Issues

1. **Whisper Import Error**
   ```bash
   pip install --upgrade openai-whisper
   # or
   pip install git+https://github.com/openai/whisper.git
   ```

2. **Audio Processing Errors**
   ```bash
   # Install FFmpeg
   # Ubuntu/Debian:
   sudo apt update && sudo apt install ffmpeg

   # macOS:
   brew install ffmpeg

   # Windows: Download from https://ffmpeg.org/
   ```

3. **Database Locked**
   - Restart application
   - Check file permissions
   - Ensure SQLite is properly closed in code

4. **Memory Issues**
   - Use smaller Whisper model
   - Increase server RAM
   - Process files in batches

### Performance Optimization

1. **Use GPU for Whisper** (if available):
   ```python
   model = whisper.load_model("base", device="cuda")
   ```

2. **Batch Processing**:
   - Process multiple files simultaneously
   - Use threading for I/O operations

3. **Database Optimization**:
   - Add indexes for frequently queried columns
   - Archive old data regularly
   - Consider PostgreSQL for high volume

## Support & Customization

### Customizing Scoring
- Modify weights in Settings page
- Adjust analysis functions in `call_processor.py`
- Add new scoring categories in database schema

### Adding Features
- **Email Reports**: Integrate with SendGrid/Mailgun
- **Advanced Analytics**: Add more visualization charts
- **API Integration**: Connect to Convoso API for lead data
- **Role-based Access**: Add user authentication
- **Real-time Alerts**: Add Slack/Teams notifications

### Integration Points
- **CRM Integration**: Pull lead data via APIs
- **Call Center Software**: Direct integration with Convoso
- **Reporting Tools**: Export to BI platforms
- **Notification Systems**: Slack, email, SMS alerts

## Security Notes

- Database contains sensitive call transcripts
- Implement proper access controls in production
- Consider encryption for stored transcripts
- Audit access logs regularly
- Delete audio files immediately after processing (implemented)

## Next Steps

1. Deploy to Render using GitHub integration
2. Configure FTP or upload mechanism
3. Test with sample call files
4. Customize scoring weights for your needs
5. Set up regular data exports
6. Train team on dashboard usage

## Cost Optimization

- **Render Starter Plan**: ~$7/month for basic usage
- **Storage**: Minimal (only metadata stored)
- **Processing**: CPU-only Whisper keeps costs low
- **Scaling**: Pay only for actual usage

---

For technical support or customization requests, refer to the code documentation or create GitHub issues.
