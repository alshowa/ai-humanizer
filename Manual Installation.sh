# Clone the repository
git clone <repository-url>
cd ai-humanizer

# Run the startup script
chmod +x start.sh
./start.sh

# Or manually:
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('punkt'); nltk.download('wordnet')"
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload