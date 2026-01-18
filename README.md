# ViaNova Travel Chatbot

A conversational AI chatbot powered by Natural Language Processing (NLP) and Retrieval-Augmented Generation (RAG) to provide travel recommendations and information about popular destinations in Pakistan.

## 📋 Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [API Documentation](#api-documentation)
- [Data Sources](#data-sources)
- [Evaluation](#evaluation)
- [Contributing](#contributing)

## ✨ Features

- **Conversational Chat Interface**: Interactive web-based chatbot for travel inquiries
- **RAG-Based Responses**: Combines retrieval of relevant travel data with generative AI
- **Multi-Destination Support**: Information about 10 popular Pakistani destinations
- **Real-time Processing**: Fast response generation using optimized NLP models
- **Evaluation Framework**: Built-in evaluation metrics for response quality

## 📁 Project Structure

```
NLP_RAG/
├── app.py                 # Flask web application
├── rag.py                 # Core RAG implementation
├── eval.py                # Evaluation framework
├── new.py                 # Additional utilities
├── updatejson.py          # JSON data update script
├── rag_evaluation.json    # Evaluation results
├── requirements.txt       # Python dependencies
├── README.md              # This file
│
├── data/                  # Travel destination data
│   ├── azadKashmir.csv
│   ├── chitral.csv
│   ├── fairyMedows.csv
│   ├── hunza.csv
│   ├── kumrat.csv
│   ├── murree.csv
│   ├── naran.csv
│   ├── neelum.csv
│   ├── sakardu.csv
│   ├── swat.csv
│   └── scraping/          # Web scraping scripts
│       ├── azadKashmir.py
│       ├── chitral.py
│       ├── fairymedow.py
│       ├── kumrat.py
│       ├── murree.py
│       ├── naran.py
│       ├── neelumValley.py
│       ├── sakardu.py
│       ├── scrapehunza.py
│       └── swat.py
│
├── static/                # Frontend assets
│   ├── css/
│   │   └── style.css
│   └── js/
│       └── chat.js
│
├── templates/             # HTML templates
│   └── index.html
│
└── tour_venv/             # Python virtual environment
```

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Setup Steps

1. **Clone the repository**
   ```bash
   cd "c:\Users\Hp\Desktop\ViaNova Travel Chatbot 2\ViaNova Travel Chatbot\NLP_RAG"
   ```

2. **Create and activate virtual environment**
   ```bash
   # Windows
   python -m venv tour_venv
   tour_venv\Scripts\activate
   
   # macOS/Linux
   python3 -m venv tour_venv
   source tour_venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## ⚙️ Configuration

### Required Dependencies

Key packages used in this project:
- **Flask**: Web framework for the chatbot interface
- **NLP Libraries**: For text processing and understanding
- **RAG Components**: For retrieval and generation tasks
- **Pandas**: Data manipulation and CSV handling

See `requirements.txt` for the complete list of dependencies.

## 💬 Usage

### Running the Chatbot Web Application

```bash
# Ensure virtual environment is activated
tour_venv\Scripts\activate

# Run the Flask application
python app.py
```

The chatbot will be available at `http://localhost:5000`

### Using the RAG System

```python
from rag import RAGSystem

# Initialize RAG system
rag = RAGSystem()

# Get a response to a travel query
response = rag.query("Tell me about Swat Valley")
print(response)
```

### Evaluating the Chatbot

```bash
python eval.py
```

Results are saved to `rag_evaluation.json`

## 📊 API Documentation

### Chat Endpoint

**POST** `/api/chat`

Request body:
```json
{
  "message": "What is there to do in Hunza?"
}
```

Response:
```json
{
  "response": "Hunza Valley is famous for...",
  "sources": ["hunza.csv"]
}
```

## 🗺️ Data Sources

The chatbot has information about these popular Pakistani destinations:

| Destination | CSV File | Status |
|------------|----------|--------|
| Azad Kashmir | azadKashmir.csv | ✓ Active |
| Chitral | chitral.csv | ✓ Active |
| Fairy Meadows | fairyMedows.csv | ✓ Active |
| Hunza Valley | hunza.csv | ✓ Active |
| Kumrat Valley | kumrat.csv | ✓ Active |
| Murree | murree.csv | ✓ Active |
| Naran | naran.csv | ✓ Active |
| Neelum Valley | neelum.csv | ✓ Active |
| Sakardu | sakardu.csv | ✓ Active |
| Swat Valley | swat.csv | ✓ Active |

Data can be updated using the web scrapers in the `data/scraping/` directory.

## 🔄 Data Updates

To update destination data from web sources:

```bash
# Update specific destination
python data/scraping/swat.py

# Update all data
python updatejson.py
```

## 📈 Evaluation

The project includes evaluation metrics to assess chatbot performance:

- **Response Quality**: Measures relevance and accuracy
- **Retrieval Accuracy**: Evaluates document retrieval effectiveness
- **User Satisfaction**: Tracks conversation metrics

View evaluation results:
```bash
cat rag_evaluation.json
```

## 🛠️ Development

### Adding New Destinations

1. Create a web scraper in `data/scraping/`
2. Generate CSV file in `data/`
3. Update `rag.py` to include the new data source
4. Test with `eval.py`

### Modifying the Chat Interface

Edit `templates/index.html` and `static/js/chat.js` for UI changes.

## 📝 License

This project is part of ViaNova Travel Solutions.

## 📧 Support

For issues or questions, please contact the development team.

---

**Last Updated**: January 2026
