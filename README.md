# 🧠 Synthora — AI Research Agent (Local Edition)

A fully **local, offline** AI research agent powered by Ollama.
No cloud APIs. No internet required. No API keys. Complete privacy.

Submit a research query against your local knowledge base → get ranked sources, a structured summary, fact-checked claims, and a full downloadable markdown report — all powered by a **local LLM via Ollama**.

---

## ✨ Features (Local)

| Feature | Detail |
|---|---|
| 🏠 **100% Local** | Runs entirely on your machine — no external APIs |
| 🤖 **Any Ollama Model** | llama2, mistral, neural-chat, or any model you pull |
| 📚 **Knowledge Base** | Upload documents directly to the app for searching |
| 🔍 **Local Search** | Keyword-based search of your knowledge base |
| 📊 **LLM Ranking** | Local model ranks sources by relevance |
| 🧠 **Structured Summary** | Key insights, statistics, arguments, risks, opportunities |
| ✅ **Fact-Checking** | Per-claim verification using local LLM |
| 📄 **Report Generation** | Full markdown report with citations |
| ⬇️ **Report Download** | One-click `.md` download |
| 💾 **JSON Persistence** | Flat file storage — no database required |
| 📚 **Research History** | All past reports loadable from history panel |
| 🔐 **Privacy-First** | Everything stays on your machine |

---

## � Documentation

| Document | Purpose |
|----------|---------|
| **[STRUCTURED_OUTPUT.md](STRUCTURED_OUTPUT.md)** | How JSON/structured output works, schemas, troubleshooting |
| **[DEPLOYMENT.md](DEPLOYMENT.md)** | Docker, VPS, Streamlit Cloud, production deployments |
| **[.env.example](.env.example)** | Configuration template |

---

## �🚀 Quick Start

### Prerequisites

1. **Ollama installed** — [ollama.com](https://ollama.com)
2. **Python 3.9+**
3. **Git**

### Installation

```bash
# Clone the repository
git clone https://github.com/your-repo/synthora.git
cd synthora

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy .env.example to .env (optional, defaults are already set)
cp .env.example .env
```

### Run

**Terminal 1: Start Ollama**

```bash
ollama serve
```

**Terminal 2: Pull a model** (first time only)

```bash
ollama pull llama2
# Or try: mistral, neural-chat, orca-mini, etc.
```

**Terminal 3: Start Synthora**

```bash
streamlit run streamlit_app.py
```

Open your browser to: **http://localhost:8501**

---

## 🎯 Usage

1. **Upload Documents** (optional)
   - Use the sidebar to upload documents to your local knowledge base
   - Documents are stored as JSON files in `data/knowledge/`

2. **Research**
   - Enter a research query
   - Click "🚀 Generate Research Report"
   - Wait for the pipeline to complete

3. **Explore Results**
   - **Report tab**: Full markdown report
   - **Summary tab**: Structured insights, statistics, risks, opportunities
   - **Fact Checks tab**: Verified/refuted claims with confidence scores
   - **Sources tab**: Ranked knowledge base documents
   - **Raw JSON tab**: Full pipeline output

4. **Download**
   - Click "⬇️ Download report (.md)" to save as markdown

---

## 📁 File Structure

```
synthora/
├── streamlit_app.py          # Main single-file Streamlit app
├── main.py                   # Optional FastAPI server entry point
├── controller.py             # Research pipeline orchestrator
├── requirements.txt          # Python dependencies
├── .env.example              # Configuration template
├── README.md                 # This file
├── agents/
│   ├── searchAgent.py        # Local knowledge base search
│   ├── summarizeAgent.py     # LLM summarization
│   ├── factCheckAgent.py     # LLM fact-checking
│   └── reportAgent.py        # LLM report generation
├── routes/
│   └── researchRoutes.py     # FastAPI routes (optional)
├── services/
│   ├── ollamaService.py      # Local Ollama client
│   └── openaiService.py      # Compatibility shim
├── utils/
│   └── jsonDB.py             # Local JSON persistence
└── data/                     # Automatically created
    ├── knowledge/            # Your uploaded documents
    ├── projects/             # Research projects (JSON)
    └── reports/              # Generated reports (JSON)
```

---

## ⚙️ Configuration

Edit `.env` to customize:

```bash
# Ollama endpoint (default: localhost:11434)
OLLAMA_BASE_URL=http://localhost:11434

# Model name (run `ollama list` to see available models)
OLLAMA_MODEL=llama2

# Request timeout in seconds
OLLAMA_TIMEOUT=300

# Data directory (where knowledge base and reports are stored)
DATA_DIR=data
```

### Recommended Models

| Model | Size | Speed | Quality | JSON Output | Command |
|-------|------|-------|---------|------------|---------|
| orca-mini | 2.7B | ⚡⚡⚡ | Good | ✅ | `ollama pull orca-mini` |
| mistral | 7B | ⚡⚡ | Excellent | ✅ | `ollama pull mistral` |
| llama2 | 7B | ⚡⚡ | Excellent | ✅ | `ollama pull llama2` |
| neural-chat | 7B | ⚡⚡ | Excellent | ✅ | `ollama pull neural-chat` |
| llama2:13b | 13B | ⚡ | Very Good | ✅ | `ollama pull llama2:13b` |

**All recommended models have been tested for reliable JSON output.**

---

## 📊 Structured Output & JSON Handling

### Overview

Synthora is designed to produce **clean, structured JSON outputs** at every stage of the research pipeline. Each agent:

1. **Specifies output schema** — agents declare expected JSON structure
2. **Uses robust extraction** — multiple fallback strategies for parsing JSON from LLM responses
3. **Validates output** — ensures data matches expected schema before downstream processing
4. **Handles gracefully** — provides sensible fallbacks if JSON parsing fails

### JSON Output at Each Stage

| Stage | Output Format | Location |
|-------|---------------|----------|
| **Search** | Array of ranked sources | `Sources` tab |
| **Summarize** | Structured insights JSON | `Summary` tab |
| **Fact-Check** | Per-claim verification array | `Fact Checks` tab |
| **Report** | Markdown with JSON metadata | `Report` tab & `Raw JSON` |

### JSON Extraction Strategy

Each agent uses `ollamaService.extract_json()` with three fallback strategies:

```python
# Strategy 1: Direct JSON parse
json.loads(response_text)

# Strategy 2: Strip markdown fences
json.loads(response_text.replace("```json", "").replace("```", ""))

# Strategy 3: Regex extraction
match = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", response_text)
json.loads(match.group(1))
```

This ensures that even if the model wraps JSON in explanation text or markdown, the JSON is correctly extracted.

### Enabling JSON Mode

The `ollamaService.generate_json()` method:

1. Embeds the expected schema in the prompt
2. Explicitly instructs the model to return **only valid JSON**
3. Validates the output structure

Example:

```python
schema = {
    "claim": "string",
    "status": "verified|partially_verified|contradicted|unverified",
    "confidence": "float 0-1",
}
result = await ollama_service.generate_json(prompt, schema)
```

### Best Practices for Reliable JSON

1. **Use small, focused prompts** — improves JSON consistency
2. **Provide clear schema examples** — models respond better to concrete specifications
3. **Test with different models** — some models produce cleaner JSON than others
4. **Monitor validation failures** — if you see JSON errors, try a different model
5. **Adjust max_tokens if needed** — truncated responses may result in incomplete JSON

### Troubleshooting JSON Output

**Q: Getting partial or malformed JSON?**
- Increase `OLLAMA_TIMEOUT` in `.env`
- Try a model known for good JSON: `mistral`, `neural-chat`, or `llama2`
- Reduce the amount of input data being processed

**Q: Model not returning JSON at all?**
- Verify the model is working: `ollama generate llama2 "Return: {\"test\": true}"`
- Check system memory — out-of-memory may cause incomplete responses
- Try `ollama pull llama2:13b` for better quality

**Q: JSON parsing errors in logs?**
- The app has fallbacks and will still work, but performance may be degraded
- Monitor the terminal output for `[extract_json]` warnings
- Switch to a more reliable model if errors persist

---

## 🔄 Pipeline

1. **🔍 Search** → Query local knowledge base (keyword match)
2. **📊 Rank** → LLM scores sources by relevance (structured JSON)
3. **🧠 Summarize** → LLM extracts insights, stats, arguments, risks, opportunities (structured JSON)
4. **✅ Fact-Check** → LLM verifies claims against source documents (structured JSON array)
5. **📄 Generate Report** → LLM creates full markdown report with citations
6. **💾 Persist** → Save to local JSON files

All stages can be viewed live in the Streamlit UI, with full JSON output available in the "Raw JSON" tab.

---

## 🛠️ Development

### Run FastAPI server (optional)

```bash
python main.py
# Server runs on http://localhost:8000
# API docs at http://localhost:8000/docs
```

### Run tests

```bash
pytest  # (to be added)
```

### Extend agents

Each agent is a simple class in `agents/`. They all:
- Use `ollama_service` from `services/ollamaService.py`
- Implement async methods
- Return structured dicts

Example:

```python
# agents/myAgent.py
from services.ollamaService import ollama_service

class MyAgent:
    def __init__(self):
        self.llm = ollama_service
    
    async def do_something(self, prompt: str):
        response = await self.llm.generate_content(prompt)
        return response

my_agent = MyAgent()
```

**Enjoy local, private, offline AI research! 🚀**
