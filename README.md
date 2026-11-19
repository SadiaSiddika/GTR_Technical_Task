# Samsung Phone Query and Review System

A comprehensive Samsung Galaxy phone specifications scraping, relational database ingestion, RAG, and **Sequential Multi-Agent Workflow** system built with **Python**, **BeautifulSoup4**, **SQLAlchemy 2.0**, **PostgreSQL**, **SQLite**, **LangChain**, **FAISS**, and **Hugging Face**.

---

## 🌟 System Architecture

```
┌───────────────────────────────────────┐
│              GSMArena                 │
│      (15 Samsung Galaxy Phones)       │
└──────────────────┬────────────────────┘
                   │
                   ▼ scraper.py (BeautifulSoup4 + Requests)
┌───────────────────────────────────────┐
│       samsung_phones_data.json        │
│      (Normalized Scraped Specs)       │
└──────────────────┬────────────────────┘
                   │
                   ▼ db_manager.py (SQLAlchemy 2.0 ORM)
┌───────────────────────────────────────────────────────────────┐
│                  Dynamic Connection Strategy                  │
│   Primary: PostgreSQL ─── (Fallback) ───► SQLite Database     │
└──────────────────┬────────────────────────────────────────────┘
                   │
                   ▼ rag_pipeline.py (LangChain RAG Pipeline)
┌───────────────────────────────────────────────────────────────┐
│ • Embeddings: Hugging Face (all-MiniLM-L6-v2)                │
│ • Local Vector Database: FAISS (faiss_index/)                 │
└──────────────────┬────────────────────────────────────────────┘
                   │
                   ▼ sequential_workflow.py (Multi-Agent Task Workflow)
┌───────────────────────────────────────────────────────────────┐
│                     USER QUERY                                │
│                         │                                     │
│                         ▼                                     │
│         [STAGE 1: Technical Analyst Agent]                    │
│   • Retrieves raw specs from Relational DB + FAISS            │
│   • Computes hardware metrics, silicon nodes, optical matrix │
│                         │                                     │
│                         ▼ (Raw Technical Data Passed Directly)│
│            [STAGE 2: Reviewer Agent]                          │
│   • Converts engineering data into engaging product review    │
│   • Formulates real-world experience, pros/cons, & verdict    │
│                         │                                     │
│                         ▼                                     │
│                 FINAL COMBINED OUTPUT                         │
└──────────────────┬────────────────────────────────────────────┘
                   │
                   ▼ app.py (Flask Web Dashboard & Studio)
┌───────────────────────────────────────────────────────────────┐
│        Interactive Web Studio on http://localhost:4005        │
│        REST API: /api/workflow/review, /api/rag/query         │
└───────────────────────────────────────────────────────────────┘
```

---

## 🤖 The Multi-Agent Sequential Workflow

The system coordinates two specialized agents operating in a sequential pipeline:

### 🔬 Stage 1: Technical Analyst Agent (`technical_analyst_agent.py`)
- **Objective:** Ingest the user's inquiry and extract exact ground-truth hardware specifications directly from the database (`samsung_phones.db`) and vector index (`faiss_index`).
- **Core Functions:**
  - `fetch_full_technical_profile(model_query)`: Queries relational database for SoC, GPU, display metrics, camera lenses, battery charging speeds, and IP rating.
  - `compare_two_devices(model_a, model_b)`: Computes an exact hardware specification comparison matrix.
  - `filter_by_criteria(...)`: Queries models matching battery capacity, processor family, or camera megapixels.
- **Output:** A structured hardware report containing verified facts, clock frequencies, sensor sizes, and hardware metrics.

### ✍️ Stage 2: Reviewer Agent (`reviewer_agent.py`)
- **Objective:** Receives the raw technical output from the Technical Analyst Agent and writes a detailed, engaging consumer product review or comparison.
- **Narrative Sections Generated:**
  1. 🌟 **The Big Picture & First Impressions:** Industrial design, materials, hand-feel, durability (`IP68`).
  2. 📸 **Camera Experience:** Daylight color accuracy, low-light nightography, optical zoom utility, video stabilization.
  3. ⚡ **Performance & Gaming:** Silicon efficiency, 3D gaming frame stability, thermal management, software support.
  4. 🔋 **Battery & Charging Experience:** Real-world screen-on time, fast charging habits, charging convenience.
  5. 📱 **Display Immersion:** Panel vibrancy, 120Hz smoothness, outdoor sunlight visibility.
  6. ⚖️ **Pros & Cons:** Clear bulleted highlights and compromises.
  7. 🏆 **Verdict & Recommendation:** Clear buying advice (Who Should Buy vs Who Should Skip).

---

## 💻 Running the Sequential Multi-Agent Workflow

### 1. From Python Code:
```python
from sequential_workflow import run_sequential_workflow

# Query a single device review
result = run_sequential_workflow("What are the camera specs of the Samsung Galaxy S23?")

# Print final review & technical breakdown
print("Sources:", result["sources"])
print("Execution Time:", result["execution_time_seconds"], "s")
print(result["final_response"])
```

### 2. From Terminal CLI:
```bash
python sequential_workflow.py
```

### 3. Via REST API (Web Server on Port 4005):
```bash
curl -X POST http://localhost:4005/api/workflow/review \
     -H "Content-Type: application/json" \
     -d '{"query": "Compare Samsung Galaxy S23 and S24 Ultra in detail"}'
```

---

## 🌐 Interactive Web Studio (Port 4005)

Visit **[http://localhost:4005](http://localhost:4005)** in your browser:
- **In-Depth Review (Sequential Mode):** Executes the full multi-agent sequential pipeline (Technical Analyst ➔ Reviewer Agent).
- **Quick Specs (Analyst Mode):** Executes instant technical data retrieval from FAISS and SQLite.
