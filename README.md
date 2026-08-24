# Feedback Vector Analytics & LLM Processing

## Description
This project implements a lightweight pipeline for storing, retrieving, and analyzing customer feedback using vector similarity search and LLMs. It uses ChromaDB for semantic search with cosine similarity and leverages Gemini API with Pydantic for strictly-typed structured JSON output without relying on high-level LLM frameworks.

## Example Output

```json
{
  "analyses": [
    {
      "sentiment": "Negative",
      "criticality": 5,
      "topics": [
        "Delivery",
        "Quality",
        "Service"
      ],
      "summary": "Delivery was delayed by 2 hours and the food arrived cold."
    },
    {
      "sentiment": "Negative",
      "criticality": 3,
      "topics": [
        "Courier",
        "Order Accuracy",
        "Service"
      ],
      "summary": "Order was mixed up with the wrong soup despite a polite courier."
    }
  ]
}
```

## Project Structure
```text
llm-structured-rag/
├── .env                  # Environment variables (GEMINI_API_KEY)
├── .dockerignore         # Excluded files for Docker build context
├── .gitignore            # Git exclusion rules
├── Dockerfile            # Multi-stage image setup using uv
├── docker-compose.yml    # Orchestration for ChromaDB and App services
├── pyproject.toml        # Project configuration and dependency definitions
├── uv.lock               # Lockfile for deterministic dependencies
├── README.md             # Project documentation
└── app.py                # Main application script
```

## Getting Started

### Prerequisites
* Created `.env` file with your Gemini API key:
  ```env
  GEMINI_API_KEY=your_actual_gemini_api_key_here
  ```

---

### Option 1: Running with Docker Compose (Recommended)

1. **Build and start the services:**
   ```bash
   docker compose up --build
   ```

---

### Option 2: Running Locally with `uv`

1. **Install dependencies:**
   ```bash
   uv sync
   ```

2. **Start ChromaDB server (in a separate terminal):**
   ```bash
   uv run chroma run --path .chroma_data
   ```

3. **Run the script:**
   ```bash
   uv run app.py
   ```

---

## Tech Stack
* **Python 3.12**
* **uv** — Fast Python package installer and resolver
* **Google GenAI SDK** (`google-genai`) — Gemini embeddings (`text-embedding-004`) & structured text generation (`gemini-2.5-flash`)
* **ChromaDB** — Vector database for storing and querying text embeddings
* **Pydantic v2** — Data validation and strict JSON schema enforcement
* **Docker & Docker Compose** — Containerization and service orchestration
