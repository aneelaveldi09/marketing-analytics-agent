<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=7c3aed&height=200&section=header&text=Marketing%20Analytics%20AI&fontSize=42&fontColor=ffffff&fontAlignY=38&desc=My%20first%20RAG%20agent%20%E2%80%94%20real%20research%2C%20real%20answers&descAlignY=58&descSize=16&descColor=a78bfa" />

<br/>

<img src="https://img.shields.io/badge/Grok--3-xAI-7c3aed?style=for-the-badge&logo=x&logoColor=white" />
<img src="https://img.shields.io/badge/FAISS-Vector%20Search-4f46e5?style=for-the-badge&logo=meta&logoColor=white" />
<img src="https://img.shields.io/badge/Streamlit-UI-ff4b4b?style=for-the-badge&logo=streamlit&logoColor=white" />
<img src="https://img.shields.io/badge/arXiv-Research%20Papers-b31b1b?style=for-the-badge&logo=arxiv&logoColor=white" />
<img src="https://img.shields.io/badge/sentence--transformers-Embeddings-f59e0b?style=for-the-badge&logo=huggingface&logoColor=white" />

<br/><br/>

**Ask marketing analytics questions. Get answers backed by real research papers, not hallucinations.**

<br/>

</div>

---

## What is this?

This is my first AI agent. I built it because I kept getting frustrated with chatbots that just make things up when you ask specific analytics questions. This one actually retrieves real context before answering.

The knowledge base has **467 chunks** from Wikipedia and **60 arXiv research papers** published between 2020-2024 on ML for marketing. When you ask a question, it does semantic search across all of that, grabs the most relevant chunks, and feeds them to Grok-3 as context. No hallucinations on stuff that's in the knowledge base.

---

## How it works

```
  Your question
       |
       v
  ┌─────────────────────────────┐
  │  sentence-transformers      │  embed your question into a vector
  │  all-MiniLM-L6-v2           │
  └─────────────────────────────┘
       |
       v
  ┌─────────────────────────────┐
  │  FAISS IndexFlatIP          │  cosine similarity search
  │  467 chunks indexed         │  returns top 6 most relevant chunks
  └─────────────────────────────┘
       |
       v
  ┌─────────────────────────────┐
  │  Grok-3 via xAI API         │  generates a streaming answer
  │  context: retrieved chunks  │  grounded in your knowledge base
  └─────────────────────────────┘
       |
       v
  Answer + sources + follow-up questions
```

---

## Knowledge base

| Source | Count | Topics |
|---|---|---|
| Wikipedia | 404 chunks | A/B testing, attribution, churn, CLV, conversion rate optimization, cohort analysis, digital marketing, email marketing, gradient boosting, influencer marketing, k-means clustering, marketing mix modeling, NLP, NPS, predictive analytics, purchase funnel, random forest, recommendation systems, regression analysis, ROMI, segmentation, sentiment analysis, social media analytics |
| arXiv papers | 63 chunks | ML churn prediction, deep learning attribution, NLP for customer reviews, RL marketing optimization, transformer text classification, GNN recommendations, causal inference, LLMs for business analytics, CLV neural networks, multi-touch attribution |

---

## Tech stack

| Layer | Tool |
|---|---|
| LLM | [xAI Grok-3](https://x.ai) (OpenAI-compatible) |
| Embeddings | `sentence-transformers` / `all-MiniLM-L6-v2` |
| Vector store | FAISS `IndexFlatIP` (cosine similarity on normalized vectors) |
| Data: Wikipedia | Wikipedia REST API with `User-Agent` header |
| Data: papers | arXiv Atom XML API (no key needed) |
| UI | Streamlit with custom glassmorphism CSS |
| Config | `python-dotenv` |

---

## Setup

**1. Clone**
```bash
git clone https://github.com/aneelaveldi09/marketing-analytics-agent.git
cd marketing-analytics-agent
```

**2. Install**
```bash
pip install -r requirements.txt
```

**3. Add your API key**
```bash
cp .env.example .env
```
Open `.env` and paste your xAI key. Get one at [x.ai](https://x.ai).
```
XAI_API_KEY=your_key_here
```

**4. Build the knowledge base**
```bash
python pipeline.py
```
Fetches Wikipedia + arXiv papers, embeds everything, saves a FAISS index locally. Takes a few minutes.

**5. Run**
```bash
streamlit run app.py
```

---

## Project structure

```
marketing-analytics-agent/
├── app.py                  # Streamlit UI (glassmorphism, streaming, source cards)
├── pipeline.py             # One-shot: fetch data, embed, build index
├── requirements.txt
├── .env.example
└── src/
    ├── config.py           # All settings: topics, queries, model names
    ├── data_loader.py      # Wikipedia + arXiv fetchers, chunker
    ├── embeddings.py       # sentence-transformers wrapper
    ├── vector_store.py     # FAISS build and search
    └── agent.py            # RAG pipeline: retrieve, stream, follow-ups
```

---

## Example questions to try

- How does multi-touch attribution work and which model is most accurate?
- What ML models best predict customer churn?
- How do you actually measure influencer marketing ROI?
- How are LLMs being used in marketing analytics today?
- What's the difference between customer segmentation methods?
- How do you design an A/B test that is statistically valid?

---

<div align="center">

Built by **Aneela Veldi**

M.S. Business Analytics, DePaul University (2026)

[GitHub](https://github.com/aneelaveldi09) · [LinkedIn](https://www.linkedin.com/in/aneelaveldi)

<img src="https://capsule-render.vercel.app/api?type=waving&color=7c3aed&height=100&section=footer" />

</div>
