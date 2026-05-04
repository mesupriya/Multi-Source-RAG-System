# 🕉️ Mahabharata Multi-Source RAG System

An advanced Retrieval-Augmented Generation (RAG) system designed to provide accurate, cited answers to questions about the Mahabharata by synthesizing information from the complete 18 Parvas dataset and live scholarly web sources.

## 🚀 Overview

This project implements a sophisticated RAG pipeline that combines semantic search, keyword retrieval, and live web exploration to navigate the complexities of the Mahabharata. It features multi-hop reasoning capabilities and provides detailed source attribution for every fact presented.

### Key Features
- **Hybrid Retrieval**: Combines BM25 (keyword-based) and FAISS (vector-based semantic search) for superior context fetching.
- **Multi-Source Synthesis**: Integrates internal dataset (18 Parvas) with live web results via the Tavily API.
- **Multi-Hop Reasoning**: Capable of answering complex queries that require connecting information across different sections of the text.
- **Transparent Citations**: Every response includes precise citations (e.g., `[Dataset-X]` or `[Web-X]`) with source previews.
- **Performance Analytics**: Real-time tracking of retrieval latency, generation time, token usage, and estimated API costs.
- **Interactive UI**: A sleek user interface built with Gradio for seamless exploration.

## 🛠️ Technology Stack

- **Large Language Models**: 
  - LLaMA 3.3-70B (Answer Generation via Groq)
  - LLaMA 3.1-8B (Fact Extraction & Query Classification)
- **Embeddings**: Sentence-Transformers (`all-MiniLM-L6-v2`)
- **Vector Database**: FAISS (Facebook AI Similarity Search)
- **Web Search**: Tavily Search API
- **Frontend**: Gradio
- **Programming Language**: Python (Jupyter Notebook)

## 📁 Project Structure

- `FinalProjectCode-DTSC5525-Supriya Veerla.ipynb`: The core implementation containing the RAG pipeline, evaluation metrics, and Gradio interface.
- `Data.json`: The processed Mahabharata dataset containing text from all 18 Parvas.
- `FinalReport-DTSC5525-Supriya Veerla.pdf`: Detailed project documentation, methodology, and experimental results.
- `FinalPresentation-DTSC5525-Supriya Veerla.pptx`: Presentation slides summarizing the project.
- `FinalPresentation-DTSC5525-Supriya Veerla.mp4`: Video version of the presentation.
- `FinalDemoVideo-DTSC5525-Supriya Veerla.mp4`: A walkthrough video demonstrating the system in action.

## ⚙️ Setup & Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/[Your-Username]/Mahabharata-RAG-System.git
   cd Mahabharata-RAG-System
   ```

2. **Install Dependencies**:
   ```bash
   pip install langchain langchain-community sentence-transformers faiss-cpu rank_bm25 groq tavily-python gradio matplotlib seaborn
   ```

3. **API Configuration**:
   The system requires API keys for Groq and Tavily. You can set them as environment variables or update them directly in the notebook:
   ```python
   os.environ["GROQ_API_KEY"] = "your_groq_api_key"
   os.environ["TAVILY_API_KEY"] = "your_tavily_api_key"
   ```

4. **Run the System**:
   Open the Jupyter Notebook and execute all cells. The Gradio interface will launch at the end, providing a local or public URL to access the chatbot.

## 📊 Evaluation Results

The system was rigorously tested across various retrieval methods. The **Hybrid Retriever** consistently outperformed standalone BM25 or FAISS in terms of F1 score and contextual relevance, especially for nuanced philosophical queries.

| Retriever | Token F1 | Avg. Latency (s) |
|-----------|----------|------------------|
| BM25      | 0.285    | 0.05             |
| FAISS     | 0.312    | 0.12             |
| **Hybrid**| **0.420**| **0.15**         |

## 🤝 Acknowledgments

This project was developed as part of the **DTSC 5525: Generative AI** course. Special thanks to the instructors and the open-source community for providing the tools and models used in this implementation.

---
*Created by Supriya Veerla*
