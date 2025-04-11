# CLAT FAQ Chatbot 

## Objective

This project implements a simple, rule-based or NLP-powered chatbot designed to assist law aspirants preparing for CLAT and other entrance exams. The chatbot answers user queries using a small, curated knowledge base and basic natural language processing techniques.

This submission addresses Task 2 of the NLTI AI/ML Internship Assignment.

---

## Problem Statement

The chatbot is designed to respond to frequently asked questions related to the CLAT exam, including but not limited to:

- Exam syllabus
- Number of questions per section
- Exam pattern and marking scheme
- Previous year cut-offs
- Key dates and participating institutions

The chatbot accepts free-text queries and returns the most relevant response from its internal knowledge base using text similarity.

---

## Approach

### 1. Knowledge Base

A static, in-memory knowledge base containing frequently asked CLAT-related questions and answers is defined directly in the script. This design choice eliminates external dependencies such as CSV or database files and keeps the system lightweight.

### 2. NLP Methodology

- All questions from the knowledge base are vectorized using **TF-IDF** (Term Frequency–Inverse Document Frequency).
- When a user inputs a query, it is also vectorized and compared against all known questions using **cosine similarity**.
- The chatbot returns the answer corresponding to the most similar known question.
- If the similarity score is below a predefined threshold (e.g., 0.2), the chatbot responds with a fallback message requesting clarification or rephrasing.

---

## Technologies Used

- Python
- scikit-learn (TF-IDF vectorization, cosine similarity)
- Google Colab (for interactive development and testing)

---

## Features

- Lightweight, no-file, self-contained implementation
- TF-IDF based vectorization for semantic matching
- Graceful fallback mechanism for unmatched queries
- Confidence score display for transparency
- Easily extendable knowledge base

---

## How to Run (Google Colab)

1. Open [Google Colab](https://colab.research.google.com/)
2. Paste the code from `chatbot.py` or your `.ipynb` file
3. Run all cells
4. Enter queries into the interactive loop to receive chatbot responses


