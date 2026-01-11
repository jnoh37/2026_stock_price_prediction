Note: Converting TXT Files to JSON for NLP Processing
1. Motivation

Raw corporate announcements and news articles are often stored in .txt format.
However, for downstream NLP tasks such as TF-IDF vectorization, event modeling, and machine learning classification, it is more practical to convert unstructured text into a structured JSON format.

JSON allows us to:

Explicitly store metadata (date, company, source)

Separate textual content from labels and identifiers

Easily scale to large document collections

2. Target JSON Schema

Each text file is converted into a single JSON object with the following structure:
{
  "type": "news or conference",
  "date": "YYYY-MM-DD",
  "company_name": "Company Name",
  "title": "Announcement Title",
  "url": "Source URL",
  "content": "Cleaned full text content"
}

Field description:

type: Document category (e.g. news, conference)

date: Publication date

company_name: Name of the company involved

title: Headline or announcement title

url: Original source link

content: Main textual content used for NLP analysis

3. Conversion Principles

When converting from .txt to .json, we follow these principles:

Preserve semantic content
Keep all meaningful paragraphs describing financial performance, strategy, or events.

Remove formatting noise
Bullet points, headings, and excessive whitespace are flattened into continuous text.

Normalize text

Merge line breaks into paragraphs

Remove duplicated titles or source headers

Keep quotes only if semantically relevant

One document = one JSON object
Each .txt file corresponds to exactly one JSON entry.

4. Example Conversion
Input (.txt)
ABN AMRO Bank posts net profit of EUR 617 million in Q3 2025
Date: 2024-11-12

ABN AMRO reported a net profit of EUR 617 million...

Output (.json)
{
  "type": "news",
  "date": "2024-11-12",
  "company_name": "ABN AMRO Bank",
  "title": "ABN AMRO posts net profit of EUR 617 million in Q3 2025",
  "url": "https://www.abnamro.com/en/news/...",
  "content": "ABN AMRO Bank reported a net profit of EUR 617 million in the third quarter of 2025..."
}

5. Usage in TF-IDF Pipeline

After conversion, the content field can be directly used as input for TF-IDF:
texts = [doc["content"] for doc in documents]
X = tfidf_vectorizer.fit_transform(texts)
This separation ensures that only meaningful textual information influences feature extraction.

6. Summary

Converting .txt files into structured .json format is a lightweight but essential preprocessing step.
It improves data consistency, reproducibility, and scalability for NLP-based stock price movement prediction.