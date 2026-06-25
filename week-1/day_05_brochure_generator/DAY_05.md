# Day 5: AI Brochure Generator (Capstone Project)

## 📋 Overview
Day 5 is the Capstone Project of Week 1. We combine everything we learned about local models, JSON formatting, prompt templates, and scraping. We build an automated **AI Brochure Generator** that takes a company website URL, crawls the homepage to find raw links, uses `llama3.2:1b` in JSON mode to filter only relevant corporate pages (About, Careers, Services), scrapes the contents of those pages, and generates a structured marketing brochure.

---

## 🎯 Learning Objectives
* Implement raw website crawling and text scraping using `requests` and `BeautifulSoup`.
* Clean HTML content by removing scripts, style tags, and structural noise.
* Design an LLM-driven web link selector that filters out social media and privacy policies.
* Build a full multi-page document aggregation and content rewriting pipeline.

---

## 📖 Key Concepts Explained (Beginner-Friendly)

### 1. Web Scraping & HTML Cleaning
To feed a website's content to an LLM, we must first download the webpage. We use:
* **`requests`:** Downloads the raw HTML code of the URL.
* **`BeautifulSoup`:** Parses the HTML structure (the DOM tree) so we can extract specific tags.

Raw HTML is filled with junk (like JavaScript code, CSS styling, input boxes, headers, and footers). If we send all of this to the LLM, it will waste thousands of tokens on unreadable code. We clean the HTML by **decomposing** (deleting) tags like `<script>`, `<style>`, `<img>`, and `<nav>`. The remaining text is clean, structured data.

### 2. LLM-Driven Crawler Navigation
When building a scraper for a company, how do you find their "About" or "Careers" page? Writing hardcoded rules is difficult because every website uses different layouts. 

An LLM is excellent at this because it understands context. We extract *all* raw hyperlinks from the homepage and pass them to `llama3.2:1b`. The LLM evaluates the text of each link (e.g. `"/jobs-at-company"` or `"/about-our-mission"`) and returns a structured JSON list of the URLs that are relevant to a company brochure, converting any relative URLs to absolute links.

### 3. Page Ingestion & Brochure Compilation
Once the LLM selects the target URLs, we loop through them, download and clean their text, and merge them into one long string. We then send this compiled text to the copywriter persona to generate a clean Markdown brochure containing Company Overview, Key Services, Careers, and Contact Information.

---

## 💻 Code Walkthrough (Simplified)

### 1. Cleaning and Extracting Web Text
```python
import requests
from bs4 import BeautifulSoup

def get_clean_text(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")
    
    # Delete script and style elements
    for element in soup(["script", "style", "nav", "footer"]):
        element.decompose()
        
    # Get clean text
    return soup.get_text(separator="\n", strip=True)
```

### 2. LLM-Driven Link Selector (JSON Response)
```python
import json
import ollama

system_prompt = """
You are a web crawler selector. Respond with JSON only.
Select links relevant for a company brochure (About, Careers, Products).
Format: {"links": [{"type": "about page", "url": "https://company.com/about"}]}
"""

raw_links = ["/about", "/privacy", "https://twitter.com/company", "/careers"]

response = ollama.chat(
    model='llama3.2:1b',
    messages=[
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': f"Target: https://company.com\nLinks: {', '.join(raw_links)}"}
    ],
    format='json'
)

data = json.loads(response.message.content)
print(data['links'])
```

---

## ❓ Interview Questions & Answers

#### Q1: Why is BeautifulSoup sufficient for simple scrapers, and when would you need tools like Selenium or Playwright?
**Answer:** `BeautifulSoup` and `requests` are static scrapers. They download raw HTML as it exists on the server. If a website is a Single Page Application (SPA) built with React or Angular, the content is generated dynamically in the browser using JavaScript. Static scrapers will only see an empty template. For dynamic websites, you need headless browsers like **Playwright** or **Selenium** to execute the JavaScript before scraping.

#### Q2: What is the benefit of using an LLM to select sub-links instead of standard regular expressions (regex)?
**Answer:** Regular expressions can only look for exact text patterns (e.g. matching URLs containing `"about"` or `"careers"`). However, companies name their pages creatively (e.g., `"/who-we-are"`, `"/our-philosophy"`, or `"/join-the-team"`). An LLM uses semantic understanding to recognize that `"/join-the-team"` is a careers page, which a standard regex rule would miss.

#### Q3: How do you handle rate limits and blockages (like 403 Forbidden) when scraping websites in production?
**Answer:** To avoid being blocked by anti-bot systems:
1. **Rotate User-Agents:** Send headers that look like standard browsers (Chrome, Safari).
2. **Implement delays:** Don't request pages too fast; add a pause (`time.sleep`) between crawls.
3. **Use proxies:** Rotate the IP addresses making the requests.
4. **Respect robots.txt:** Check the website's scraping rules before crawling.

---

## 📝 Resume Bullet Points
* *Architected an automated end-to-end AI Web Scraping and Document Generation pipeline to crawl corporate domains, clean HTML boilerplate, and generate brochures.*
* *Implemented semantic link filtering using local LLMs in JSON mode, achieving automated discovery and categorization of relevant sub-pages.*
* *Utilized BeautifulSoup and requests libraries with user-agent rotation to scrape text content from multiple targets dynamically.*
