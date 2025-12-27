# Review: Roman Empire History RAG Assistant

## Overall Assessment: **Good Foundation, Needs Significant Expansion**

This PRD demonstrates a solid understanding of RAG basics and chooses a workable domain. However, it has **critical gaps** in multimodal implementation details, data acquisition strategy, and evaluation methodology. The scope also raises questions about true novelty and depth.

---

## CRITICAL ISSUES

### **1. Wikipedia Data Source - Major Concerns**

**Your PRD states:**
> "Data Sources: Wikipedia articles related to the Roman Empire"
> "Around 600 text documents and around 1,800 associated images"

#### **Problem 1: Data Acquisition Strategy is Vague**

**Critical questions:**
- How will you collect these 600 articles?
- Which Wikipedia categories will you scrape?
- How will you handle Wikipedia's structure (sections, infoboxes, references)?
- How will you download and attribute images?

#### **Problem 2: Wikipedia Scraping Legal/Ethical Considerations**

Wikipedia content is **CC BY-SA licensed**, which requires:
- Attribution to Wikipedia
- Share-alike licensing (your derivative work must also be open)
- Proper image licensing (each image has its own license!)

**Many Wikipedia images are NOT freely reusable:**
- Some are "fair use" (can't be redistributed)
- Some require specific attribution
- Some are copyrighted

**You MUST:**
```markdown
### Wikipedia Image Licensing Strategy

**Approach:**
1. Filter for images with permissive licenses:
   - Public Domain (CC0)
   - CC BY (Creative Commons Attribution)
   - CC BY-SA (Creative Commons Attribution-ShareAlike)
2. Skip images marked "Fair Use" (can't legally use in your project)
3. Store attribution metadata for each image:
   {
     'image_url': 'https://upload.wikimedia.org/...',
     'license': 'CC BY-SA 3.0',
     'author': 'User:HistorianXYZ',
     'source': 'https://commons.wikimedia.org/...',
     'attribution_required': True
   }
   
4. Display attribution in UI (legal requirement)

**Estimated Impact:**
- May lose 30-40% of images due to licensing restrictions
- Revised estimate: ~1,200 usable images from 1,800 total
```

#### **Problem 3: Wikipedia Quality Varies**

Wikipedia articles have:
- Good: Well-sourced, comprehensive articles (e.g., "Roman Empire", "Augustus")
- Mediocre: Stub articles with minimal content
- Bad: Poorly written, lacking citations, biased

**You need a quality filter:**

```markdown
### Wikipedia Article Selection Criteria

**Quality Filters:**
1. Article length > 5,000 characters (avoid stubs)
2. Has infobox (indicates structured content)
3. Has at least 5 citations (indicates sourcing)
4. Not flagged with {{citation needed}} or {{disputed}} tags
5. Exists in English Wikipedia (not just translations)

**Article Categories to Scrape:**
- Category:Roman Empire
- Category:Roman emperors
- Category:Roman Republic
- Category:Ancient Rome
- Category:Roman military
- Category:Roman law
- Category:Roman architecture
- [Be specific about which categories]

**Exclusion List:**
- Modern fiction (e.g., "Rome (TV series)")
- Video games (e.g., "Total War: Rome")
- Modern places named after Rome
```

---

### **2. "Context-Aware Conversations" - Architectural Ambiguity**

**Your PRD says:**
> "Context-aware conversations across multiple messages"

**This is a MAJOR feature that needs detailed specification.**

#### **Question 1: How will you maintain context?**

**Option A: Naive Approach (Not Recommended)**
```
# Store entire chat history, send to LLM each time
chat_history = [
    {"user": "Who was Augustus?", "assistant": "Augustus was..."},
    {"user": "When did he rule?", "assistant": "He ruled from..."}
]

# New query: "What were his achievements?"
# Problem: "his" refers to Augustus, but retrieval doesn't know that!
query = "What were his achievements?"
results = vector_store.search(query)  # Will return generic "achievements" results
```

**Option B: Query Rewriting (Recommended)**
```
# Rewrite query with chat context
chat_context = "Previous topic: Augustus (Roman Emperor)"
original_query = "What were his achievements?"

# Use LLM to rewrite
rewritten_query = llm.rewrite_query(
    query=original_query,
    context=chat_context
)
# Output: "What were Augustus's achievements as Roman Emperor?"

# Now retrieval works properly
results = vector_store.search(rewritten_query)
```

**Option C: Conversational RAG with Memory**
```
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory()
memory.save_context(
    {"input": "Who was Augustus?"}, 
    {"output": "Augustus was the first Roman Emperor..."}
)

# Next query automatically includes context
chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory
)

response = chain({"question": "What were his achievements?"})
# Chain automatically resolves "his" → "Augustus"
```
---

## **Add to PRD:**

### Conversational Context Management

**Architecture:**
1. **Chat Session Storage:**
   - Each chat has unique session_id
   - Store messages: [{role, content, timestamp, sources}]
   - Store metadata: {topic, key_entities, created_at}

2. **Query Rewriting:**
   ```
   def rewrite_query_with_context(current_query, chat_history):
       """
       Rewrite query to be self-contained using chat history.
       """
       # Extract last 3 messages for context
       recent_context = chat_history[-3:]
       
       # Prompt LLM to rewrite
       prompt = f"""
       Given the conversation history and current query, 
       rewrite the query to be self-contained.
       
       Chat history:
       {recent_context}
       
       Current query: {current_query}
       
       Rewritten query:
       """
       
       rewritten = llm.generate(prompt)
       return rewritten
   ```

3. **Entity Tracking:**
   - Track mentioned entities (emperors, battles, places)
   - Resolve pronouns: "he" → "Augustus", "it" → "Roman Empire"

4. **Context Window Management:**
   - Include last N messages (N=5 recommended)
   - Summarize older messages if chat is long
   - Avoid context overflow (stay within token limits)

**Example Flow:**
```
User: "Who was Augustus?"
→ Retrieval: "Augustus Roman Emperor"
→ Response: "Augustus was the first Roman Emperor..."
→ Store: {entity: "Augustus", topic: "Roman Emperors"}

User: "When did he rule?"
→ Rewrite: "When did Augustus rule?"
→ Retrieval: "Augustus reign dates"
→ Response: "Augustus ruled from 27 BC to 14 AD..."

User: "What about Julius Caesar?"
→ New topic detected
→ Update: {entity: "Julius Caesar"}
→ Retrieval: "Julius Caesar"
```

**Challenges:**
- Ambiguous references ("What about Trajan?" after discussing multiple emperors)
- Topic switches (moving from politics to architecture)
- Long conversations (context grows too large)

**Mitigation:**
- Confidence scoring for entity resolution
- Explicit clarification: "Did you mean Trajan or Hadrian?"
- Context pruning: summarize distant messages

---

### **3. Multimodal Implementation Details Missing**

**Your PRD mentions:**
> "Image embeddings: OpenCLIP"

**But provides NO details on:**
- How images will be embedded
- How text-to-image retrieval will work
- Whether images will be retrieved independently or via text

#### **Critical Gap: Image Retrieval Strategy**

## **Add this section to PRD:**

### Multimodal Retrieval Architecture

**Dual-Collection Approach:**

**Collection 1: Text Chunks**
- Source: Wikipedia article sections
- Embedding: sentence-transformers/all-MiniLM-L6-v2
- Metadata: {article_title, section, url, related_images}

**Collection 2: Image Embeddings**
- Source: Wikipedia images (Commons)
- Embedding: OpenCLIP (ViT-B/32)
- Metadata: {caption, article, license, attribution}

**Retrieval Strategies:**

**Strategy A: Text-First with Image Linking (Simpler)**
```
def retrieve_with_images(query):
    # 1. Retrieve text chunks
    text_results = qdrant.search(
        collection_name="roman_empire_text",
        query_vector=embed_text(query),
        limit=5
    )
    
    # 2. Get linked images
    images = []
    for result in text_results:
        related_image_ids = result.payload.get('related_images', [])
        images.extend(load_images(related_image_ids))
    
    return {
        'text_chunks': text_results,
        'images': images[:3]  # Top 3 images
    }
```

**Strategy B: Parallel Text + Image Search (True Multimodal)**
```
def retrieve_multimodal(query):
    # 1. Embed query with OpenCLIP text encoder
    text_embedding = clip_text_encoder(query)
    
    # 2. Search text collection
    text_results = qdrant.search(
        collection_name="roman_empire_text",
        query_vector=sentence_transformer_embed(query),
        limit=5
    )
    
    # 3. Search image collection with CLIP
    image_results = qdrant.search(
        collection_name="roman_empire_images",
        query_vector=text_embedding,  # Same embedding space as images
        limit=3
    )
    
    # 4. Merge results
    return {
        'text_chunks': text_results,
        'images': image_results
    }
```

**Image-Centric Query Handling:**

For queries like "What did the borders look like...":
1. Detect image intent (keywords: "look like", "show", "map")
2. Boost image retrieval weight
3. Prioritize map-type images (filter by metadata)

**Implementation:**
```
IMAGE_QUERY_KEYWORDS = ['look like', 'show', 'map', 'depict', 'image', 'picture']

def detect_image_query(query):
    return any(keyword in query.lower() for keyword in IMAGE_QUERY_KEYWORDS)

def retrieve(query):
    is_image_query = detect_image_query(query)
    
    if is_image_query:
        # Prioritize images
        image_results = search_images(query, limit=5)
        text_results = search_text(query, limit=3)
    else:
        # Prioritize text
        text_results = search_text(query, limit=5)
        image_results = search_images(query, limit=2)
    
    return merge_results(text_results, image_results)
```

**Caption Extraction:**
Wikipedia images often have good captions in the HTML:
```
<img src="augustus.jpg" alt="Statue of Augustus">
<figcaption>Statue of Augustus as a younger Octavian, c. 30 BC</figcaption>
```

Extract and store these captions for embedding.

---

### **4. Example Queries Lack Diversity**

**Current queries:** All are text-focused factual questions.

**Missing:**
- Truly image-centric queries
- Comparison queries
- Timeline queries
- Complex multi-part queries

**Recommended additions:**

```
### Example Queries (Revised - 15 Total)

**Factual (Text-Heavy) - 5 queries:**
1. What are the main periods in the history of the Roman Empire?
2. Who was Trajan?
3. Why did the Roman Empire collapse?
4. How did the form of government differ between the Republican and Imperial periods?
5. Why was the capital moved from Rome to Constantinople?

**Image-Centric - 5 queries:**
6. What did the borders look like during the empire's greatest territorial extent?
7. What did the borders of the Western and Eastern Roman Empires look like?
8. Show me a map of the Roman road network
9. What did Roman soldiers' armor look like?
10. Show me examples of Roman architecture

**Conversational/Follow-up - 3 queries:**
11. "Who was Augustus?" → "When did he rule?" → "What were his major achievements?"
12. "Tell me about the Punic Wars" → "Who was Hannibal?" → "Show me a map of his route"
13. "What was the Praetorian Guard?" → "Who commanded them?"

**Complex/Analytical - 2 queries:**
14. Compare the military strategies of Julius Caesar and Pompey
15. How did Roman coinage change from the Republic to the Empire?
```

---

### **5. Evaluation Methodology Too Vague**

**Current:**
> "Retrieval relevance: top-k results contain the correct source for at least 70% of test queries"

**Problems:**
- No mention of evaluation dataset size
- No details on gold standard creation
- No image retrieval metrics
- No conversational evaluation

**Add comprehensive evaluation section:**

```
### Evaluation Methodology (Detailed)

**Evaluation Dataset:**
- **Size:** 50 queries (30 single-turn, 20 conversational)
- **Distribution:**
  - 20 factual queries (text-heavy)
  - 15 image-centric queries
  - 15 conversational chains (3 chains × 5 turns)

**Gold Standard Creation:**
1. For each query, manually identify:
   - Expected Wikipedia articles (1-3 per query)
   - Expected text sections (paragraph-level)
   - Expected images (if applicable)
2. Two annotators independently label, resolve conflicts
3. Store in evaluation dataset: `eval_queries.json`

**Retrieval Metrics:**

**Text Retrieval (30 queries):**
- **Recall@5:** Is at least one gold-standard chunk in top-5?
  - Target: ≥75%
- **MRR (Mean Reciprocal Rank):** Average position of first relevant result
  - Target: ≥0.6
- **Precision@5:** What proportion of top-5 are relevant?
  - Target: ≥60%

**Image Retrieval (15 image-centric queries):**
- **Image Hit Rate@3:** Is expected image in top-3?
  - Target: ≥70%
- **Image Type Accuracy:** For map queries, are retrieved images actually maps?
  - Target: ≥80%

**Conversational Metrics (20 queries in 5 chains):**
- **Context Resolution Rate:** Are follow-up queries correctly resolved?
  - Example: "When did he rule?" correctly identifies "he" = Augustus
  - Target: ≥80%
- **Topic Coherence:** Do retrieved sources stay on topic across conversation?
  - Target: ≥75%

**Generation Metrics:**

**Faithfulness (Critical):**
- Manual binary evaluation: Is answer grounded in sources?
- Check for hallucinations (invented facts, dates, names)
- Target: 100% faithful (zero tolerance for historical inaccuracies)

**Citation Quality:**
- Are Wikipedia sources properly cited?
- Format: "[Source: Wikipedia - Julius Caesar, Section: Early Career]"
- Target: 95%+ citations included

**Conversational Coherence:**
- Does assistant maintain context across turns?
- Does it gracefully handle topic switches?
- Manual 3-point scale: Good (2), Adequate (1), Poor (0)
- Target: Average ≥1.5

**Latency:**
- Measure end-to-end: query → retrieval → generation → response
- Single-turn queries: Target <5s
- Conversational queries (with context): Target <7s
- Breakdown: Retrieval (<2s), Generation (<4s)

**Baseline Comparisons:**
1. **vs. Wikipedia Search:** Does RAG beat native Wikipedia search?
2. **vs. ChatGPT (no RAG):** Show hallucination reduction
3. **vs. Text-Only RAG:** Demonstrate value of images

**Error Analysis Categories:**
- Query ambiguity (unclear user intent)
- Missing information in corpus (gap in Wikipedia coverage)
- Poor retrieval (wrong chunks/images retrieved)
- Generation hallucination (LLM invents facts despite correct context)
- Context resolution failure (conversational queries)
- License issues (unavailable images)

**Reporting:**
Create evaluation report with:
- Aggregate metrics table
- Per-query breakdown (identify hardest queries)
- Failure examples with root cause analysis
- Suggestions for improvement
```

---

### **6. Data Pipeline Missing**

**How will you actually acquire this data?**

**Add detailed ingestion section:**

### Data Ingestion Pipeline

**Phase 1: Wikipedia Article Collection**

**Step 1: Identify Target Articles**
```
import wikipediaapi

wiki = wikipediaapi.Wikipedia('en')

# Define seed categories
seed_categories = [
    'Roman_Empire',
    'Roman_emperors',
    'Roman_Republic',
    'Roman_military',
    'Roman_law'
]

def collect_articles_from_category(category_name, max_depth=2):
    """
    Recursively collect all articles in a category.
    """
    category = wiki.page(f"Category:{category_name}")
    articles = []
    
    for page in category.categorymembers.values():
        if page.namespace == 0:  # Article namespace
            articles.append({
                'title': page.title,
                'url': page.fullurl,
                'summary': page.summary
            })
        elif page.namespace == 14 and max_depth > 0:  # Category
            # Recurse into subcategory
            articles.extend(collect_articles_from_category(
                page.title.replace('Category:', ''),
                max_depth - 1
            ))
    
    return articles

# Collect all Roman Empire articles
roman_articles = []
for category in seed_categories:
    roman_articles.extend(collect_articles_from_category(category))

# Deduplicate
roman_articles = list({a['title']: a for a in roman_articles}.values())
print(f"Collected {len(roman_articles)} unique articles")
```

**Step 2: Download Article Content**
```
def download_article(title):
    """
    Download full article content with sections.
    """
    page = wiki.page(title)
    
    if not page.exists():
        return None
    
    return {
        'title': page.title,
        'url': page.fullurl,
        'summary': page.summary,
        'sections': extract_sections(page.text),
        'categories': page.categories.keys(),
        'images': extract_image_urls(page),
        'references': extract_references(page),
        'last_modified': page.touched  # Timestamp
    }
```

**Step 3: Filter for Quality**
```
def is_quality_article(article):
    """
    Filter out stub/poor quality articles.
    """
    # Check article length
    if len(article['text']) < 5000:
        return False
    
    # Check for citations
    if len(article.get('references', [])) < 5:
        return False
    
    # Check for infobox (indicates structured content)
    if '{{Infobox' not in article['raw_text']:
        return False
    
    # Check for quality issues
    problematic_tags = [
        '{{citation needed}}',
        '{{disputed}}',
        '{{cleanup}}',
        '{{stub}}'
    ]
    if any(tag in article['raw_text'] for tag in problematic_tags):
        return False
    
    return True

# Filter
quality_articles = [a for a in roman_articles if is_quality_article(a)]
print(f"{len(quality_articles)} articles passed quality filter")
```

**Phase 2: Image Collection with Licensing**

```
import requests
from bs4 import BeautifulSoup

def get_image_license(image_url):
    """
    Fetch license information from Wikimedia Commons.
    """
    # Parse Commons page
    commons_url = image_url.replace(
        'upload.wikimedia.org',
        'commons.wikimedia.org/wiki/File:'
    )
    
    response = requests.get(commons_url)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Extract license information
    license_section = soup.find('table', {'class': 'licensetpl'})
    
    if not license_section:
        return None
    
    license_text = license_section.get_text()
    
    # Check if permissive license
    permissive_licenses = ['CC0', 'CC BY', 'CC BY-SA', 'Public Domain']
    is_permissive = any(lic in license_text for lic in permissive_licenses)
    
    return {
        'url': image_url,
        'license': extract_license_name(license_text),
        'author': extract_author(license_section),
        'attribution_required': 'CC BY' in license_text,
        'is_permissive': is_permissive
    }

def download_images_for_article(article):
    """
    Download images with proper licensing.
    """
    images = []
    
    for img_url in article['images']:
        license_info = get_image_license(img_url)
        
        # Only keep permissively licensed images
        if license_info and license_info['is_permissive']:
            # Download image
            img_data = requests.get(img_url).content
            
            # Save locally
            filename = f"images/{sanitize_filename(img_url)}"
            with open(filename, 'wb') as f:
                f.write(img_data)
            
            images.append({
                'filename': filename,
                'url': img_url,
                'license': license_info,
                'caption': extract_caption(article, img_url),
                'article_title': article['title']
            })
    
    return images
```

**Phase 3: Text Chunking**

```
def chunk_wikipedia_article(article):
    """
    Chunk article by sections, preserving structure.
    """
    chunks = []
    
    for section in article['sections']:
        section_text = section['text']
        
        # If section is long, split further
        if len(section_text) > 1500:
            # Split by paragraphs
            paragraphs = section_text.split('\n\n')
            
            current_chunk = ""
            for para in paragraphs:
                if len(current_chunk) + len(para) < 1000:
                    current_chunk += para + "\n\n"
                else:
                    # Save chunk
                    chunks.append({
                        'text': current_chunk,
                        'article': article['title'],
                        'section': section['heading'],
                        'url': article['url'],
                        'related_images': find_images_in_section(section)
                    })
                    current_chunk = para + "\n\n"
            
            # Save last chunk
            if current_chunk:
                chunks.append({...})
        else:
            # Keep section as single chunk
            chunks.append({
                'text': section_text,
                'article': article['title'],
                'section': section['heading'],
                'url': article['url'],
                'related_images': find_images_in_section(section)
            })
    
    return chunks
```

**Expected Output:**
- `data/articles/` - Raw Wikipedia articles (JSON)
- `data/chunks/` - Processed text chunks (JSON)
- `data/images/` - Downloaded images (JPG/PNG)
- `data/image_metadata/` - Image licenses and attributions (JSON)
- `data/index/` - Qdrant vector store

---

## Missing Sections - Must Add

Your PRD needs these sections:

1. **Data Acquisition Pipeline** (CRITICAL)
   - Wikipedia scraping strategy
   - Quality filtering
   - License compliance

2. **Image Licensing Strategy** (CRITICAL - LEGAL REQUIREMENT)
   - How to identify permissive licenses
   - Attribution requirements
   - Handling of fair-use images

3. **Multimodal Retrieval Architecture** (CRITICAL)
   - Text vs. image retrieval strategies
   - CLIP implementation details
   - Cross-modal search

4. **Conversational Context Management** (CRITICAL)
   - Query rewriting approach
   - Entity tracking
   - Context window management

5. **Detailed Evaluation Methodology**
   - Evaluation dataset size and composition
   - Conversational query evaluation
   - Baseline comparisons

6. **UI Wireframe**
   - Chat interface mockup
   - Image display strategy
   - Chat history navigation

7. **Implementation Timeline**
   - Week-by-week breakdown
   - Milestones and deliverables

8. **Risk Analysis**
   - Wikipedia API rate limits
   - Image licensing violations
   - LLaMA 3 8B context limitations

---

## Strengths to Build On

### **What's Good:**

1. **Established domain:** Roman Empire is well-documented
2. **Clear target users:** Students and self-learners
3. **Reasonable scope:** Not trying to cover all of history
4. **Good technical stack:**
   - LLaMA 3 8B (capable, runs locally)
   - all-MiniLM-L6-v2 (fast, proven)
   - OpenCLIP (standard for multimodal)
   - Qdrant (production-grade vector DB)
5. **Conversational feature:** Adds value beyond simple Q&A

---

## Concerns & Recommendations

### **Concern 1: Dataset Size May Be Inflated**

**Your estimate: 600 articles, 1,800 images**

After quality filtering and license restrictions, expect:
- **Articles:** ~300-400 high-quality articles
- **Usable images:** ~800-1,000 (after removing fair-use)

**This is still plenty!** Just set realistic expectations.

---

### **Concern 2: Is This Novel Enough?**

Roman Empire history assistants exist (ChatGPT, Perplexity, etc.). 

**What makes yours unique?**
- Grounded in Wikipedia (no hallucinations)
- Source citations
- Conversational memory
- Image integration

**Add to PRD:**
```
### Differentiation from Existing Solutions

**vs. ChatGPT/Claude:**
- ❌ ChatGPT: Hallucinates historical facts, no sources
- ✅ Our system: Every fact traced to Wikipedia source

**vs. Wikipedia Search:**
- ❌ Wikipedia: Must know exact article title
- ✅ Our system: Natural language queries, cross-article synthesis

**vs. Perplexity:**
- ❌ Perplexity: Live web search (slower, less focused)
- ✅ Our system: Fast local RAG, Roman Empire-specific

**Unique Value:**
- Conversational learning (follow-up questions)
- Image-grounded explanations (maps, portraits, artifacts)
- Educational focus (clear citations for students)
```

---

### **Concern 3: "Ability to create new chats and continue previous conversations"**

**This requires database + backend architecture.**

**Add infrastructure section:**

```
**Backend Architecture:**

┌─────────────────────────────────────┐
│         Streamlit/Gradio UI         │
├─────────────────────────────────────┤
│         FastAPI Backend             │
│  ┌───────────────────────────────┐  │
│  │  Chat Management Service      │  │
│  │  - Create/list chats          │  │
│  │  - Store messages             │  │
│  └───────────────────────────────┘  │
│  ┌───────────────────────────────┐  │
│  │  RAG Pipeline                 │  │
│  │  - Query rewriting            │  │
│  │  - Retrieval (Qdrant)         │  │
│  │  - Generation (LLaMA 3)       │  │
│  └───────────────────────────────┘  │
├─────────────────────────────────────┤
│       SQLite/PostgreSQL             │
│  (Chat history + metadata)          │
├─────────────────────────────────────┤
│            Qdrant                   │
│  (Vector embeddings)                │
└─────────────────────────────────────┘
```

---

## Final Verdict

**Status:** **Needs Significant Revision Before Approval**
