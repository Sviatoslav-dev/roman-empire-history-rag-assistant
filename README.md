# Roman Empire History RAG Assistant

A chat-based system that helps users explore the history of the Roman Empire by providing contextual answers supported by text and images

## Project Goal

The chatbot helps users learn Roman Empire history and quickly obtain answers to questions on this topic

## Target Users

- School and university students studying ancient history
- Self-learners interested in the Roman Empire
- Anyone looking for concise, source-based historical explanations

## MVP Scope

### What the App Does in the First Version

The first version of the application provides the following functionality:

- Chat-based question answering
- Textual answers enriched with relevant images
- Context-aware conversations across multiple messages
- Ability to create new chats and continue previous conversations

### Context-Aware Conversations

Context awareness is implemented via **query rewriting based on chat history**.

For each user message, the system maintains a lightweight representation of the recent conversation context (e.g., current subject or entity). Follow-up queries are automatically rewritten into fully self-contained queries before retrieval, resolving references such as pronouns or implicit subjects.

### Out of Scope

- Topics unrelated to the history of the Roman Empire
- Autonomous agents and web browsing

## Content & Data

### Data Sources

The dataset is constructed from English Wikipedia articles related to the Roman Empire.

Articles are selected from a predefined set of Wikipedia categories and their subcategories, including:

- Roman Empire(30 pages)
- Byzantine Empire(15 pages)
- Western Roman Empire(10 pages)
- Government of the Roman Empire(35 pages)
- Roman emperors(15 pages)
- Roman Empire art(24 pages)
- History of the Roman Empire(7 pages)
- Wars involving the Roman Empire(64 pages)
- Culture of ancient Rome(68 pages)
- Economy of ancient Rome(39 pages)
- Ancient Roman religion(118 pages)
- Centuries in the Roman Empire with subcategories(60 pages in total)

These categories are chosen to ensure coverage of multiple dimensions of Roman history, including governance, warfare, culture, religion, and societal development.

### Article Processing and Text Chunking

Each article is parsed into structured text chunks based on its internal hierarchy.
- The lowest-level paragraphs are used as the primary chunking unit.
- Each chunk retains its section and subsection titles as metadata.
- Article infoboxes are extracted and stored as separate structured chunks.

Tables are processed depending on their size:
- Small tables are ingested as a single chunk.
- Large tables are split row-by-row, with each row stored as an individual chunk.
- Table chunks include the table title and column context as metadata.

### Image Extraction and Licensing Strategy

Images are extracted from the article content and linked to the specific paragraph in which they appear.

Each image record stores:
- Image URL
- Caption
- Source page
- License type
- Attribution information
- Associated article and paragraph identifiers

Only images with permissive or attribution-based licenses are included(e.g. Public Domain, CC BY, CC BY-SA, GFDL or compatible licenses)

Images classified as non-free content (e.g. fair use, NC, ND, Wikipedia-only, or “all rights reserved”) are excluded from the dataset.

To ensure dataset quality and consistency, articles are filtered using the following criteria:
- Article length exceeds 5,000 characters
- Contains at least 5 citations
- Not marked as dispute and not marked with a Problem or Update box
- Available in the English Wikipedia

Paragraphs marked with citation-needed templates are excluded from ingestion.

Articles that are tangentially related but not focused on Roman history are explicitly excluded, including:
- Modern fiction and entertainment media
- Video games
- Modern locations named after Rome
- Histories of modern states located on former Roman territory


### Dataset Size (Approximate)

In total, the selected Wikipedia categories contain approximately 485 articles. After applying article quality filters, the dataset is expected to contain around 388 articles.

On average, each article contains about 2.6 images, which results in approximately 1,008 extracted images in total. After applying image license filtering licenses, the final dataset is expected to include around 672 usable images.

### Text–Image Alignment

Images are explicitly aligned with the article and paragraph from which they are extracted.
This alignment enables paragraph-level association between textual content and visual material.

## Example Queries

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

## Evaluation Methodology

### Evaluation Dataset

The evaluation dataset consists of 50 queries designed to reflect realistic usage patterns of the system:
- **30 single-turn queries**
- **20 conversational queries**

Query distribution:
- 20 factual (text-heavy) queries
- 15 image-centric queries
- 15 conversational queries organized into 3 multi-turn chains (5 turns per chain)

### Gold Standard Definition

For each query, a gold standard was manually constructed:

1. Identification of:
   - Expected Wikipedia articles (1–3 per query)
   - Relevant text sections at paragraph level
   - Expected images (where applicable)
2. Annotations were performed independently by two annotators.
3. Disagreements were resolved through manual review.
4. Final annotations were stored in `eval_queries.json`.

### Retrieval Evaluation

#### Text Retrieval (30 queries)

The following metrics are used to evaluate text retrieval quality:

- **Recall@5**  
  Measures whether at least one gold-standard chunk appears in the top-5 retrieved results.  
  Target: ≥ 75%

- **Mean Reciprocal Rank (MRR)**  
  Measures the average rank position of the first relevant retrieved chunk.  
  Target: ≥ 0.6

- **Precision@5**  
  Measures the proportion of relevant chunks among the top-5 results.  
  Target: ≥ 60%

#### Image Retrieval (15 image-centric queries)

Image retrieval performance is evaluated using:

- **Image Hit Rate@3**  
  Measures whether the expected image appears in the top-3 retrieved results.  
  Target: ≥ 70%

- **Image Type Accuracy**  
  Verifies whether retrieved images match the expected semantic type (e.g., maps for map-based queries).  
  Target: ≥ 80%

### Conversational Retrieval Evaluation (20 queries, 5 chains)

Conversational performance is evaluated across multi-turn interactions:

- **Context Resolution Rate**  
  Measures correct resolution of references in follow-up queries (e.g., pronouns such as “he”).  
  Target: ≥ 80%

- **Topic Coherence**  
  Measures whether retrieved sources remain consistent with the conversation topic across turns.  
  Target: ≥ 75%

### Generation Evaluation

#### Faithfulness

- Manual binary evaluation is applied to each generated response.
- Answers must be fully grounded in retrieved sources.
- Any hallucinated facts (dates, names, events) are considered failures.  
- Target: 100% faithfulness (zero tolerance for historical inaccuracies).

#### Citation Quality

- Generated answers must include explicit Wikipedia citations.
- Citation format:  
  `[Source: Wikipedia – Julius Caesar, Section: Early Career]`
- Target: ≥ 95% of responses include correct citations.

#### Conversational Coherence

- Assesses the model’s ability to maintain context across turns and handle topic shifts.
- Evaluated manually on a 3-point scale:
  - Good (2)
  - Adequate (1)
  - Poor (0)
- Target: Average score ≥ 1.5.

### Latency Evaluation

End-to-end latency is measured for each query, including retrieval and generation:

- **Single-turn queries:** < 5 seconds
- **Conversational queries:** < 7 seconds

Latency breakdown targets:
- Retrieval: < 2 seconds
- Generation: < 4 seconds

### Error Analysis

Errors are categorized into the following groups:

- Query ambiguity
- Missing information in the corpus
- Incorrect text or image retrieval
- Generation hallucinations despite correct context
- Context resolution failures in conversational queries
- Image licensing or availability issues

### Reporting

The evaluation results are summarized in a structured report containing:

- Aggregate metrics table
- Per-query performance breakdown
- Identification of the most challenging queries
- Failure case examples with root-cause analysis
- Actionable improvement suggestions

## User Interface Expectations

Users should be able to see:

- The generated answer
- Related images(with sourse)
- Full chat history
- A list of chats

## Architecture

### Conversational Context Management

1. Chat Session Storage:
- Each chat has unique session_id
- Store messages: [{role, content, timestamp, sources}]
- Store metadata: {topic, key_entities, created_at}
2. Query Rewriting.
  
3. Entity Tracking:
- Track mentioned entities (emperors, battles, places)
- Resolve pronouns: "he" → "Augustus", "it" → "Roman Empire"
4. Context Window Management:
- Include last N messages (N=5 recommended)
- Summarize older messages if chat is long
- Avoid context overflow (stay within token limits)

Example Flow:

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

### Multimodal Retrieval Strategy

Dual-Collection Approach:

Collection 1: Text Chunks
- Source: Wikipedia article sections
- Embedding: sentence-transformers/all-MiniLM-L6-v2
- Metadata: {article_title, section, url, related_images}

Collection 2: Image Embeddings
- Source: Wikipedia images (Commons)
- Embedding: OpenCLIP
- Metadata: {corresponding text section, caption, article, license, attribution}


The system performs parallel multimodal retrieval, where text and images are retrieved independently in response to the same user query.

At query time, two searches are executed in parallel:
- A semantic search over the text collection to retrieve the most relevant textual chunks.
- A text-to-image similarity search over the image collection to retrieve visually relevant images.

Textual and visual results are not derived from one another and do not depend on prior filtering. Instead, both modalities contribute independently to the final context. The application layer merges the retrieved text chunks and images into a single multimodal result set, which is then passed to the downstream generation or reasoning component.

When a query includes both text and image input, each modality is processed independently.
The text input is used for semantic text retrieval, while the image input is used to retrieve visually similar images. Retrieved images will additionally contribute related textual context through their associated captions or article sections. All results are merged at the application level into a unified multimodal context.

### Backend Architecture
```
┌─────────────────────────────────────┐
│         Streamlit UI                │
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
## Data Ingestion

Phase 1: Article Collection

- Step 1: Retrieve article links from Wikipedia categories

- Step 2: Download HTML pages

- Step 3: Filter low-quality articles

Phase 2: Text & Image Processing

- Step 1: Parse HTML and clean content

- Step 2: Split articles into text chunks

- Step 3: Extract images and validate licenses

- Step 4: Create text–image correspondences

Phase 3: Data Storage

- Step 1: Store text chunks in the database

- Step 2: Store images and link them to corresponding text

### Expected Output

data/articles/ - Raw Wikipedia articles (HTML)

data/images/ - Downloaded images (JPG / PNG / SVG)

Qdrant collections:

- texts — vectorized text chunks with metadata
- images — vectorized images with metadata

## Technical Stack

**Language Model:** LLaMA 3 8B

**Text embeddings:** sentence-transformers/all-MiniLM-L6-v2

**Image embeddings:** OpenCLIP

**Vector Database:** Qdrant

## Differentiation from Existing Solutions

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
