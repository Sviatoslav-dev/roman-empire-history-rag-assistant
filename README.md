# Roman Empire History RAG Assistant
A chat-based system that helps users explore the history of the Roman Empire by providing contextual answers supported by text and images

## Project Goal
The chatbot helps users learn Roman Empire history and quickly obtain answers to questions on this topic

## Target Users
- School and university students studying ancient history
- Self-learners interested in the Roman Empire
- Anyone looking for concise, source-based historical explanations

## MVP Scope
### What will the app do in this first version
- Chat-based question answering
- Textual answers enriched with relevant images
- Context-aware conversations across multiple messages
- Ability to create new chats and continue previous conversations

### Out of Scope
- Topics unrelated to the history of the Roman Empire
- Autonomous agents and web browsing

## Content & Data
### Data Sources
Wikipedia articles related to the Roman Empire

### Dataset Size (Approximate)
Around 600 text documents and around 1,800 associated images

### Text–Image Alignment
Images are linked to the same article and paragraph as the corresponding text

## Example Queries
1. What are the main periods in the history of the Roman Empire?
2. What is this? (in the photo)
3. What did the borders look like during the period of the empire’s greatest territorial extent?
4. Who was Trajan?
5. Who is depicted in this image?
6. Which century does this belong to? (in the image)
7. Find similar artifacts (to the attached image)
8. Why was the capital moved from Rome to Constantinople?
9. Why did the Roman Empire collapse?
10. What did the borders of the Western and Eastern Roman Empires look like?
11. How did Roman coinage change during the reign of Augustus?

## Success Metrics
**Retrieval relevance:** top-k results contain the correct source for at least 70% of test queries

**Answer faithfulness:** generated answers are supported by retrieved sources

**Latency:** less than 5 seconds per query

## User Interface Expectations
Users should be able to see:
- The generated answer
- Related images
- Full chat history
- A list of chats

## Technical Stack
**Language Model:** LLaMA 3 8B

**Text embeddings:** sentence-transformers/all-MiniLM-L6-v2

**Image embeddings:** OpenCLIP

**Vector Database:** Qdrant
