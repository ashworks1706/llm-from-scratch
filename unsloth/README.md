### 1. **Personal Code Assistant**
**Goal**: Fine-tune a 1B model to help with your specific coding tasks
- **Dataset**: CodeAlpaca + your own code examples
- **Skills**: Basic SFT, data formatting
- **Use case**: Generates code in your style/framework

### 2. **Domain Expert Chatbot** 
**Goal**: Create a chatbot expert in a specific field (medical, legal, gaming)
- **Dataset**: Wikipedia articles + Q&A from domain
- **Skills**: Chat templates, multi-turn conversations
- **Use case**: Answers questions about specific topics

### 4. **Helpful & Harmless Assistant**
**Goal**: Align a chatbot to be both helpful and refuse harmful requests
- **Pipeline**: SFT (helpful responses) → DPO (prefer safe over unsafe)
- **Dataset**: Anthropic HH-RLHF
- **Skills**: Full SFT → DPO pipeline

### 5. **Math Reasoning Specialist**
**Goal**: Fine-tune for step-by-step math problem solving
- **Dataset**: GSM8K, MATH dataset
- **Skills**: Chain-of-thought prompting, evaluation
- **Use case**: Solve grade school to competition math
- **Tip**: Use DPO with correct vs incorrect solutions

### 7. **Fact-Checker with Citations**
**Goal**: Model that verifies claims and provides sources
- **Pipeline**: SFT (claim + sources → verdict) → DPO (prefer cited over uncited)
- **Dataset**: FEVER dataset + custom preference pairs
- **Skills**: Structured outputs, preference learning

### 8. **Iterative Self-Improvement**
**Goal**: Model improves its own responses through self-critique
- **Pipeline**: 
  1. SFT on base responses
  2. Generate critiques and improvements
  3. DPO on original vs improved
  4. Repeat
- **Dataset**: UltraFeedback + self-generated data
- **Skills**: Data flywheel, synthetic data generation

### 9. **Multi-Modal Document Understanding**
**Goal**: Process documents with text + tables + images
- **Model**: Start with Llama + Vision adapter or use PaliGemma
- **Dataset**: DocVQA, InfographicVQA
- **Skills**: Multi-modal fine-tuning, structured extraction
- **Challenge**: Handling different document layouts

