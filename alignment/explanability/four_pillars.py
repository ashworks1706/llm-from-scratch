# four_pillars.py
# This module provides functions to showcase the four pillars of explainability:
# 1. Data Distribution - Understanding text length and vocabulary
# 2. Semantic Patterns - Topic modeling and semantic clustering
# 3. Linguistic Features - Analyzing complexity, readability, and syntactic patterns
# 4. Token Relationships - Visualizing n-gram patterns and word co-occurrences

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from collections import Counter
import nltk
from nltk.util import ngrams
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA, LatentDirichletAllocation
from wordcloud import WordCloud
import plotly.express as px
import plotly.graph_objects as go
from textstat import textstat
import warnings
nltk.download('punkt_tab')
nltk.download('punkt')
nltk.download('stopwords')
# Download necessary NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
    
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt', quiet=True)  
def visualize(dataset, sample_size=1000, pillars=None):
    """
    Visualize the four pillars of explainability for text data.
    
    Args:
        dataset: A Hugging Face dataset object
        sample_size: Number of examples to sample for visualization (default: 1000)
        pillars: List of pillars to visualize (1, 2, 3, 4) or None for all pillars
    
    Returns:
        None, displays visualizations
    """
    warnings.filterwarnings('ignore')
    print("Analyzing dataset for the four pillars of explainability...")
    
    if 'train' in dataset:
        data = dataset['train'].shuffle(seed=42).select(range(min(sample_size, len(dataset['train']))))
    else:
        first_split = list(dataset.keys())[0]
        data = dataset[first_split].shuffle(seed=42).select(range(min(sample_size, len(dataset[first_split]))))
    
    # Extract text field - assuming 'text' is the main field, adjust if needed
    text_field = 'text' if 'text' in data.features else list(data.features.keys())[0]
    texts = data[text_field]
    
    # Setting up the figure layout
    plt.figure(figsize=(20, 20))
    
    # Default to all pillars if none specified
    if pillars is None:
        pillars = [1, 2, 3, 4]
    
    # Call each selected pillar function
    if 1 in pillars:
        pillar1_data_distribution(texts)
    if 2 in pillars:
        pillar2_semantic_patterns(texts)
    if 3 in pillars:
        pillar3_linguistic_features(texts)
    if 4 in pillars:
        pillar4_token_relationships(texts)
    
    plt.tight_layout()
    plt.show()
    
    print("Analysis complete!")
def pillar1_data_distribution(texts):
    """Visualize data distribution: text length, vocabulary distribution"""
    print("\n📊 PILLAR 1: DATA DISTRIBUTION")
    
    # Calculate text lengths
    text_lengths = [len(text.split()) for text in texts]
    char_lengths = [len(text) for text in texts]
    
    # Create a figure with multiple subplots
    fig = plt.figure(figsize=(15, 10))
    
    # 1. Text Length Distribution
    plt.subplot(2, 2, 1)
    sns.histplot(text_lengths, kde=True)
    plt.title('Distribution of Text Lengths (Words)')
    plt.xlabel('Number of Words')
    plt.ylabel('Frequency')
    
    # 2. Character Length Distribution
    plt.subplot(2, 2, 2)
    sns.histplot(char_lengths, kde=True)
    plt.title('Distribution of Text Lengths (Characters)')
    plt.xlabel('Number of Characters')
    plt.ylabel('Frequency')
    
    # 3. Vocabulary Analysis - Word Frequency
    all_words = ' '.join(texts).lower().split()
    word_freq = Counter(all_words).most_common(30)
    words, freqs = zip(*word_freq)
    
    plt.subplot(2, 2, 3)
    sns.barplot(x=list(words), y=list(freqs))
    plt.title('Top 30 Most Common Words')
    plt.xticks(rotation=45, ha='right')
    plt.xlabel('Words')
    plt.ylabel('Frequency')
    # 4. Word Cloud
    plt.subplot(2, 2, 4)
    wordcloud = WordCloud(width=800, height=400, background_color='white', 
                          max_words=200, contour_width=3, contour_color='steelblue')
    wordcloud.generate(' '.join(texts))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title('Word Cloud Visualization')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Print key statistics
    print(f"Average text length: {np.mean(text_lengths):.2f} words")
    print(f"Median text length: {np.median(text_lengths):.2f} words")
    print(f"Vocabulary size: {len(set(all_words))} unique words")
     

def pillar2_semantic_patterns(texts):
    """Visualize semantic patterns: topics, embeddings, clusters"""
    print("\n🧩 PILLAR 2: SEMANTIC PATTERNS")
    
    # Create vectorizer
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(texts)
    
    # Topic Modeling with LDA
    n_topics = 5
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(tfidf_matrix)
    
    # Get top words for each topic
    feature_names = vectorizer.get_feature_names_out()
    
    # Create figure for topic visualization
    fig = plt.figure(figsize=(15, 10))
    
    # 1. Topic Keywords Visualization
    n_top_words = 10
    topic_keywords = []
    
    for topic_idx, topic in enumerate(lda.components_):
        top_keywords_idx = topic.argsort()[:-n_top_words - 1:-1]
        top_keywords = [feature_names[i] for i in top_keywords_idx]
        topic_keywords.append(top_keywords)
    
    # Plotting topic keywords
    for i, keywords in enumerate(topic_keywords):
        plt.subplot(2, 3, i+1)
        y_pos = np.arange(len(keywords))
        topic_importance = lda.components_[i][lda.components_[i].argsort()[:-n_top_words - 1:-1]]
        plt.barh(y_pos, topic_importance)
        plt.yticks(y_pos, keywords)
        plt.title(f'Topic {i+1} Keywords')
    
    # 2. Document-Topic Distribution
    doc_topic_dist = lda.transform(tfidf_matrix)
    
    # Create a stacked chart for document-topic distribution
    plt.subplot(2, 3, 6)
    
    # Sample a subset of documents for visualization
    n_docs = min(50, len(texts))
    doc_indices = np.random.choice(len(texts), n_docs, replace=False)
    
    # Create the stacked bar chart
    bottom = np.zeros(n_docs)
    for topic_idx in range(n_topics):
        plt.bar(range(n_docs), doc_topic_dist[doc_indices, topic_idx], bottom=bottom, label=f'Topic {topic_idx+1}')
        bottom += doc_topic_dist[doc_indices, topic_idx]
    
    plt.title('Document-Topic Distribution')
    plt.xlabel('Documents')
    plt.ylabel('Topic Probability')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=n_topics)
    
    plt.tight_layout()
    plt.show()
    
    # Print key insights
    print(f"Number of topics analyzed: {n_topics}")
    print("Top keywords for each topic:")
    for i, keywords in enumerate(topic_keywords):
        print(f"  Topic {i+1}: {', '.join(keywords[:5])}")

def pillar3_linguistic_features(texts):
    """Visualize linguistic features: complexity, sentiment, readability"""
    print("\n📝 PILLAR 3: LINGUISTIC FEATURES")
    
    # Calculate readability scores
    flesch_scores = [textstat.flesch_reading_ease(text) for text in texts]
    grade_levels = [textstat.text_standard(text, float_output=True) for text in texts]
    
    # Calculate sentence lengths
    sentence_lengths = []
    for text in texts:
        sentences = nltk.sent_tokenize(text)
        for sentence in sentences:
            words = nltk.word_tokenize(sentence)
            sentence_lengths.append(len(words))
    
    # Calculate average word length
    word_lengths = []
    for text in texts:
        words = nltk.word_tokenize(text)
        for word in words:
            if word.isalpha():  # Only consider alphabetic words
                word_lengths.append(len(word))
    
    # Create visualization
    fig = plt.figure(figsize=(15, 10))
    
    # 1. Readability Score Distribution
    plt.subplot(2, 2, 1)
    sns.histplot(flesch_scores, kde=True)
    plt.title('Flesch Reading Ease Score Distribution')
    plt.xlabel('Flesch Score (Higher = Easier to Read)')
    plt.ylabel('Frequency')
    
    # 2. Grade Level Distribution
    plt.subplot(2, 2, 2)
    sns.histplot(grade_levels, kde=True)
    plt.title('US Grade Level Distribution')
    plt.xlabel('US Grade Level')
    plt.ylabel('Frequency')
    
    # 3. Sentence Length Distribution
    plt.subplot(2, 2, 3)
    sns.histplot(sentence_lengths, kde=True)
    plt.title('Sentence Length Distribution')
    plt.xlabel('Words per Sentence')
    plt.ylabel('Frequency')
    
    # 4. Word Length Distribution
    plt.subplot(2, 2, 4)
    sns.histplot(word_lengths, kde=True)
    plt.title('Word Length Distribution')
    plt.xlabel('Characters per Word')
    plt.ylabel('Frequency')
    
    plt.tight_layout()
    plt.show()
    
    # Print key insights
    print(f"Average Flesch Reading Ease: {np.mean(flesch_scores):.2f} (90-100: Very Easy, 0-30: Very Difficult)")
    print(f"Average US Grade Level: {np.mean(grade_levels):.2f}")
    print(f"Average Sentence Length: {np.mean(sentence_lengths):.2f} words")
    print(f"Average Word Length: {np.mean(word_lengths):.2f} characters")

def pillar4_token_relationships(texts):
    """Visualize token relationships: n-grams, co-occurrences"""
    print("\n🔄 PILLAR 4: TOKEN RELATIONSHIPS")
    
    # Join all texts and tokenize
    all_tokens = []
    for text in texts:
        tokens = nltk.word_tokenize(text.lower())
        all_tokens.extend([token for token in tokens if token.isalpha()])
    
    # Generate n-grams
    bigrams_list = list(ngrams(all_tokens, 2))
    trigrams_list = list(ngrams(all_tokens, 3))
    
    # Count frequencies
    bigram_freq = Counter(bigrams_list).most_common(15)
    trigram_freq = Counter(trigrams_list).most_common(15)
    
    # Create string representations for plotting
    bigram_labels = [' '.join(bigram) for bigram, _ in bigram_freq]
    bigram_counts = [count for _, count in bigram_freq]
    
    trigram_labels = [' '.join(trigram) for trigram, _ in trigram_freq]
    trigram_counts = [count for _, count in trigram_freq]
    
    # Create figure
    fig = plt.figure(figsize=(15, 10))
    
    # 1. Bigram Frequency
    plt.subplot(2, 2, 1)
    sns.barplot(x=bigram_counts, y=bigram_labels)
    plt.title('Top 15 Bigrams')
    plt.xlabel('Frequency')
    
    # 2. Trigram Frequency
    plt.subplot(2, 2, 2)
    sns.barplot(x=trigram_counts, y=trigram_labels)
    plt.title('Top 15 Trigrams')
    plt.xlabel('Frequency')
    
    # 3. Word Transition Network (Co-occurrence)
    # Find top word pairs for network visualization
    top_pairs = Counter(bigrams_list).most_common(50)
    
    # Get unique words from top pairs
    unique_words = set()
    for (word1, word2), _ in top_pairs:
        unique_words.add(word1)
        unique_words.add(word2)
    
    # Create word-to-index mapping
    word_to_idx = {word: i for i, word in enumerate(unique_words)}
    
    # Create adjacency matrix
    n_words = len(unique_words)
    adjacency = np.zeros((n_words, n_words))
    
    for (word1, word2), count in top_pairs:
        i, j = word_to_idx[word1], word_to_idx[word2]
        adjacency[i, j] = count
    
    # Create a heatmap of word co-occurrences
    plt.subplot(2, 2, 3)
    # Take top 20 words for better visualization
    top_words = [word for word, _ in Counter(all_tokens).most_common(20)]
    top_indices = [word_to_idx[word] for word in top_words if word in word_to_idx]
    
    if top_indices:
        sub_adjacency = adjacency[np.ix_(top_indices, top_indices)]
        sns.heatmap(sub_adjacency, xticklabels=[list(unique_words)[i] for i in top_indices],
                    yticklabels=[list(unique_words)[i] for i in top_indices], cmap='viridis')
        plt.title('Word Co-occurrence Matrix')
    else:
        plt.text(0.5, 0.5, "Insufficient data for co-occurrence matrix", 
                 horizontalalignment='center', verticalalignment='center')
        plt.title('Word Co-occurrence Matrix (Not Available)')
    
    # 4. Sequential Pattern
    plt.subplot(2, 2, 4)
    follow_counts = {}
    
    # Choose a common word to analyze
    common_words = [word for word, _ in Counter(all_tokens).most_common(5)]
    if common_words:
        target_word = common_words[0]
        
        for i in range(len(all_tokens) - 1):
            if all_tokens[i] == target_word:
                next_word = all_tokens[i + 1]
                if next_word in follow_counts:
                    follow_counts[next_word] += 1
                else:
                    follow_counts[next_word] = 1
        
        top_followers = Counter(follow_counts).most_common(10)
        follower_words = [word for word, _ in top_followers]
        follower_counts = [count for _, count in top_followers]
        
        sns.barplot(x=follower_counts, y=follower_words)
        plt.title(f'Top 10 Words Following "{target_word}"')
        plt.xlabel('Frequency')
    else:
        plt.text(0.5, 0.5, "Insufficient data for sequential pattern analysis", 
                 horizontalalignment='center', verticalalignment='center')
        plt.title('Sequential Pattern Analysis (Not Available)')
    
    plt.tight_layout()
    plt.show()
    
    print(f"Most common bigram: {bigram_labels[0]} (occurs {bigram_counts[0]} times)")
    print(f"Most common trigram: {trigram_labels[0]} (occurs {trigram_counts[0]} times)")
    if common_words:
        print(f"Words most likely to follow '{target_word}': {', '.join(follower_words[:3])}")
