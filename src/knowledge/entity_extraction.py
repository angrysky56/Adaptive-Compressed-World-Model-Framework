"""
Entity Extraction Module

This module provides utilities for extracting important entities
and concepts from text for the Adaptive Compressed World Model Framework.
"""

import re
from typing import List, Set, Dict, Tuple

# Common stopwords that should be excluded from entity extraction
STOPWORDS = {
    'a', 'an', 'the', 'and', 'or', 'but', 'if', 'then', 'else', 'when',
    'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into',
    'through', 'during', 'before', 'after', 'above', 'below', 'from',
    'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
    'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why',
    'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other',
    'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
    'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should',
    'now', 'to', 'of', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'this',
    'that', 'these', 'those', 'am', 'it', 'its', 'they', 'them', 'their',
    'what', 'which', 'who', 'whom', 'would', 'could', 'should', 'shall',
    'also', 'many', 'might', 'must', 'said'
}

def extract_entities(text: str, min_word_length: int = 3, max_entities: int = 15) -> List[str]:
    """
    Extract important entities from text using simple heuristics.
    
    This function uses a combination of frequency analysis, capitalization detection,
    and filtering to identify likely important terms in the text.
    
    Args:
        text: The input text to analyze
        min_word_length: Minimum length of words to consider
        max_entities: Maximum number of entities to return
        
    Returns:
        List of extracted entities sorted by importance
    """
    if not text:
        return []
    
    # Extract all words and clean them
    words = re.findall(r'\b[a-zA-Z][a-zA-Z0-9]*\b', text)
    words = [word.lower() for word in words]
    
    # Count word frequencies (excluding stopwords and short words)
    word_counts = {}
    for word in words:
        if word not in STOPWORDS and len(word) >= min_word_length:
            word_counts[word] = word_counts.get(word, 0) + 1
    
    # Find capitalized phrases (potential named entities)
    named_entities = set()
    capitalized_phrases = re.findall(r'\b([A-Z][a-z]+ )+[A-Z][a-z]+\b|\b[A-Z][a-z]+\b', text)
    
    for phrase in capitalized_phrases:
        # Skip if it's at the beginning of a sentence (common words can be capitalized there)
        if not re.search(r'^\s*' + re.escape(phrase), text) and not re.search(r'[.!?]\s+' + re.escape(phrase), text):
            named_entities.add(phrase.lower())
    
    # Prioritize named entities and sort by frequency for regular words
    entities = []
    
    # First add named entities
    for entity in named_entities:
        if entity not in STOPWORDS and len(entity) >= min_word_length:
            entities.append(entity)
    
    # Then add high-frequency words
    for word, count in sorted(word_counts.items(), key=lambda x: x[1], reverse=True):
        if word not in entities:  # Avoid duplicates
            entities.append(word)
    
    # Limit to max_entities
    return entities[:max_entities]

async def extract_entities_with_llm(text: str, llm_linker, max_entities: int = 10) -> List[str]:
    """
    Extract important entities using LLM assistance.
    
    This function leverages the LLM to identify key concepts and entities 
    in the text that may not be obvious through simple heuristics.
    
    Args:
        text: The input text to analyze
        llm_linker: The LLM context linker to use
        max_entities: Maximum number of entities to return
        
    Returns:
        List of extracted entities sorted by importance
    """
    # First get basic entities using the heuristic approach
    basic_entities = extract_entities(text, max_entities=max_entities * 2)
    
    # If LLM linker is not available, return basic entities only
    if not llm_linker or not hasattr(llm_linker, 'extract_key_concepts'):
        return basic_entities[:max_entities]
    
    # Use LLM to extract and enhance entities
    try:
        # Prepare prompt for the LLM
        prompt = f"""Extract the most important entities or concepts from the following text. 
Focus on technical terms, proper nouns, and key concepts that are central to understanding the content.
Exclude common stopwords, general terms, and words like 'this', 'that', etc.

Text:
{text[:2000]}... # Truncate if too long

Respond with a list of key entities, sorted by importance:"""

        # Get LLM response
        llm_entities = await llm_linker.extract_key_concepts(prompt)
        
        # Combine both sources of entities, prioritizing LLM suggestions
        combined_entities = []
        
        # Add LLM entities first
        for entity in llm_entities:
            if entity.lower() not in STOPWORDS and len(entity) >= 3:
                combined_entities.append(entity)
        
        # Then add basic entities not already included
        for entity in basic_entities:
            if entity not in combined_entities and entity.lower() not in STOPWORDS:
                combined_entities.append(entity)
                
        return combined_entities[:max_entities]
    
    except Exception as e:
        # Fall back to basic entities if LLM processing fails
        print(f"Error in LLM entity extraction: {e}")
        return basic_entities[:max_entities]

def filter_extracted_entities(entities: List[str]) -> List[str]:
    """
    Apply additional filtering to the extracted entities to remove low-quality matches.
    
    Args:
        entities: List of extracted entities
        
    Returns:
        Filtered list of entities
    """
    filtered = []
    
    for entity in entities:
        # Skip single-character entities
        if len(entity) <= 1:
            continue
            
        # Skip purely numeric entities
        if entity.isdigit():
            continue
            
        # Skip entities that are just common words
        if entity.lower() in STOPWORDS:
            continue
            
        # Skip very common words that might not be in stopwords
        common_terms = {'example', 'used', 'using', 'use', 'uses', 'like', 'first', 'second', 'third',
                        'one', 'two', 'three', 'however', 'therefore', 'thus', 'hence', 'often',
                        'sometimes', 'rather', 'instead', 'part', 'whole', 'without', 'within',
                        'although', 'though', 'even', 'next', 'last', 'around', 'throughout', 'every'}
        if entity.lower() in common_terms:
            continue
            
        # Keep the entity if it passes all filters
        filtered.append(entity)
        
    return filtered
