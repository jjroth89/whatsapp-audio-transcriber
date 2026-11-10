import re
from pathlib import Path
from typing import List, Optional, Set
from collections import Counter
from .parser import ChatMessage

# Regex for uppercase acronyms (2 to 6 letters, like CNO, ITBI, RET)
ACRONYM_REGEX = re.compile(r"\b[A-Z]{2,6}\b")
# Regex for capitalized proper words (excluding start of sentences if desired, but good for names)
PROPER_NAME_REGEX = re.compile(r"\b[A-Z][a-zÀ-ÿ]{2,}\b")

# Common Portuguese and English stopwords to avoid polluting Whisper prompt
STOPWORDS = {
    "A", "O", "Os", "As", "Um", "Uma", "De", "Do", "Da", "Dos", "Das",
    "Em", "No", "Na", "Nos", "Nas", "Por", "Para", "Com", "Sem", "Que",
    "Se", "Mas", "Como", "Quando", "Muito", "Mais", "Hoje", "Ontem", "Amanhã",
    "Você", "Vocês", "Eu", "Ele", "Ela", "Nós", "Eles", "Tudo", "Bem", "Bom",
    "Boa", "Oi", "Ola", "Olá", "The", "And", "Or", "In", "On", "At", "To", "For"
}

def harvest_vocabulary_from_chat(
    messages: List[ChatMessage],
    dictionary_file: Optional[Path] = None,
    max_terms: int = 40
) -> str:
    """
    Harvests specialized vocabulary, acronyms, and proper nouns dynamically from the chat,
    and merges them with an optional user dictionary file.
    """
    acronym_counter = Counter()
    name_counter = Counter()
    custom_terms: List[str] = []

    # 1. Load user dictionary file if provided
    if dictionary_file and dictionary_file.is_file():
        text = dictionary_file.read_text(encoding="utf-8")
        for line in text.splitlines():
            cleaned = line.strip()
            if cleaned and not cleaned.startswith("#"):
                # Split comma separated or line separated
                parts = [p.strip() for p in cleaned.split(",") if p.strip()]
                custom_terms.extend(parts)

    # 2. Extract sender names
    senders = {msg.sender for msg in messages if msg.sender and msg.sender != "System"}

    # 3. Harvest acronyms and proper names from text message contents
    for msg in messages:
        if msg.is_system or not msg.content:
            continue

        # Find acronyms
        for acr in ACRONYM_REGEX.findall(msg.content):
            if acr not in STOPWORDS:
                acronym_counter[acr] += 1

        # Find capitalized words
        for name in PROPER_NAME_REGEX.findall(msg.content):
            if name not in STOPWORDS and len(name) > 2:
                name_counter[name] += 1

    # Select top terms
    top_acronyms = [item[0] for item in acronym_counter.most_common(15)]
    top_names = [item[0] for item in name_counter.most_common(20)]

    # Merge and preserve order while deduplicating
    seen: Set[str] = set()
    combined_list: List[str] = []

    # Custom terms take highest priority
    for term in custom_terms:
        term_clean = term.strip()
        if term_clean and term_clean.lower() not in seen:
            seen.add(term_clean.lower())
            combined_list.append(term_clean)

    # Then senders
    for sender in sorted(senders):
        if sender.lower() not in seen:
            seen.add(sender.lower())
            combined_list.append(sender)

    # Then top acronyms
    for acr in top_acronyms:
        if acr.lower() not in seen:
            seen.add(acr.lower())
            combined_list.append(acr)

    # Then top names / technical terms
    for name in top_names:
        if name.lower() not in seen:
            seen.add(name.lower())
            combined_list.append(name)

    final_terms = combined_list[:max_terms]
    return ", ".join(final_terms)
