import re
from typing import List
from rapidfuzz import process, fuzz

EMAIL_TOKEN_PATTERNS = [
    (r'\b\(?(at|@)\)?\b', '@'),
    (r'\b(dot)\b', '.'),
    (r'\s*@\s*', '@'),
    (r'\s*\.\s*', '.')
]

# ensures basic sentence-level capitalization and ending punctuation.
def add_sentence_punctuation(s: str) -> str:
    s = s.strip()
    # Capitalize first letter
    if s and s[0].islower():
        s = s[0].upper() + s[1:]
    # Add period at end if missing
    if s[-1] not in '.!?,;':
        s += '.'

    return s

# converts sequences of three or more single-letter tokens (e.g., 'g m a i l') 
# into a single, combined, lowercase word (e.g., 'gmail').
def collapse_spelled_letters(s: str) -> str:
    """
    Collapse sequences like 'g m a i l' -> 'gmail' or 'a b c' -> 'abc'.
    Now works for ANY length sequence of 3+ single letters.
    """
    tokens = s.split()
    out = []
    i = 0
    
    while i < len(tokens):
        # Look ahead for consecutive single-letter tokens
        j = i
        while j < len(tokens) and len(tokens[j]) == 1 and tokens[j].isalpha():
            j += 1
        
        # If we found 3+ consecutive single letters, collapse them
        if j - i >= 3:
            out.append(''.join(tokens[i:j]).lower())
            i = j
        else:
            out.append(tokens[i])
            i += 1
    
    return ' '.join(out)

# cleans up common spoken email components and formats the address structure.
def normalize_email_tokens(s: str) -> str:
    s2 = s
    s2 = collapse_spelled_letters(s2)
    for pat, rep in EMAIL_TOKEN_PATTERNS:
        s2 = re.sub(pat, rep, s2, flags=re.IGNORECASE)
    # remove spaces around @ and . inside emails
    s2 = re.sub(r'\s*([@\.])\s*', r'\1', s2)

    # Handle common email domain patterns
    s2 = re.sub(r'gmail\.com', 'gmail.com', s2, flags=re.IGNORECASE)
    s2 = re.sub(r'yahoo\.com', 'yahoo.com', s2, flags=re.IGNORECASE)

    return s2

# Numbers: handle 'double nine', 'triple zero', 'oh' for zero
NUM_WORD = {
    'zero':'0','oh':'0','o':'0','one':'1','two':'2','three':'3','four':'4','five':'5',
    'six':'6','seven':'7','eight':'8','nine':'9'
}

# "five lakh thirty thousand" → "530000"
INDIAN_NUM_WORDS = {
    'hundred': 100,
    'thousand': 1000,
    'lakh': 100000,
    'lac': 100000,
    'crore': 10000000,
    'crores': 10000000,
    'lakhs': 100000,
    'lacs': 100000
}

# Basic digit words for Indian number conversion
DIGIT_WORDS = {
    'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
    'six': 6, 'seven': 7, 'eight': 8, 'nine': 9,
    'ten': 10, 'eleven': 11, 'twelve': 12, 'thirteen': 13, 'fourteen': 14,
    'fifteen': 15, 'sixteen': 16, 'seventeen': 17, 'eighteen': 18, 'nineteen': 19,
    'twenty': 20, 'thirty': 30, 'forty': 40, 'fifty': 50,
    'sixty': 60, 'seventy': 70, 'eighty': 80, 'ninety': 90
}

# addresses the unique grouping used in the Indian numbering system.
def convert_indian_spoken_numbers(s: str) -> str:
    """
    Convert spoken Indian numbers to digits.
    Examples:
    - "five lakh thirty thousand" → "530000"
    - "two crore fifty lakh" → "25000000"
    """
    s_lower = s.lower()
    
    # Pattern: number + multiplier (lakh, crore, thousand, hundred)
    for multiplier, value in sorted(INDIAN_NUM_WORDS.items(), key=lambda x: -x[1]):
        # Find patterns like "five lakh", "fifty thousand", etc.
        pattern = r'\b(' + '|'.join(DIGIT_WORDS.keys()) + r')\s+' + multiplier + r'\b'
        
        def replace_num(match):
            num_word = match.group(1)
            num_val = DIGIT_WORDS.get(num_word, 1)
            return str(num_val * value)
        
        s_lower = re.sub(pattern, replace_num, s_lower)
    
    # Handle "lakh" or "crore" without preceding number (assume 1)
    for multiplier, value in INDIAN_NUM_WORDS.items():
        s_lower = re.sub(r'\b' + multiplier + r'\b', str(value), s_lower)
    
    return s_lower

# converts a list of number-related words (e.g., 'nine', 'double', 'oh') 
# into a string of digits.
def words_to_digits(seq: List[str]) -> str:
    """
    Convert word sequences to digits.
    Handles: 'double nine' → '99', 'triple zero' → '000', 'oh' → '0'
    """
    out = []
    i = 0
    while i < len(seq):
        tok = seq[i].lower()
        if tok in ('double','triple') and i+1 < len(seq):
            times = 2 if tok=='double' else 3
            nxt = seq[i+1].lower()
            if nxt in NUM_WORD:
                out.append(NUM_WORD[nxt]*times)
                i += 2
                continue
        if tok in NUM_WORD:
            out.append(NUM_WORD[tok])
            i += 1
        else:
            # Not a number word, stop processing this sequence
            break
    return ''.join(out)

# Convert any sequence of spoken digits or Indian number phrases 
# into a single string of numerals.
def normalize_numbers_spoken(s: str) -> str:
    """
    Replace spoken digit sequences with actual digits.
    Improved to handle variable-length sequences and 'oh' in context.
    """
    # First, convert Indian number words (lakh, crore)
    s = convert_indian_spoken_numbers(s)
    
    tokens = s.split()
    out = []
    i = 0
    
    while i < len(tokens):
        # Try to convert a sequence of number words
        # Look ahead up to 12 tokens (for longer phone numbers)
        max_window = min(12, len(tokens) - i)
        
        # Try progressively smaller windows
        converted = False
        for window_size in range(max_window, 0, -1):
            window = tokens[i:i + window_size]
            wd = words_to_digits(window)
            
            # If we got at least 2 digits, it's likely a number
            if len(wd) >= 2:
                out.append(wd)
                i += window_size
                converted = True
                break
        
        if not converted:
            out.append(tokens[i])
            i += 1
    
    return ' '.join(out)

# standardizes currency notation to the Indian Rupee symbol ($\text{₹}$) 
# and applies Indian comma grouping.
def normalize_currency(s: str) -> str:
    """
    Replace 'rupees' with ₹ and format with Indian digit grouping.
    """
    # Replace 'rupees' with ₹ symbol
    s = re.sub(r'\brupees?\s+', '₹', s, flags=re.IGNORECASE)
    s = re.sub(r'\brs\.?\s+', '₹', s, flags=re.IGNORECASE)  # Handle "Rs."
    
    def indian_group(num):
        """Format number with Indian grouping: X,XX,XXX"""
        x = str(num)
        if len(x) <= 3:
            return x
        
        last3 = x[-3:]
        rest = x[:-3]
        parts = []
        
        while len(rest) > 2:
            parts.insert(0, rest[-2:])
            rest = rest[:-2]
        
        if rest:
            parts.insert(0, rest)
        
        return ','.join(parts + [last3])
    
    def repl(m):
        """Replace currency amount with formatted version"""
        raw = re.sub('[^0-9]', '', m.group(0))
        if not raw:
            return m.group(0)
        return '₹' + indian_group(int(raw))
    
    # Find and format currency amounts
    s = re.sub(r'₹\s*[0-9][0-9,\.]*', repl, s)
    
    return s

# correct potentially misspelled names using fuzzy matching against a 
# predefined list (lexicon) of correct names.
def correct_names_with_lexicon(s: str, names_lex: List[str], threshold: int = 85) -> str:
    """
    Correct names using fuzzy matching against lexicon.
    Improved: Only match capitalized words or words that could be names.
    """
    if not names_lex:
        return s
    
    # Common words to skip (not names)
    COMMON_WORDS = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'can', 'my', 'your', 'his', 'her', 'their',
        'this', 'that', 'these', 'those', 'rupees', 'email', 'call', 'contact'
    }
    
    tokens = s.split()
    out = []
    
    for t in tokens:
        # Skip common words and very short words
        if t.lower() in COMMON_WORDS or len(t) < 3:
            out.append(t)
            continue
        
        # Skip words that contain numbers or special characters (likely not names)
        if re.search(r'[0-9@._-]', t):
            out.append(t)
            continue
        
        # Only try to correct if word starts with capital or is all lowercase
        # (might be incorrectly lowercased name)
        if t[0].isupper() or t.islower():
            best = process.extractOne(t, names_lex, scorer=fuzz.ratio)
            if best and best[1] >= threshold:
                out.append(best[0])
            else:
                out.append(t)
        else:
            out.append(t)
    
    return ' '.join(out)

def generate_candidates(text: str, names_lex: List[str]) -> List[str]:
    cands = set()
    t = text
    
    # Full pipeline with punctuation
    t1 = normalize_email_tokens(t)
    t1 = normalize_numbers_spoken(t1)
    t1 = normalize_currency(t1)
    t1 = correct_names_with_lexicon(t1, names_lex)
    t1 = add_sentence_punctuation(t1)
    cands.add(t1)

    # Email + numbers + punctuation (no names, no currency)
    t2 = normalize_email_tokens(text)
    t2 = normalize_numbers_spoken(t2)
    t2 = add_sentence_punctuation(t2)
    cands.add(t2)

    # Currency + numbers + punctuation
    t3 = normalize_numbers_spoken(text)
    t3 = normalize_currency(t3)
    t3 = add_sentence_punctuation(t3)
    cands.add(t3)

    # Names + punctuation only
    t4 = correct_names_with_lexicon(text, names_lex)
    t4 = add_sentence_punctuation(t4)
    cands.add(t4)

    # Variants
    # Variant 2: only email normalization
    t2 = correct_names_with_lexicon(normalize_email_tokens(text), names_lex)
    cands.add(t2)

    # Variant 3: currency + numbers only
    t3 = normalize_currency(normalize_numbers_spoken(text))
    cands.add(t3)

    # Variant 4: names only
    t4 = correct_names_with_lexicon(text, names_lex)
    cands.add(t4)

    # Email + names + punctuation
    t5 = normalize_email_tokens(text)
    t5 = correct_names_with_lexicon(t5, names_lex)
    t5 = add_sentence_punctuation(t5)
    cands.add(t5)

    # Just punctuation on original
    t6 = add_sentence_punctuation(text)
    cands.add(t6)

    # ensure original too
    cands.add(text)

    # Deduplicate and limit
    out = list(cands)
    out = sorted(out, key=lambda x: len(x))[:8]  # simple cap
    return out
