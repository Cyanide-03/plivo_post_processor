import re
from typing import List
from rapidfuzz import process, fuzz

EMAIL_TOKEN_PATTERNS = [
    (r'\b\(?(at|@)\)?\b', '@'),
    (r'\b(dot)\b', '.'),
    (r'\s*@\s*', '@'),
    (r'\s*\.\s*', '.')
]

def add_sentence_punctuation(s: str) -> str:
    s = s.strip()
    # Capitalize first letter
    if s and s[0].islower():
        s = s[0].upper() + s[1:]
    # Add period at end if missing
    if s[-1] not in '.!?,;':
        s += '.'

    return s

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

# def collapse_spelled_letters(s: str) -> str: # ! works for only 5 single letter words
#     # Collapse sequences like 'g m a i l' -> 'gmail'
#     tokens = s.split()
#     out = []
#     i = 0
#     while i < len(tokens):
#         # lookahead for sequences of single letters
#         if all(len(t)==1 for t in tokens[i:i+5]) and i+4 <= len(tokens):
#             out.append(''.join(tokens[i:i+5]))
#             i += 5
#         else:
#             out.append(tokens[i])
#             i += 1
#     return ' '.join(out)

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

def normalize_numbers_spoken(s: str) -> str:
    # Replace simple spoken digit sequences with digits
    tokens = s.split()
    out = []
    i = 0
    while i < len(tokens):
        # greedy take up to 8 tokens for number
        window = tokens[i:i+8]
        wd = words_to_digits(window)
        if len(wd) >= 2:  # treat as a number
            out.append(wd)
            i += len(window)  # move past window
        else:
            out.append(tokens[i])
            i += 1
    return ' '.join(out)

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


def normalize_currency(s: str) -> str:
    # Replace 'rupees ...' preceding digits with ₹
    s = re.sub(r'\brupees\s+', '₹', s, flags=re.IGNORECASE)
    # Insert commas in Indian grouping (basic version) for 5+ digits
    def indian_group(num):
        # last 3, then every 2
        x = str(num)
        if len(x) <= 3: return x
        last3 = x[-3:]
        rest = x[:-3]
        parts = []
        while len(rest) > 2:
            parts.insert(0, rest[-2:])
            rest = rest[:-2]
        if rest: parts.insert(0, rest)
        return ','.join(parts + [last3])
    def repl(m):
        raw = re.sub('[^0-9]', '', m.group(0))
        if not raw: return m.group(0)
        return '₹' + indian_group(int(raw))
    s = re.sub(r'₹\s*[0-9][0-9,\.]*', repl, s)
    return s

def correct_names_with_lexicon(s: str, names_lex: List[str], threshold: int = 90) -> str:
    tokens = s.split()
    out = []
    for t in tokens:
        best = process.extractOne(t, names_lex, scorer=fuzz.ratio)
        if best and best[1] >= threshold:
            out.append(best[0])
        else:
            out.append(t)
    return ' '.join(out)

def generate_candidates(text: str, names_lex: List[str]) -> List[str]:
    cands = set()
    t = text
    # Base clean-ups
    t1 = normalize_email_tokens(t)
    t1 = normalize_numbers_spoken(t1)
    t1 = normalize_currency(t1)
    t1 = correct_names_with_lexicon(t1, names_lex)
    cands.add(t1)

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

    # ensure original too
    cands.add(text)

    # Deduplicate and limit
    out = list(cands)
    out = sorted(out, key=lambda x: len(x))[:5]  # simple cap
    return out
