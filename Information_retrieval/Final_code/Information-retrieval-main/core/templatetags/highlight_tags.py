import re
from django import template
from django.utils.safestring import mark_safe

register = template.Library()

STOPWORDS = {
    "a","an","the","and","or","but","if","then","else","for","to","of","in","on","at","by","with","as",
    "is","are","was","were","be","been","being","this","that","these","those","it","its","from","into",
    "we","you","they","he","she","i","me","my","our","your","their","them"
}

@register.filter(name='highlight')
def highlight(text, q):
    if not q:
        return text
    
    # Split query into words and remove small/stopwords
    words = [re.escape(w) for w in q.split() if w.lower() not in STOPWORDS and len(w) > 1]
    if not words:
        # If all words are stopwords, try highlighting the full query if it's not empty
        if q.strip():
            words = [re.escape(q.strip())]
        else:
            return text
    
    # Create a pattern that matches any of the words
    pattern = re.compile(f'({"|".join(words)})', re.IGNORECASE)
    
    # Wrap matches in <mark> tags
    highlighted = pattern.sub(r'<mark class="search-highlight">\1</mark>', str(text))
    
    return mark_safe(highlighted)
