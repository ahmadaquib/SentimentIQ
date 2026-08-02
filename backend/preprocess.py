import re
import string 

def clean_text(text: str) -> str:
    
    """
    Cleans raw input text for NLP processing.
    
    Steps:
    - Converts to lowercase
    - Removes URLs
    - Removes punctuation
    - Strips extra whitespace

    Args:
        text: Raw input string
    Returns:
        Cleaned string
    """

    #Lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove extra whitespaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

if __name__ == '__main__':
    test = "Check this out! https://example.com                It's AMAZING!!!"
    print(clean_text(test))