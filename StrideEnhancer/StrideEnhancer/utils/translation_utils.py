import os
from openai import OpenAI

def get_available_languages():
    """
    Get a list of available languages for translation.
    
    Returns:
        List of language names
    """
    return [
        "English", "Spanish", "French", "German", "Italian", "Portuguese", 
        "Dutch", "Russian", "Japanese", "Chinese", "Korean", "Arabic"
    ]

def translate_text(text, target_language):
    """
    Translate text to the target language using OpenAI's API.
    
    Args:
        text: The text to translate
        target_language: The target language
        
    Returns:
        Translated text
    """
    if target_language == "English":
        return text
    
    api_key = os.environ.get("OPENAI_API_KEY")
    
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set")
    
    client = OpenAI(api_key=api_key)
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",  # the newest OpenAI model is "gpt-4o" which was released May 13, 2024
            messages=[
                {
                    "role": "system",
                    "content": f"You are a translator. Translate the given text to {target_language}. "
                               f"Maintain the formatting, including any markdown or special characters."
                },
                {
                    "role": "user",
                    "content": text
                }
            ],
            max_tokens=min(len(text) * 2, 4000)  # Estimate token count for translation
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Translation error: {str(e)}"

def batch_translate(data, target_language):
    """
    Translate multiple fields in a data structure to the target language.
    This can handle nested dictionaries and lists.
    
    Args:
        data: Data structure (dict, list, or string) to translate
        target_language: The target language
        
    Returns:
        Translated data structure
    """
    if target_language == "English":
        return data
    
    if isinstance(data, str):
        return translate_text(data, target_language)
    
    elif isinstance(data, list):
        return [batch_translate(item, target_language) for item in data]
    
    elif isinstance(data, dict):
        translated_dict = {}
        for key, value in data.items():
            # Don't translate certain keys that should remain in the original language
            if key in ["id", "risk_score", "datetime"]:
                translated_dict[key] = value
            else:
                translated_dict[key] = batch_translate(value, target_language)
        return translated_dict
    
    else:
        # For numbers, booleans, None, etc.
        return data