import os
import base64
import json
import io
from openai import OpenAI
from PIL import Image

def encode_image(image):
    """Convert PIL Image to base64 encoded string."""
    if isinstance(image, Image.Image):
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
    else:
        # If already a file path or file-like object
        return base64.b64encode(image).decode('utf-8')

def analyze_image(image):
    """Analyze an image to identify system components and potential threats."""
    api_key = os.environ.get("OPENAI_API_KEY")
    
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set")
    
    client = OpenAI(api_key=api_key)
    
    # Convert image to base64
    base64_image = encode_image(image)
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",  # the newest OpenAI model is "gpt-4o" which was released May 13, 2024
            messages=[
                {
                    "role": "system",
                    "content": "You are a cybersecurity expert specializing in threat modeling and system architecture analysis."
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Analyze this system diagram image and identify the following:\n"
                                    "1. Key components and their functions\n"
                                    "2. Data flows between components\n"
                                    "3. External interfaces\n"
                                    "4. Potential trust boundaries\n"
                                    "5. Types of data being processed\n"
                                    "Format your response as detailed, structured text that could be used in a threat modeling exercise."
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{base64_image}"}
                        }
                    ]
                }
            ],
            max_tokens=1500
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error analyzing image: {str(e)}"

def analyze_requirements(text):
    """Analyze textual requirements to identify system components and potential threats."""
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
                    "content": "You are a cybersecurity expert specializing in threat modeling and requirements analysis."
                },
                {
                    "role": "user",
                    "content": f"Analyze these system requirements and identify the following:\n"
                               f"1. Key components and their functions\n"
                               f"2. Data flows and interactions\n"
                               f"3. External interfaces and integrations\n"
                               f"4. Types of data being processed\n"
                               f"5. Potential security concerns based on the requirements\n\n"
                               f"Requirements document:\n{text}"
                }
            ],
            max_tokens=1500
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error analyzing requirements: {str(e)}"

def generate_threats_and_mitigations(image=None, requirements=None, framework="STRIDE", hybrid_components=None, language="English"):
    """Generate threats and mitigations based on the system diagram or requirements."""
    api_key = os.environ.get("OPENAI_API_KEY")
    
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set")
    
    client = OpenAI(api_key=api_key)
    
    # Prepare system analysis from either image or text
    system_analysis = ""
    messages = []
    
    # System message explaining the framework
    framework_description = f"You're using the {framework} framework for threat modeling."
    
    if hybrid_components:
        framework_description += f" This is a hybrid approach combining: {', '.join(hybrid_components)}."
    
    messages.append({
        "role": "system",
        "content": f"You are a cybersecurity expert specializing in threat modeling using the {framework} framework. "
                   f"Provide detailed, actionable threat assessments and mitigations."
    })
    
    # Process image if provided
    if image:
        base64_image = encode_image(image)
        image_analysis_prompt = "Analyze this system diagram and identify potential security threats using the "
        image_analysis_prompt += f"{framework} framework. " if framework != "Custom" else "specified hybrid framework. "
        image_analysis_prompt += "For each threat, provide detailed mitigation strategies."
        
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": image_analysis_prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
            ]
        })
    
    # Process requirements if provided
    if requirements:
        text_analysis_prompt = "Analyze these requirements and identify potential security threats using the "
        text_analysis_prompt += f"{framework} framework. " if framework != "Custom" else "specified hybrid framework. "
        text_analysis_prompt += "For each threat, provide detailed mitigation strategies.\n\n"
        text_analysis_prompt += f"Requirements:\n{requirements}"
        
        messages.append({
            "role": "user",
            "content": text_analysis_prompt
        })
    
    # Request JSON format response
    if messages[-1]["role"] == "user":
        if isinstance(messages[-1]["content"], list):
            messages[-1]["content"].append({
                "type": "text", 
                "text": "Return results in JSON format with threats and mitigations structured as an array of objects."
            })
        else:
            messages[-1]["content"] += "\n\nReturn results in JSON format with threats and mitigations structured as an array of objects."
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",  # the newest OpenAI model is "gpt-4o" which was released May 13, 2024
            messages=messages,
            response_format={"type": "json_object"},
            max_tokens=2000
        )
        
        result = response.choices[0].message.content
        
        # Parse JSON response
        try:
            return json.loads(result)
        except json.JSONDecodeError:
            # If response is not valid JSON, return it as is
            return {"error": "Failed to parse JSON response", "raw_response": result}
    
    except Exception as e:
        return {"error": f"Error generating threats and mitigations: {str(e)}"}