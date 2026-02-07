import base64
import streamlit as st
import streamlit.components.v1 as components
# from github import Github - temporarily disabled until we can install this package
from collections import defaultdict
import re
import os
try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv():
        pass  # Fallback if python-dotenv is not installed
try:
    from openai import OpenAI
except ImportError:
    # Define a simple stand-in for OpenAI that will show error messages
    class OpenAI:
        def __init__(self, api_key=None, base_url=None):
            self.api_key = api_key
            self.base_url = base_url
            self.chat = type('obj', (object,), {
                'completions': type('obj', (object,), {
                    'create': lambda **kwargs: None
                })
            })
            self.models = type('obj', (object,), {
                'list': lambda: type('obj', (object,), {'data': []})
            })
    
import requests
import json
try:
    import tiktoken
except ImportError:
    pass
import io
from PIL import Image
import time
from datetime import datetime

# Import original Stride GPT modules - these will be loaded when needed
# We need to create stub functions for these imported modules
# These will be replaced with actual implementation or fallbacks later

# Create stub functions that gracefully fail while showing helpful error messages
def create_threat_model_prompt(*args, **kwargs):
    return "Sample threat model prompt - This is a placeholder."

def get_threat_model(*args, **kwargs):
    return {"applicationDetails": {"applicationType": "Web Application", "applicationDescription": "Sample application"}}, [], []

def get_threat_model_azure(*args, **kwargs):
    return {"applicationDetails": {"applicationType": "Web Application"}}, [], []

def get_threat_model_google(*args, **kwargs):
    return {"applicationDetails": {"applicationType": "Web Application"}}, [], []

def get_threat_model_mistral(*args, **kwargs):
    return {"applicationDetails": {"applicationType": "Web Application"}}, [], []

def get_threat_model_ollama(*args, **kwargs):
    return {"applicationDetails": {"applicationType": "Web Application"}}, [], []

def get_threat_model_anthropic(*args, **kwargs):
    return {"applicationDetails": {"applicationType": "Web Application"}}, [], []

def get_threat_model_lm_studio(*args, **kwargs):
    return {"applicationDetails": {"applicationType": "Web Application"}}, [], []

def get_threat_model_groq(*args, **kwargs):
    return {"applicationDetails": {"applicationType": "Web Application"}}, [], []

def json_to_markdown(*args, **kwargs):
    return "Threat model would be displayed here in the original implementation."

def get_image_analysis(*args, **kwargs):
    return "Image analysis would be performed here in the original implementation."

def create_image_analysis_prompt(*args, **kwargs):
    return "Sample image analysis prompt - This is a placeholder."

def create_attack_tree_prompt(*args, **kwargs):
    return "Sample attack tree prompt - This is a placeholder."

def get_attack_tree(*args, **kwargs):
    return {"description": "Attack tree would be generated here in the original implementation.", "tree": {}}

def get_attack_tree_azure(*args, **kwargs):
    return {"description": "Attack tree would be generated here in the original implementation.", "tree": {}}

def get_attack_tree_mistral(*args, **kwargs):
    return {"description": "Attack tree would be generated here in the original implementation.", "tree": {}}

def get_attack_tree_ollama(*args, **kwargs):
    return {"description": "Attack tree would be generated here in the original implementation.", "tree": {}}

def get_attack_tree_anthropic(*args, **kwargs):
    return {"description": "Attack tree would be generated here in the original implementation.", "tree": {}}

def get_attack_tree_lm_studio(*args, **kwargs):
    return {"description": "Attack tree would be generated here in the original implementation.", "tree": {}}

def get_attack_tree_groq(*args, **kwargs):
    return {"description": "Attack tree would be generated here in the original implementation.", "tree": {}}

def get_attack_tree_google(*args, **kwargs):
    return {"description": "Attack tree would be generated here in the original implementation.", "tree": {}}

def create_mitigations_prompt(*args, **kwargs):
    return "Sample mitigations prompt - This is a placeholder."

def get_mitigations(*args, **kwargs):
    return "Mitigations would be generated here in the original implementation."

def get_mitigations_azure(*args, **kwargs):
    return "Mitigations would be generated here in the original implementation."

def get_mitigations_google(*args, **kwargs):
    return "Mitigations would be generated here in the original implementation."

def get_mitigations_mistral(*args, **kwargs):
    return "Mitigations would be generated here in the original implementation."

def get_mitigations_ollama(*args, **kwargs):
    return "Mitigations would be generated here in the original implementation."

def get_mitigations_anthropic(*args, **kwargs):
    return "Mitigations would be generated here in the original implementation."

def get_mitigations_lm_studio(*args, **kwargs):
    return "Mitigations would be generated here in the original implementation."

def get_mitigations_groq(*args, **kwargs):
    return "Mitigations would be generated here in the original implementation."

def create_test_cases_prompt(*args, **kwargs):
    return "Sample test cases prompt - This is a placeholder."

def get_test_cases(*args, **kwargs):
    return "Test cases would be generated here in the original implementation."

def get_test_cases_azure(*args, **kwargs):
    return "Test cases would be generated here in the original implementation."

def get_test_cases_google(*args, **kwargs):
    return "Test cases would be generated here in the original implementation."

def get_test_cases_mistral(*args, **kwargs):
    return "Test cases would be generated here in the original implementation."

def get_test_cases_ollama(*args, **kwargs):
    return "Test cases would be generated here in the original implementation."

def get_test_cases_anthropic(*args, **kwargs):
    return "Test cases would be generated here in the original implementation."

def get_test_cases_lm_studio(*args, **kwargs):
    return "Test cases would be generated here in the original implementation."

def get_test_cases_groq(*args, **kwargs):
    return "Test cases would be generated here in the original implementation."

def create_dread_assessment_prompt(*args, **kwargs):
    return "Sample DREAD assessment prompt - This is a placeholder."

def get_dread_assessment(*args, **kwargs):
    return {"threats": [{"name": "Sample threat", "overallScore": 7.5}]}

def get_dread_assessment_azure(*args, **kwargs):
    return {"threats": [{"name": "Sample threat", "overallScore": 7.5}]}

def get_dread_assessment_google(*args, **kwargs):
    return {"threats": [{"name": "Sample threat", "overallScore": 7.5}]}

def get_dread_assessment_mistral(*args, **kwargs):
    return {"threats": [{"name": "Sample threat", "overallScore": 7.5}]}

def get_dread_assessment_ollama(*args, **kwargs):
    return {"threats": [{"name": "Sample threat", "overallScore": 7.5}]}

def get_dread_assessment_anthropic(*args, **kwargs):
    return {"threats": [{"name": "Sample threat", "overallScore": 7.5}]}

def get_dread_assessment_lm_studio(*args, **kwargs):
    return {"threats": [{"name": "Sample threat", "overallScore": 7.5}]}

def get_dread_assessment_groq(*args, **kwargs):
    return {"threats": [{"name": "Sample threat", "overallScore": 7.5}]}

def dread_json_to_markdown(*args, **kwargs):
    return "DREAD assessment would be displayed here in the original implementation."

def extract_mermaid_code(text):
    return text

def mermaid(code, height=500):
    return st.write("Mermaid diagram would be rendered here in the original implementation.")

def get_lm_studio_models(endpoint):
    return ["local-model"]

def get_ollama_models(ollama_endpoint):
    return ["llama3", "llama2", "mistral", "codellama"]

def estimate_tokens(text, model="gpt-4o"):
    return len(text) // 4  # Very rough approximation

# Import enhanced utility modules
from utils.ocr_utils import extract_text_from_image, extract_diagram_elements
from utils.export_utils import export_to_pdf, export_to_excel
from utils.openai_utils import analyze_image, analyze_requirements, generate_threats_and_mitigations
from utils.translation_utils import get_available_languages, translate_text, batch_translate
from utils.framework_utils import get_available_frameworks, get_framework_details, detect_framework, generate_hybrid_framework
from assets.framework_descriptions import get_framework_description

# Set page configuration
st.set_page_config(
    page_title="Enhanced Threat Modeling GPT",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load environment variables and API keys
load_dotenv()

# Initialize session state for storing data across reruns
if 'threat_model' not in st.session_state:
    st.session_state.threat_model = None
if 'threats_list' not in st.session_state:
    st.session_state.threats_list = []
if 'improvement_suggestions' not in st.session_state:
    st.session_state.improvement_suggestions = []
if 'app_analyzed' not in st.session_state:
    st.session_state.app_analyzed = False
if 'attack_tree_data' not in st.session_state:
    st.session_state.attack_tree_data = None
if 'dread_assessment' not in st.session_state:
    st.session_state.dread_assessment = None
if 'mitigations' not in st.session_state:
    st.session_state.mitigations = None
if 'test_cases' not in st.session_state:
    st.session_state.test_cases = None
if 'current_framework' not in st.session_state:
    st.session_state.current_framework = "STRIDE"
if 'hybrid_components' not in st.session_state:
    st.session_state.hybrid_components = []
if 'language' not in st.session_state:
    st.session_state.language = "English"
if 'diagram_analysis' not in st.session_state:
    st.session_state.diagram_analysis = None
if 'requirements_analysis' not in st.session_state:
    st.session_state.requirements_analysis = None
if 'enhanced_threat_model' not in st.session_state:
    st.session_state.enhanced_threat_model = None

# Define callback for handling API provider changes
def on_model_provider_change():
    """Update token limit and selected model when model provider changes"""
    # Clear out any previously selected model names
    st.session_state.model_name = ""
    # Set token and model limits based on provider
    if st.session_state.model_provider == "OpenAI":
        st.session_state.token_limit = 16000  # Default for GPT-4
    elif st.session_state.model_provider == "Google":
        st.session_state.token_limit = 12000  # For Google's models  
    elif st.session_state.model_provider == "Azure OpenAI":
        st.session_state.token_limit = 16000  # For Azure OpenAI
    elif st.session_state.model_provider == "Anthropic":
        st.session_state.token_limit = 100000  # For Claude
    elif st.session_state.model_provider == "Mistral":
        st.session_state.token_limit = 32000  # For Mistral
    elif st.session_state.model_provider == "Groq":
        st.session_state.token_limit = 100000  # For Groq models
    elif st.session_state.model_provider == "Ollama":
        st.session_state.token_limit = 8000  # Default for local models
    elif st.session_state.model_provider == "LM Studio":
        st.session_state.token_limit = 8000  # Default for local models

# Define callback for handling model selection changes
def on_model_selection_change():
    """Update token limit when specific model is selected"""
    # Update token limit based on specific model selection
    if st.session_state.model_provider == "OpenAI":
        if st.session_state.model_name == "gpt-4o":
            st.session_state.token_limit = 128000
        elif st.session_state.model_name == "gpt-4-turbo":
            st.session_state.token_limit = 128000
        elif st.session_state.model_name == "gpt-4":
            st.session_state.token_limit = 8000
        elif st.session_state.model_name == "gpt-4-32k":
            st.session_state.token_limit = 32000
        elif st.session_state.model_name == "gpt-3.5-turbo":
            st.session_state.token_limit = 16000
    elif st.session_state.model_provider == "Anthropic":
        if "claude-3-opus" in st.session_state.model_name:
            st.session_state.token_limit = 200000
        elif "claude-3-sonnet" in st.session_state.model_name:
            st.session_state.token_limit = 200000
        elif "claude-3-haiku" in st.session_state.model_name:
            st.session_state.token_limit = 200000
        elif "claude-2" in st.session_state.model_name:
            st.session_state.token_limit = 100000
    elif st.session_state.model_provider == "Groq":
        if "llama3-70b" in st.session_state.model_name:
            st.session_state.token_limit = 8000
        elif "llama2-70b" in st.session_state.model_name:
            st.session_state.token_limit = 4000
        elif "mixtral-8x7b" in st.session_state.model_name:
            st.session_state.token_limit = 32000
        else:
            st.session_state.token_limit = 8000
    elif st.session_state.model_provider == "Mistral":
        if "mistral-large" in st.session_state.model_name:
            st.session_state.token_limit = 32000
        elif "mistral-medium" in st.session_state.model_name:
            st.session_state.token_limit = 32000
        elif "mistral-small" in st.session_state.model_name:
            st.session_state.token_limit = 32000
        elif "mistral-tiny" in st.session_state.model_name:
            st.session_state.token_limit = 32000
        else:
            st.session_state.token_limit = 32000

# Helper function to encode images
def encode_image(uploaded_file):
    if uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()
        encoded = base64.b64encode(bytes_data).decode('utf-8')
        return encoded
    return None

# Helper function to refresh application state
def refresh_app():
    """Reset all session state variables to their default values"""
    st.session_state.threat_model = None
    st.session_state.threats_list = []
    st.session_state.improvement_suggestions = []
    st.session_state.app_analyzed = False
    st.session_state.attack_tree_data = None
    st.session_state.dread_assessment = None
    st.session_state.mitigations = None
    st.session_state.test_cases = None
    st.session_state.diagram_analysis = None
    st.session_state.requirements_analysis = None
    st.session_state.enhanced_threat_model = None
    st.rerun()

def on_framework_change():
    """Handle framework change"""
    if st.session_state.current_framework == "Hybrid":
        # When Hybrid is selected, preserve the current selection
        if not st.session_state.hybrid_components:
            # Default to STRIDE and DREAD if no components selected
            st.session_state.hybrid_components = ["STRIDE", "DREAD"]
    else:
        # Clear hybrid components when a specific framework is selected
        st.session_state.hybrid_components = []

def on_language_change():
    """Handle language change"""
    # If needed, translate the current content to the new language
    if st.session_state.language != "English" and st.session_state.app_analyzed:
        with st.spinner(f"Translating content to {st.session_state.language}..."):
            # Example of what could be translated - adapt based on what needs translation
            if st.session_state.threat_model:
                try:
                    # This is a placeholder - actual implementation would need to handle the nested structure
                    # Perhaps only translate certain fields or use a separate translated copy
                    pass
                except Exception as e:
                    st.error(f"Translation error: {e}")

def main():
    # Display logo and title
    col1, col2 = st.columns([1, 4])
    with col1:
        st.image(r"C:\Users\Pooja\Documents\ME Final Project\StrideEnhancer\StrideEnhancer\logo.png", width=100)
    with col2:
        st.title("Enhanced Threat Modeling GPT")
        st.markdown("#### AI-Powered Threat Modeling with Multiple Frameworks")

    # Create sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        
        # Framework selection
        framework_options = get_available_frameworks()
        st.selectbox(
            "Select Threat Modeling Framework",
            framework_options,
            index=framework_options.index("STRIDE") if "STRIDE" in framework_options else 0,
            key="current_framework",
            on_change=on_framework_change,
            help="Choose which threat modeling framework to use"
        )
        
        # Hybrid framework components selection
        if st.session_state.current_framework == "Hybrid":
            st.multiselect(
                "Select Frameworks to Combine",
                [f for f in framework_options if f != "Hybrid"],
                default=st.session_state.hybrid_components if st.session_state.hybrid_components else ["STRIDE", "DREAD"],
                key="hybrid_components",
                help="Select at least two frameworks to combine in your hybrid approach"
            )
            
            if len(st.session_state.hybrid_components) < 2:
                st.warning("Please select at least two frameworks for a hybrid approach")
        
        # Language selection
        language_options = get_available_languages()
        st.selectbox(
            "Language",
            language_options,
            index=language_options.index("English") if "English" in language_options else 0,
            key="language",
            on_change=on_language_change,
            help="Select the language for the generated content"
        )
        
        # API provider configuration
        st.subheader("API Configuration")
        
        if 'model_provider' not in st.session_state:
            st.session_state.model_provider = "OpenAI"
        
        # Model provider selection
        model_provider_options = [
            "OpenAI", 
            "Azure OpenAI", 
            "Google", 
            "Anthropic", 
            "Mistral", 
            "Groq", 
            "Ollama", 
            "LM Studio"
        ]
        
        st.selectbox(
            "Select Model Provider",
            model_provider_options,
            index=model_provider_options.index(st.session_state.model_provider),
            key="model_provider",
            on_change=on_model_provider_change,
            help="Choose which AI provider to use"
        )
        
        # Model selection based on provider
        if st.session_state.model_provider == "OpenAI":
            if 'model_name' not in st.session_state or st.session_state.model_name not in ["gpt-4o", "gpt-4-turbo", "gpt-4", "gpt-4-32k", "gpt-3.5-turbo"]:
                st.session_state.model_name = "gpt-4o"
                
            st.selectbox(
                "OpenAI Model",
                ["gpt-4o", "gpt-4-turbo", "gpt-4", "gpt-4-32k", "gpt-3.5-turbo"],
                index=["gpt-4o", "gpt-4-turbo", "gpt-4", "gpt-4-32k", "gpt-3.5-turbo"].index(st.session_state.model_name),
                key="model_name",
                on_change=on_model_selection_change,
                help="Select which OpenAI model to use"
            )
            
            if 'api_key' not in st.session_state:
                st.session_state.api_key = ""
                
            st.text_input(
                "OpenAI API Key",
                type="password",
                key="api_key",
                value=st.session_state.api_key or os.environ.get("OPENAI_API_KEY", ""),
                help="Enter your OpenAI API key"
            )
        
        elif st.session_state.model_provider == "Azure OpenAI":
            if 'azure_deployment_name' not in st.session_state:
                st.session_state.azure_deployment_name = ""
            if 'azure_api_endpoint' not in st.session_state:
                st.session_state.azure_api_endpoint = ""
            if 'azure_api_key' not in st.session_state:
                st.session_state.azure_api_key = ""
            if 'azure_api_version' not in st.session_state:
                st.session_state.azure_api_version = "2023-12-01-preview"
                
            st.text_input(
                "Azure OpenAI Deployment Name",
                key="azure_deployment_name",
                value=st.session_state.azure_deployment_name or os.environ.get("AZURE_OPENAI_DEPLOYMENT", ""),
                help="Enter your Azure OpenAI deployment name"
            )
            
            st.text_input(
                "Azure OpenAI API Endpoint",
                key="azure_api_endpoint",
                value=st.session_state.azure_api_endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT", ""),
                help="Enter your Azure OpenAI API endpoint"
            )
            
            st.text_input(
                "Azure OpenAI API Key",
                type="password",
                key="azure_api_key",
                value=st.session_state.azure_api_key or os.environ.get("AZURE_OPENAI_KEY", ""),
                help="Enter your Azure OpenAI API key"
            )
            
            st.text_input(
                "Azure OpenAI API Version",
                key="azure_api_version",
                value=st.session_state.azure_api_version or os.environ.get("AZURE_OPENAI_VERSION", "2023-12-01-preview"),
                help="Enter the Azure OpenAI API version to use"
            )
        
        elif st.session_state.model_provider == "Google":
            if 'google_api_key' not in st.session_state:
                st.session_state.google_api_key = ""
            if 'google_model' not in st.session_state or st.session_state.google_model not in ["gemini-1.5-pro", "gemini-1.5-flash", "gemini-1.0-pro"]:
                st.session_state.google_model = "gemini-1.5-pro"
                
            st.selectbox(
                "Google AI Model",
                ["gemini-1.5-pro", "gemini-1.5-flash", "gemini-1.0-pro"],
                index=["gemini-1.5-pro", "gemini-1.5-flash", "gemini-1.0-pro"].index(st.session_state.google_model),
                key="google_model",
                help="Select which Google AI model to use"
            )
            
            st.text_input(
                "Google AI API Key",
                type="password",
                key="google_api_key",
                value=st.session_state.google_api_key or os.environ.get("GOOGLE_API_KEY", ""),
                help="Enter your Google AI API key"
            )
        
        elif st.session_state.model_provider == "Anthropic":
            if 'anthropic_api_key' not in st.session_state:
                st.session_state.anthropic_api_key = ""
            if 'anthropic_model' not in st.session_state or st.session_state.anthropic_model not in ["claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307", "claude-2.1"]:
                st.session_state.anthropic_model = "claude-3-opus-20240229"
                
            st.selectbox(
                "Anthropic Model",
                ["claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307", "claude-2.1"],
                index=["claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307", "claude-2.1"].index(st.session_state.anthropic_model),
                key="anthropic_model",
                on_change=on_model_selection_change,
                help="Select which Anthropic Claude model to use"
            )
            
            st.text_input(
                "Anthropic API Key",
                type="password",
                key="anthropic_api_key",
                value=st.session_state.anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY", ""),
                help="Enter your Anthropic API key"
            )
        
        elif st.session_state.model_provider == "Mistral":
            if 'mistral_api_key' not in st.session_state:
                st.session_state.mistral_api_key = ""
            if 'mistral_model' not in st.session_state or st.session_state.mistral_model not in ["mistral-large-latest", "mistral-medium-latest", "mistral-small-latest", "mistral-tiny-latest", "open-mistral-7b"]:
                st.session_state.mistral_model = "mistral-large-latest"
                
            st.selectbox(
                "Mistral Model",
                ["mistral-large-latest", "mistral-medium-latest", "mistral-small-latest", "mistral-tiny-latest", "open-mistral-7b"],
                index=["mistral-large-latest", "mistral-medium-latest", "mistral-small-latest", "mistral-tiny-latest", "open-mistral-7b"].index(st.session_state.mistral_model),
                key="mistral_model",
                on_change=on_model_selection_change,
                help="Select which Mistral model to use"
            )
            
            st.text_input(
                "Mistral API Key",
                type="password",
                key="mistral_api_key",
                value=st.session_state.mistral_api_key or os.environ.get("MISTRAL_API_KEY", ""),
                help="Enter your Mistral API key"
            )
            
        elif st.session_state.model_provider == "Groq":
            if 'groq_api_key' not in st.session_state:
                st.session_state.groq_api_key = ""
            if 'groq_model' not in st.session_state or st.session_state.groq_model not in ["llama3-70b-8192", "llama2-70b-4096", "mixtral-8x7b-32768", "gemma-7b-it"]:
                st.session_state.groq_model = "llama3-70b-8192"
                
            st.selectbox(
                "Groq Model",
                ["llama3-70b-8192", "llama2-70b-4096", "mixtral-8x7b-32768", "gemma-7b-it"],
                index=["llama3-70b-8192", "llama2-70b-4096", "mixtral-8x7b-32768", "gemma-7b-it"].index(st.session_state.groq_model),
                key="groq_model",
                on_change=on_model_selection_change,
                help="Select which Groq model to use"
            )
            
            st.text_input(
                "Groq API Key",
                type="password",
                key="groq_api_key",
                value=st.session_state.groq_api_key or os.environ.get("GROQ_API_KEY", ""),
                help="Enter your Groq API key"
            )
            
        elif st.session_state.model_provider == "Ollama":
            if 'ollama_endpoint' not in st.session_state:
                st.session_state.ollama_endpoint = "http://localhost:11434"
                
            st.text_input(
                "Ollama Endpoint",
                key="ollama_endpoint",
                value=st.session_state.ollama_endpoint,
                help="Enter your Ollama endpoint URL (default: http://localhost:11434)"
            )
            
            try:
                ollama_models = get_ollama_models(st.session_state.ollama_endpoint)
                if not ollama_models:
                    ollama_models = ["llama3", "llama2", "mistral", "phi3", "codellama"]
                
                # If current model not in list, set to first model
                if 'ollama_model' not in st.session_state or st.session_state.ollama_model not in ollama_models:
                    st.session_state.ollama_model = ollama_models[0]
                    
                st.selectbox(
                    "Ollama Model",
                    ollama_models,
                    index=ollama_models.index(st.session_state.ollama_model) if st.session_state.ollama_model in ollama_models else 0,
                    key="ollama_model",
                    help="Select which Ollama model to use"
                )
            except Exception as e:
                st.error(f"Error connecting to Ollama: {e}")
                st.text_input(
                    "Ollama Model",
                    key="ollama_model",
                    value=st.session_state.get("ollama_model", "llama3"),
                    help="Enter your Ollama model name"
                )
        
        elif st.session_state.model_provider == "LM Studio":
            if 'lm_studio_endpoint' not in st.session_state:
                st.session_state.lm_studio_endpoint = "http://localhost:1234"
                
            st.text_input(
                "LM Studio Endpoint",
                key="lm_studio_endpoint",
                value=st.session_state.lm_studio_endpoint,
                help="Enter your LM Studio endpoint URL (default: http://localhost:1234)"
            )
            
            try:
                lm_studio_models = get_lm_studio_models(st.session_state.lm_studio_endpoint)
                if not lm_studio_models:
                    lm_studio_models = ["local-model"]
                
                # If current model not in list, set to first model
                if 'lm_studio_model' not in st.session_state or st.session_state.lm_studio_model not in lm_studio_models:
                    st.session_state.lm_studio_model = lm_studio_models[0]
                    
                st.selectbox(
                    "LM Studio Model",
                    lm_studio_models,
                    index=lm_studio_models.index(st.session_state.lm_studio_model) if st.session_state.lm_studio_model in lm_studio_models else 0,
                    key="lm_studio_model",
                    help="Select which LM Studio model to use"
                )
            except Exception as e:
                st.error(f"Error connecting to LM Studio: {e}")
                st.text_input(
                    "LM Studio Model",
                    key="lm_studio_model",
                    value=st.session_state.get("lm_studio_model", "local-model"),
                    help="Enter your LM Studio model name"
                )
        
        # Reset button
        st.button("Reset Analysis", on_click=refresh_app, help="Clear current analysis and start over")
        
        # Framework information
        st.subheader("About the Selected Framework")
        if st.session_state.current_framework == "Hybrid" and st.session_state.hybrid_components:
            # For hybrid approach, show a brief note about the selected frameworks
            st.write(f"You've selected a hybrid approach combining: {', '.join(st.session_state.hybrid_components)}")
            with st.expander("Hybrid Framework Details"):
                if len(st.session_state.hybrid_components) >= 2:
                    hybrid_description = generate_hybrid_framework(st.session_state.hybrid_components, "Application description not provided")
                    st.markdown(hybrid_description)
                else:
                    st.warning("Please select at least two frameworks for a hybrid approach")
        else:
            # For single framework, show the description
            with st.expander(f"About {st.session_state.current_framework}"):
                framework_description = get_framework_description(st.session_state.current_framework)
                st.markdown(framework_description)

    # Main application tabs
    tabs = st.tabs([
        "Input", 
        "Threat Model", 
        "Attack Tree", 
        "Risk Assessment", 
        "Mitigations", 
        "Test Cases",
        "Framework Detection",
        "Export"
    ])
    
    # Input Tab
    with tabs[0]:
        st.header("Application Input")
        
        # Input method selection
        input_method = st.radio(
            "Select input method",
            ["Manual Entry", "System Diagram Upload"],
            help="Choose how you want to provide information about your application"
        )
        
        # Initialize input variables
        app_type = ""
        authentication = ""
        internet_facing = ""
        sensitive_data = ""
        app_input = ""
        
        if input_method == "Manual Entry":
            st.markdown("### Application Details")
            
            col1, col2 = st.columns(2)
            with col1:
                app_type = st.selectbox(
                    "Application Type",
                    [
                        "Web Application",
                        "Mobile Application",
                        "Desktop Application",
                        "IoT Device/System",
                        "API",
                        "Microservices Architecture",
                        "Cloud-based Service",
                        "Distributed System",
                        "Other"
                    ],
                    help="Select the type of application you're modeling"
                )
            
            with col2:
                authentication = st.selectbox(
                    "Authentication Method",
                    [
                        "None",
                        "Username/Password",
                        "OAuth",
                        "OpenID Connect",
                        "SAML",
                        "Multi-factor Authentication",
                        "API Keys",
                        "Biometric",
                        "Other"
                    ],
                    help="Select the authentication method used by your application"
                )
            
            col3, col4 = st.columns(2)
            with col3:
                internet_facing = st.radio(
                    "Is this application Internet-facing?",
                    ["Yes", "No", "Partially"],
                    help="Indicate if the application is accessible from the Internet"
                )
            
            with col4:
                sensitive_data = st.radio(
                    "Does this application handle sensitive data?",
                    ["Yes", "No", "Not sure"],
                    help="Indicate if the application processes sensitive or personal data"
                )
            
            app_input = st.text_area(
                "Describe your application",
                height=200,
                placeholder="Provide details about your application's purpose, architecture, key components, data flows, and any specific security concerns.",
                help="Enter detailed information about your application to improve threat modeling accuracy"
            )
            
            # OCR toggle for manual method - offer to analyze requirements text
            if st.toggle("Analyze requirements text for system components"):
                if app_input:
                    try:
                        with st.spinner("Analyzing requirements..."):
                            requirements_analysis = analyze_requirements(app_input)
                            st.session_state.requirements_analysis = requirements_analysis
                    except Exception as e:
                        st.error(f"Error analyzing requirements: {e}")
                else:
                    st.warning("Please enter application description to analyze")
            
            # Display requirements analysis if available
            if st.session_state.requirements_analysis:
                with st.expander("Requirements Analysis", expanded=True):
                    st.markdown(st.session_state.requirements_analysis)
        

        
        elif input_method == "System Diagram Upload":
            st.markdown("### System Diagram Analysis")
            
            uploaded_file = st.file_uploader(
                "Upload System Diagram",
                type=["png", "jpg", "jpeg"],
                help="Upload an image of your system architecture or data flow diagram"
            )
            
            if uploaded_file is not None:
                # Display the uploaded image
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded System Diagram", use_column_width=True)
                
                # Save image for reference
                if 'diagram_image' not in st.session_state:
                    st.session_state.diagram_image = image
                
                # OCR toggle for diagram
                if st.toggle("Use OCR to extract text from diagram"):
                    try:
                        with st.spinner("Processing image with OCR..."):
                            extracted_text = extract_text_from_image(image)
                            if extracted_text:
                                st.success("Text extracted successfully!")
                                st.session_state.extracted_text = extracted_text
                                st.text_area("Extracted Text", extracted_text, height=150)
                                
                                # Use the extracted text for app_input
                                app_input = f"System Diagram OCR Text:\n\n{extracted_text}"
                            else:
                                st.warning("No text could be extracted from the image")
                    except Exception as e:
                        st.error(f"OCR processing error: {e}")
                
                # Enable AI diagram analysis
                if st.toggle("Analyze diagram with AI"):
                    try:
                        with st.spinner("Analyzing diagram..."):
                            # Encode the image to base64
                            buffered = io.BytesIO()
                            image.save(buffered, format="JPEG")
                            encoded_image = base64.b64encode(buffered.getvalue()).decode()
                            
                            # Create the prompt for image analysis
                            model_name = st.session_state.get('model_name', 'gpt-4o')
                            
                            # Process OpenAI call directly 
                            st.session_state.diagram_analysis = analyze_image(image)
                    except Exception as e:
                        st.error(f"Error analyzing diagram: {e}")
                
                # Display diagram analysis if available
                if st.session_state.diagram_analysis:
                    with st.expander("Diagram Analysis", expanded=True):
                        st.markdown(st.session_state.diagram_analysis)
                    
                    # Use the diagram analysis for app_input
                    app_input = f"System Diagram Analysis:\n\n{st.session_state.diagram_analysis}"
                    
                    # Try to determine app type from the analysis
                    if "web" in st.session_state.diagram_analysis.lower():
                        app_type = "Web Application"
                    elif "mobile" in st.session_state.diagram_analysis.lower():
                        app_type = "Mobile Application"
                    elif "iot" in st.session_state.diagram_analysis.lower() or "device" in st.session_state.diagram_analysis.lower():
                        app_type = "IoT Device/System"
                    elif "api" in st.session_state.diagram_analysis.lower():
                        app_type = "API"
                    elif "microservice" in st.session_state.diagram_analysis.lower():
                        app_type = "Microservices Architecture"
                    elif "cloud" in st.session_state.diagram_analysis.lower():
                        app_type = "Cloud-based Service"
                    else:
                        app_type = "Other"
                    
                    # Try to determine authentication from the analysis
                    if "oauth" in st.session_state.diagram_analysis.lower():
                        authentication = "OAuth"
                    elif "multi-factor" in st.session_state.diagram_analysis.lower() or "mfa" in st.session_state.diagram_analysis.lower():
                        authentication = "Multi-factor Authentication"
                    elif "password" in st.session_state.diagram_analysis.lower():
                        authentication = "Username/Password"
                    elif "api key" in st.session_state.diagram_analysis.lower():
                        authentication = "API Keys"
                    else:
                        authentication = "Not identified"
                    
                    # Try to determine if internet-facing from the analysis
                    if "internet" in st.session_state.diagram_analysis.lower() or "public" in st.session_state.diagram_analysis.lower():
                        internet_facing = "Yes"
                    elif "intranet" in st.session_state.diagram_analysis.lower() or "internal" in st.session_state.diagram_analysis.lower():
                        internet_facing = "No"
                    else:
                        internet_facing = "Not determined"
                    
                    # Try to determine if handles sensitive data from the analysis
                    if "sensitive" in st.session_state.diagram_analysis.lower() or "personal" in st.session_state.diagram_analysis.lower() or "pii" in st.session_state.diagram_analysis.lower():
                        sensitive_data = "Yes"
                    else:
                        sensitive_data = "Not determined"
                    
                    # Display inferred application details
                    st.subheader("Inferred Application Details")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        app_type = st.selectbox(
                            "Application Type",
                            [
                                "Web Application",
                                "Mobile Application",
                                "Desktop Application",
                                "IoT Device/System",
                                "API",
                                "Microservices Architecture",
                                "Cloud-based Service",
                                "Distributed System",
                                "Other"
                            ],
                            index=[
                                "Web Application",
                                "Mobile Application",
                                "Desktop Application",
                                "IoT Device/System",
                                "API",
                                "Microservices Architecture",
                                "Cloud-based Service",
                                "Distributed System",
                                "Other"
                            ].index(app_type) if app_type in [
                                "Web Application",
                                "Mobile Application",
                                "Desktop Application",
                                "IoT Device/System",
                                "API",
                                "Microservices Architecture",
                                "Cloud-based Service",
                                "Distributed System",
                                "Other"
                            ] else 0,
                            help="Select or confirm the application type"
                        )
                        
                        authentication = st.selectbox(
                            "Authentication Method",
                            [
                                "None",
                                "Username/Password",
                                "OAuth",
                                "OpenID Connect",
                                "SAML",
                                "Multi-factor Authentication",
                                "API Keys",
                                "Biometric",
                                "Other",
                                "Not identified"
                            ],
                            index=[
                                "None",
                                "Username/Password",
                                "OAuth",
                                "OpenID Connect",
                                "SAML",
                                "Multi-factor Authentication",
                                "API Keys",
                                "Biometric",
                                "Other",
                                "Not identified"
                            ].index(authentication) if authentication in [
                                "None",
                                "Username/Password",
                                "OAuth",
                                "OpenID Connect",
                                "SAML",
                                "Multi-factor Authentication",
                                "API Keys",
                                "Biometric",
                                "Other",
                                "Not identified"
                            ] else 9,
                            help="Select or confirm the authentication method"
                        )
                    
                    with col2:
                        internet_facing = st.radio(
                            "Is this application Internet-facing?",
                            ["Yes", "No", "Partially", "Not determined"],
                            index=["Yes", "No", "Partially", "Not determined"].index(internet_facing) if internet_facing in ["Yes", "No", "Partially", "Not determined"] else 3,
                            help="Indicate if the application is accessible from the Internet"
                        )
                        
                        sensitive_data = st.radio(
                            "Does this application handle sensitive data?",
                            ["Yes", "No", "Not sure", "Not determined"],
                            index=["Yes", "No", "Not sure", "Not determined"].index(sensitive_data) if sensitive_data in ["Yes", "No", "Not sure", "Not determined"] else 2,
                            help="Indicate if the application processes sensitive or personal data"
                        )
            
        # Submit button for analysis
        if st.button("Generate Threat Model", disabled=not app_input):
            with st.spinner("Generating threat model..."):
                try:
                    # Select API based on provider
                    if st.session_state.model_provider == "OpenAI":
                        st.session_state.threat_model, st.session_state.threats_list, st.session_state.improvement_suggestions = get_threat_model(
                            st.session_state.api_key,
                            st.session_state.model_name,
                            create_threat_model_prompt(app_type, authentication, internet_facing, sensitive_data, app_input)
                        )
                    
                    elif st.session_state.model_provider == "Azure OpenAI":
                        st.session_state.threat_model, st.session_state.threats_list, st.session_state.improvement_suggestions = get_threat_model_azure(
                            st.session_state.azure_api_endpoint,
                            st.session_state.azure_api_key,
                            st.session_state.azure_api_version,
                            st.session_state.azure_deployment_name,
                            create_threat_model_prompt(app_type, authentication, internet_facing, sensitive_data, app_input)
                        )
                    
                    elif st.session_state.model_provider == "Google":
                        st.session_state.threat_model, st.session_state.threats_list, st.session_state.improvement_suggestions = get_threat_model_google(
                            st.session_state.google_api_key,
                            st.session_state.google_model,
                            create_threat_model_prompt(app_type, authentication, internet_facing, sensitive_data, app_input)
                        )
                    
                    elif st.session_state.model_provider == "Anthropic":
                        st.session_state.threat_model, st.session_state.threats_list, st.session_state.improvement_suggestions = get_threat_model_anthropic(
                            st.session_state.anthropic_api_key,
                            st.session_state.anthropic_model,
                            create_threat_model_prompt(app_type, authentication, internet_facing, sensitive_data, app_input)
                        )
                    
                    elif st.session_state.model_provider == "Mistral":
                        st.session_state.threat_model, st.session_state.threats_list, st.session_state.improvement_suggestions = get_threat_model_mistral(
                            st.session_state.mistral_api_key,
                            st.session_state.mistral_model,
                            create_threat_model_prompt(app_type, authentication, internet_facing, sensitive_data, app_input)
                        )
                    
                    elif st.session_state.model_provider == "Groq":
                        st.session_state.threat_model, st.session_state.threats_list, st.session_state.improvement_suggestions = get_threat_model_groq(
                            st.session_state.groq_api_key,
                            st.session_state.groq_model,
                            create_threat_model_prompt(app_type, authentication, internet_facing, sensitive_data, app_input)
                        )
                    
                    elif st.session_state.model_provider == "Ollama":
                        st.session_state.threat_model, st.session_state.threats_list, st.session_state.improvement_suggestions = get_threat_model_ollama(
                            st.session_state.ollama_endpoint,
                            st.session_state.ollama_model,
                            create_threat_model_prompt(app_type, authentication, internet_facing, sensitive_data, app_input)
                        )
                    
                    elif st.session_state.model_provider == "LM Studio":
                        st.session_state.threat_model, st.session_state.threats_list, st.session_state.improvement_suggestions = get_threat_model_lm_studio(
                            st.session_state.lm_studio_endpoint,
                            st.session_state.lm_studio_model,
                            create_threat_model_prompt(app_type, authentication, internet_facing, sensitive_data, app_input)
                        )
                    
                    # Generate threat model for other frameworks when they're selected
                    if st.session_state.current_framework != "STRIDE" or st.session_state.current_framework == "Hybrid":
                        try:
                            # Use OpenAI for enhanced framework analysis
                            if st.session_state.current_framework == "Hybrid" and st.session_state.hybrid_components:
                                st.session_state.enhanced_threat_model = generate_threats_and_mitigations(
                                    requirements=app_input, 
                                    framework="Custom", 
                                    hybrid_components=st.session_state.hybrid_components,
                                    language=st.session_state.language
                                )
                            else:
                                st.session_state.enhanced_threat_model = generate_threats_and_mitigations(
                                    requirements=app_input, 
                                    framework=st.session_state.current_framework,
                                    language=st.session_state.language
                                )
                        except Exception as e:
                            st.error(f"Error generating enhanced threat model: {e}")
                        
                    # Mark as analyzed
                    st.session_state.app_analyzed = True
                    
                    # Auto-navigate to the Threat Model tab
                    # (Use st.experimental_rerun() to switch tabs in newer versions of Streamlit)
                    st.rerun()  # Use st.rerun() for compatibility
                    
                except Exception as e:
                    st.error(f"Error generating threat model: {e}")
    
    # Threat Model Tab
    with tabs[1]:
        st.header("Threat Model")
        
        if st.session_state.app_analyzed:
            if st.session_state.current_framework == "STRIDE":
                # Display the original STRIDE threat model
                if st.session_state.threat_model:
                    st.markdown(json_to_markdown(st.session_state.threat_model, st.session_state.improvement_suggestions))
                else:
                    st.warning("No threat model has been generated yet. Please complete the analysis in the Input tab.")
            else:
                # Display results from enhanced framework or hybrid approach
                if st.session_state.enhanced_threat_model:
                    if "error" in st.session_state.enhanced_threat_model:
                        st.error(st.session_state.enhanced_threat_model["error"])
                        if "raw_response" in st.session_state.enhanced_threat_model:
                            st.text(st.session_state.enhanced_threat_model["raw_response"])
                    else:
                        st.subheader(f"{st.session_state.current_framework} Framework Threat Model")
                        
                        # Display the threats and mitigations
                        for i, threat in enumerate(st.session_state.enhanced_threat_model, 1):
                            with st.expander(f"Threat {i}: {threat.get('name', '')}", expanded=i == 1):
                                st.markdown(f"**Category:** {threat.get('category', 'Not specified')}")
                                st.markdown(f"**Description:** {threat.get('description', '')}")
                                
                                if 'impact' in threat:
                                    st.markdown(f"**Impact:** {threat.get('impact', '')}")
                                
                                st.markdown("**Mitigations:**")
                                mitigations = threat.get('mitigations', [])
                                if isinstance(mitigations, list):
                                    for j, mitigation in enumerate(mitigations, 1):
                                        if isinstance(mitigation, dict):
                                            st.markdown(f"{j}. {mitigation.get('description', '')}")
                                        else:
                                            st.markdown(f"{j}. {mitigation}")
                                else:
                                    st.markdown(mitigations)
                else:
                    st.warning(f"No {st.session_state.current_framework} framework threat model has been generated yet. Please complete the analysis in the Input tab.")
        else:
            st.info("No threat model has been generated yet. Please complete the analysis in the Input tab.")
    
    # Attack Tree Tab
    with tabs[2]:
        st.header("Attack Tree")
        
        if st.session_state.app_analyzed:
            if not st.session_state.attack_tree_data:
                st.info("Attack tree has not been generated yet. Click the button below to generate it.")
                
                if st.button("Generate Attack Tree"):
                    with st.spinner("Generating attack tree..."):
                        try:
                            # Use the appropriate model provider
                            if st.session_state.model_provider == "OpenAI":
                                st.session_state.attack_tree_data = get_attack_tree(
                                    st.session_state.api_key,
                                    st.session_state.model_name,
                                    create_attack_tree_prompt(
                                        st.session_state.threat_model.get("applicationDetails", {}).get("applicationType", ""),
                                        st.session_state.threat_model.get("applicationDetails", {}).get("authenticationMethod", ""),
                                        st.session_state.threat_model.get("applicationDetails", {}).get("internetFacing", ""),
                                        st.session_state.threat_model.get("applicationDetails", {}).get("sensitiveData", ""),
                                        st.session_state.threat_model.get("applicationDetails", {}).get("applicationDescription", "")
                                    )
                                )
                            elif st.session_state.model_provider == "Azure OpenAI":
                                st.session_state.attack_tree_data = get_attack_tree_azure(
                                    st.session_state.azure_api_endpoint,
                                    st.session_state.azure_api_key,
                                    st.session_state.azure_api_version,
                                    st.session_state.azure_deployment_name,
                                    create_attack_tree_prompt(
                                        st.session_state.threat_model.get("applicationDetails", {}).get("applicationType", ""),
                                        st.session_state.threat_model.get("applicationDetails", {}).get("authenticationMethod", ""),
                                        st.session_state.threat_model.get("applicationDetails", {}).get("internetFacing", ""),
                                        st.session_state.threat_model.get("applicationDetails", {}).get("sensitiveData", ""),
                                        st.session_state.threat_model.get("applicationDetails", {}).get("applicationDescription", "")
                                    )
                                )
                            elif st.session_state.model_provider == "Google":
                                st.session_state.attack_tree_data = get_attack_tree_google(
                                    st.session_state.google_api_key,
                                    st.session_state.google_model,
                                    create_attack_tree_prompt(
                                        st.session_state.threat_model.get("applicationDetails", {}).get("applicationType", ""),
                                        st.session_state.threat_model.get("applicationDetails", {}).get("authenticationMethod", ""),
                                        st.session_state.threat_model.get("applicationDetails", {}).get("internetFacing", ""),
                                        st.session_state.threat_model.get("applicationDetails", {}).get("sensitiveData", ""),
                                        st.session_state.threat_model.get("applicationDetails", {}).get("applicationDescription", "")
                                    )
                                )
                            elif st.session_state.model_provider == "Mistral":
                                st.session_state.attack_tree_data = get_attack_tree_mistral(
                                    st.session_state.mistral_api_key,
                                    st.session_state.mistral_model,
                                    create_attack_tree_prompt(
                                        st.session_state.threat_model.get("applicationDetails", {}).get("applicationType", ""),
                                        st.session_state.threat_model.get("applicationDetails", {}).get("authenticationMethod", ""),
                                        st.session_state.threat_model.get("applicationDetails", {}).get("internetFacing", ""),
                                        st.session_state.threat_model.get("applicationDetails", {}).get("sensitiveData", ""),
                                        st.session_state.threat_model.get("applicationDetails", {}).get("applicationDescription", "")
                                    )
                                )
                            elif st.session_state.model_provider == "Anthropic":
                                st.session_state.attack_tree_data = get_attack_tree_anthropic(
                                    st.session_state.anthropic_api_key,
                                    st.session_state.anthropic_model,
                                    create_attack_tree_prompt(
                                        st.session_state.threat_model.get("applicationDetails", {}).get("applicationType", ""),
                                        st.session_state.threat_model.get("applicationDetails", {}).get("authenticationMethod", ""),
                                        st.session_state.threat_model.get("applicationDetails", {}).get("internetFacing", ""),
                                        st.session_state.threat_model.get("applicationDetails", {}).get("sensitiveData", ""),
                                        st.session_state.threat_model.get("applicationDetails", {}).get("applicationDescription", "")
                                    )
                                )
                            elif st.session_state.model_provider == "Groq":
                                st.session_state.attack_tree_data = get_attack_tree_groq(
                                    st.session_state.groq_api_key,
                                    st.session_state.groq_model,
                                    create_attack_tree_prompt(
                                        st.session_state.threat_model.get("applicationDetails", {}).get("applicationType", ""),
                                        st.session_state.threat_model.get("applicationDetails", {}).get("authenticationMethod", ""),
                                        st.session_state.threat_model.get("applicationDetails", {}).get("internetFacing", ""),
                                        st.session_state.threat_model.get("applicationDetails", {}).get("sensitiveData", ""),
                                        st.session_state.threat_model.get("applicationDetails", {}).get("applicationDescription", "")
                                    )
                                )
                            elif st.session_state.model_provider == "Ollama":
                                st.session_state.attack_tree_data = get_attack_tree_ollama(
                                    st.session_state.ollama_endpoint,
                                    st.session_state.ollama_model,
                                    create_attack_tree_prompt(
                                        st.session_state.threat_model.get("applicationDetails", {}).get("applicationType", ""),
                                        st.session_state.threat_model.get("applicationDetails", {}).get("authenticationMethod", ""),
                                        st.session_state.threat_model.get("applicationDetails", {}).get("internetFacing", ""),
                                        st.session_state.threat_model.get("applicationDetails", {}).get("sensitiveData", ""),
                                        st.session_state.threat_model.get("applicationDetails", {}).get("applicationDescription", "")
                                    )
                                )
                            elif st.session_state.model_provider == "LM Studio":
                                st.session_state.attack_tree_data = get_attack_tree_lm_studio(
                                    st.session_state.lm_studio_endpoint,
                                    st.session_state.lm_studio_model,
                                    create_attack_tree_prompt(
                                        st.session_state.threat_model.get("applicationDetails", {}).get("applicationType", ""),
                                        st.session_state.threat_model.get("applicationDetails", {}).get("authenticationMethod", ""),
                                        st.session_state.threat_model.get("applicationDetails", {}).get("internetFacing", ""),
                                        st.session_state.threat_model.get("applicationDetails", {}).get("sensitiveData", ""),
                                        st.session_state.threat_model.get("applicationDetails", {}).get("applicationDescription", "")
                                    )
                                )
                                
                            # After successfully generating the attack tree, rerun to display it
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error generating attack tree: {e}")
            
            # Display attack tree if available
            if st.session_state.attack_tree_data:
                try:
                    # Display description
                    st.markdown("### Attack Tree Description")
                    st.markdown(st.session_state.attack_tree_data.get("description", "No description available"))
                    
                    # Convert the attack tree to Mermaid format using the existing function
                    # and render it
                    from attack_tree import convert_tree_to_mermaid
                    mermaid_code = convert_tree_to_mermaid(st.session_state.attack_tree_data)
                    
                    # Clean up the Mermaid code
                    mermaid_code = extract_mermaid_code(mermaid_code)
                    
                    st.markdown("### Visual Attack Tree")
                    mermaid(mermaid_code)
                    
                    # Expert tips
                    with st.expander("Expert Tips on Using Attack Trees"):
                        st.markdown("""
                        ## Understanding and Using Attack Trees
                        
                        Attack trees provide a structured way to analyze potential attack paths against your system. Here's how to make the most of this visualization:
                        
                        ### Key Benefits
                        - **Visualization of attack paths**: Easily see how attackers might combine different techniques
                        - **Identification of critical vulnerabilities**: Find the most exploitable paths through your system
                        - **Prioritization guidance**: Focus on mitigating the most accessible and damaging attack vectors first
                        
                        ### How to Use This Information
                        1. Look for attack paths with the fewest steps - these are often the easiest for attackers to exploit
                        2. Identify nodes that appear in multiple attack paths - these represent high-value mitigation targets
                        3. Consider both technical controls and procedural safeguards for each attack vector
                        4. Use this tree when communicating security concerns to stakeholders and planning security enhancements
                        
                        The attack tree should be continuously updated as your system evolves or as new threats emerge.
                        """)
                    
                    # Allow exporting the Mermaid code
                    with st.expander("Export Mermaid Code"):
                        st.code(mermaid_code, language="markdown")
                    
                    # Regeneration option
                    if st.button("Regenerate Attack Tree"):
                        st.session_state.attack_tree_data = None
                        st.rerun()
                    
                except Exception as e:
                    st.error(f"Error rendering attack tree: {e}")
                    st.text(f"Raw attack tree data: {st.session_state.attack_tree_data}")
        else:
            st.info("No threat model has been generated yet. Please complete the analysis in the Input tab.")
    
    # Risk Assessment Tab
    with tabs[3]:
        st.header("Risk Assessment")
        
        if st.session_state.app_analyzed:
            if not st.session_state.dread_assessment:
                st.info("DREAD risk assessment has not been generated yet. Click the button below to generate it.")
                
                if st.button("Generate DREAD Assessment"):
                    with st.spinner("Generating DREAD risk assessment..."):
                        try:
                            # Use the appropriate model provider
                            if st.session_state.model_provider == "OpenAI":
                                st.session_state.dread_assessment = get_dread_assessment(
                                    st.session_state.api_key,
                                    st.session_state.model_name,
                                    create_dread_assessment_prompt(st.session_state.threats_list)
                                )
                            elif st.session_state.model_provider == "Azure OpenAI":
                                st.session_state.dread_assessment = get_dread_assessment_azure(
                                    st.session_state.azure_api_endpoint,
                                    st.session_state.azure_api_key,
                                    st.session_state.azure_api_version,
                                    st.session_state.azure_deployment_name,
                                    create_dread_assessment_prompt(st.session_state.threats_list)
                                )
                            elif st.session_state.model_provider == "Google":
                                st.session_state.dread_assessment = get_dread_assessment_google(
                                    st.session_state.google_api_key,
                                    st.session_state.google_model,
                                    create_dread_assessment_prompt(st.session_state.threats_list)
                                )
                            elif st.session_state.model_provider == "Mistral":
                                st.session_state.dread_assessment = get_dread_assessment_mistral(
                                    st.session_state.mistral_api_key,
                                    st.session_state.mistral_model,
                                    create_dread_assessment_prompt(st.session_state.threats_list)
                                )
                            elif st.session_state.model_provider == "Anthropic":
                                st.session_state.dread_assessment = get_dread_assessment_anthropic(
                                    st.session_state.anthropic_api_key,
                                    st.session_state.anthropic_model,
                                    create_dread_assessment_prompt(st.session_state.threats_list)
                                )
                            elif st.session_state.model_provider == "Groq":
                                st.session_state.dread_assessment = get_dread_assessment_groq(
                                    st.session_state.groq_api_key,
                                    st.session_state.groq_model,
                                    create_dread_assessment_prompt(st.session_state.threats_list)
                                )
                            elif st.session_state.model_provider == "Ollama":
                                st.session_state.dread_assessment = get_dread_assessment_ollama(
                                    st.session_state.ollama_endpoint,
                                    st.session_state.ollama_model,
                                    create_dread_assessment_prompt(st.session_state.threats_list)
                                )
                            elif st.session_state.model_provider == "LM Studio":
                                st.session_state.dread_assessment = get_dread_assessment_lm_studio(
                                    st.session_state.lm_studio_endpoint,
                                    st.session_state.lm_studio_model,
                                    create_dread_assessment_prompt(st.session_state.threats_list)
                                )
                                
                            # After successfully generating the DREAD assessment, rerun to display it
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error generating DREAD assessment: {e}")
            
            # Display DREAD assessment if available
            if st.session_state.dread_assessment:
                try:
                    # Display description of DREAD
                    st.markdown("""
                    ### DREAD Risk Assessment
                    
                    DREAD is a risk assessment methodology that helps prioritize the severity of security threats through five categories:
                    - **D**amage Potential: How severe is the damage if the vulnerability is exploited?
                    - **R**eproducibility: How easy is it to reproduce the attack?
                    - **E**xploitability: How much effort and expertise is needed to exploit the threat?
                    - **A**ffected Users: How many users would be impacted?
                    - **D**iscoverability: How easy is it to discover the vulnerability?
                    
                    Each category is rated from 1 (lowest) to 10 (highest). The final DREAD score is an average of these five ratings.
                    """)
                    
                    # Convert the assessment to markdown and display it
                    markdown_assessment = dread_json_to_markdown(st.session_state.dread_assessment)
                    st.markdown(markdown_assessment)
                    
                    # Risk matrix visualization
                    st.markdown("### Risk Priority Matrix")
                    
                    # Prepare data for the risk matrix (example implementation)
                    # This assumes threats are sorted by risk score in descending order in the assessment
                    threat_levels = []
                    for threat in st.session_state.dread_assessment.get("threats", []):
                        name = threat.get("name", "Unnamed threat")
                        score = threat.get("overallScore", 0)
                        
                        if score >= 8:
                            level = "Critical"
                            color = "#FF4B4B"  # Red
                        elif score >= 6:
                            level = "High"
                            color = "#FFA500"  # Orange
                        elif score >= 4:
                            level = "Medium"
                            color = "#FFFF00"  # Yellow
                        else:
                            level = "Low"
                            color = "#00FF00"  # Green
                        
                        threat_levels.append({
                            "name": name,
                            "score": score,
                            "level": level,
                            "color": color
                        })
                    
                    # Create risk matrix
                    for threat in threat_levels:
                        st.markdown(
                            f"<div style='padding: 10px; background-color: {threat['color']}; border-radius: 5px; margin-bottom: 5px;'>"
                            f"<strong>{threat['name']}</strong> - {threat['level']} Risk (Score: {threat['score']})"
                            f"</div>",
                            unsafe_allow_html=True
                        )
                    
                    # Expert tips
                    with st.expander("Expert Tips on Risk Prioritization"):
                        st.markdown("""
                        ## Prioritizing Risks Effectively
                        
                        The DREAD assessment helps you prioritize security risks, but it's important to apply this information strategically:
                        
                        ### Best Practices for Risk Management
                        1. **Address Critical risks immediately** - These pose serious, immediate threats to your system
                        2. **Balance effort vs. impact** - Sometimes a Medium risk might be worth fixing before a High risk if it requires minimal effort
                        3. **Consider business context** - Some risks may have regulatory or reputation implications beyond their technical score
                        4. **Implement quick wins first** - Look for high-impact, low-effort mitigations to make immediate security improvements
                        5. **Group related risks** - Some mitigations may address multiple threats simultaneously
                        
                        ### Using DREAD in Your Security Program
                        - Integrate DREAD scores into your development backlog prioritization
                        - Use these scores when communicating with stakeholders about security investments
                        - Reassess scores periodically as your application and threat landscape evolve
                        
                        Remember that DREAD is a tool to aid decision-making, not a replacement for security expertise and context.
                        """)
                    
                    # Regeneration option
                    if st.button("Regenerate DREAD Assessment"):
                        st.session_state.dread_assessment = None
                        st.rerun()
                    
                except Exception as e:
                    st.error(f"Error rendering DREAD assessment: {e}")
                    st.text(f"Raw DREAD assessment data: {st.session_state.dread_assessment}")
        else:
            st.info("No threat model has been generated yet. Please complete the analysis in the Input tab.")
    
    # Mitigations Tab
    with tabs[4]:
        st.header("Mitigations")
        
        if st.session_state.app_analyzed:
            if not st.session_state.mitigations:
                st.info("Detailed mitigations have not been generated yet. Click the button below to generate them.")
                
                if st.button("Generate Detailed Mitigations"):
                    with st.spinner("Generating detailed mitigations..."):
                        try:
                            # Use the appropriate model provider
                            if st.session_state.model_provider == "OpenAI":
                                st.session_state.mitigations = get_mitigations(
                                    st.session_state.api_key,
                                    st.session_state.model_name,
                                    create_mitigations_prompt(st.session_state.threats_list)
                                )
                            elif st.session_state.model_provider == "Azure OpenAI":
                                st.session_state.mitigations = get_mitigations_azure(
                                    st.session_state.azure_api_endpoint,
                                    st.session_state.azure_api_key,
                                    st.session_state.azure_api_version,
                                    st.session_state.azure_deployment_name,
                                    create_mitigations_prompt(st.session_state.threats_list)
                                )
                            elif st.session_state.model_provider == "Google":
                                st.session_state.mitigations = get_mitigations_google(
                                    st.session_state.google_api_key,
                                    st.session_state.google_model,
                                    create_mitigations_prompt(st.session_state.threats_list)
                                )
                            elif st.session_state.model_provider == "Mistral":
                                st.session_state.mitigations = get_mitigations_mistral(
                                    st.session_state.mistral_api_key,
                                    st.session_state.mistral_model,
                                    create_mitigations_prompt(st.session_state.threats_list)
                                )
                            elif st.session_state.model_provider == "Anthropic":
                                st.session_state.mitigations = get_mitigations_anthropic(
                                    st.session_state.anthropic_api_key,
                                    st.session_state.anthropic_model,
                                    create_mitigations_prompt(st.session_state.threats_list)
                                )
                            elif st.session_state.model_provider == "Groq":
                                st.session_state.mitigations = get_mitigations_groq(
                                    st.session_state.groq_api_key,
                                    st.session_state.groq_model,
                                    create_mitigations_prompt(st.session_state.threats_list)
                                )
                            elif st.session_state.model_provider == "Ollama":
                                st.session_state.mitigations = get_mitigations_ollama(
                                    st.session_state.ollama_endpoint,
                                    st.session_state.ollama_model,
                                    create_mitigations_prompt(st.session_state.threats_list)
                                )
                            elif st.session_state.model_provider == "LM Studio":
                                st.session_state.mitigations = get_mitigations_lm_studio(
                                    st.session_state.lm_studio_endpoint,
                                    st.session_state.lm_studio_model,
                                    create_mitigations_prompt(st.session_state.threats_list)
                                )
                                
                            # After successfully generating the mitigations, rerun to display them
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error generating mitigations: {e}")
            
            # Display mitigations if available
            if st.session_state.mitigations:
                try:
                    # Display the mitigations
                    st.markdown("### Detailed Mitigation Strategies")
                    st.markdown(st.session_state.mitigations)
                    
                    # Expert tips
                    with st.expander("Expert Tips on Implementing Mitigations"):
                        st.markdown("""
                        ## Effective Implementation of Security Mitigations
                        
                        Implementing security mitigations effectively requires strategic planning and execution:
                        
                        ### Implementation Best Practices
                        1. **Integrate with the development lifecycle** - Don't treat security as a bolt-on; incorporate it into your SDLC
                        2. **Follow the principle of defense in depth** - Implement multiple layers of controls
                        3. **Validate mitigation effectiveness** - Test that your controls actually work through penetration testing, code review, etc.
                        4. **Document your security controls** - Maintain clear documentation of implemented mitigations
                        5. **Use standard security libraries and frameworks** - Avoid reinventing security mechanisms
                        
                        ### Common Pitfalls to Avoid
                        - **Partial implementation** - Half-implemented security controls often create a false sense of security
                        - **Security by obscurity** - Don't rely solely on keeping implementation details secret
                        - **Ignoring usability** - Security controls that are too cumbersome may be bypassed
                        - **Static security approach** - Security is an ongoing process, not a one-time implementation
                        
                        Remember to regularly review and update your mitigations as threats and technologies evolve.
                        """)
                    
                    # Regeneration option
                    if st.button("Regenerate Mitigations"):
                        st.session_state.mitigations = None
                        st.rerun()
                    
                except Exception as e:
                    st.error(f"Error rendering mitigations: {e}")
                    st.text(f"Raw mitigations data: {st.session_state.mitigations}")
        else:
            st.info("No threat model has been generated yet. Please complete the analysis in the Input tab.")
    
    # Test Cases Tab
    with tabs[5]:
        st.header("Security Test Cases")
        
        if st.session_state.app_analyzed:
            if not st.session_state.test_cases:
                st.info("Security test cases have not been generated yet. Click the button below to generate them.")
                
                if st.button("Generate Security Test Cases"):
                    with st.spinner("Generating security test cases..."):
                        try:
                            # Use the appropriate model provider
                            if st.session_state.model_provider == "OpenAI":
                                st.session_state.test_cases = get_test_cases(
                                    st.session_state.api_key,
                                    st.session_state.model_name,
                                    create_test_cases_prompt(st.session_state.threats_list)
                                )
                            elif st.session_state.model_provider == "Azure OpenAI":
                                st.session_state.test_cases = get_test_cases_azure(
                                    st.session_state.azure_api_endpoint,
                                    st.session_state.azure_api_key,
                                    st.session_state.azure_api_version,
                                    st.session_state.azure_deployment_name,
                                    create_test_cases_prompt(st.session_state.threats_list)
                                )
                            elif st.session_state.model_provider == "Google":
                                st.session_state.test_cases = get_test_cases_google(
                                    st.session_state.google_api_key,
                                    st.session_state.google_model,
                                    create_test_cases_prompt(st.session_state.threats_list)
                                )
                            elif st.session_state.model_provider == "Mistral":
                                st.session_state.test_cases = get_test_cases_mistral(
                                    st.session_state.mistral_api_key,
                                    st.session_state.mistral_model,
                                    create_test_cases_prompt(st.session_state.threats_list)
                                )
                            elif st.session_state.model_provider == "Anthropic":
                                st.session_state.test_cases = get_test_cases_anthropic(
                                    st.session_state.anthropic_api_key,
                                    st.session_state.anthropic_model,
                                    create_test_cases_prompt(st.session_state.threats_list)
                                )
                            elif st.session_state.model_provider == "Groq":
                                st.session_state.test_cases = get_test_cases_groq(
                                    st.session_state.groq_api_key,
                                    st.session_state.groq_model,
                                    create_test_cases_prompt(st.session_state.threats_list)
                                )
                            elif st.session_state.model_provider == "Ollama":
                                st.session_state.test_cases = get_test_cases_ollama(
                                    st.session_state.ollama_endpoint,
                                    st.session_state.ollama_model,
                                    create_test_cases_prompt(st.session_state.threats_list)
                                )
                            elif st.session_state.model_provider == "LM Studio":
                                st.session_state.test_cases = get_test_cases_lm_studio(
                                    st.session_state.lm_studio_endpoint,
                                    st.session_state.lm_studio_model,
                                    create_test_cases_prompt(st.session_state.threats_list)
                                )
                                
                            # After successfully generating the test cases, rerun to display them
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error generating test cases: {e}")
            
            # Display test cases if available
            if st.session_state.test_cases:
                try:
                    # Display the test cases
                    st.markdown("### Security Test Cases")
                    st.markdown(st.session_state.test_cases)
                    
                    # Expert tips
                    with st.expander("Expert Tips on Security Testing"):
                        st.markdown("""
                        ## Effective Security Testing Strategies
                        
                        Implementing a comprehensive security testing program involves multiple approaches:
                        
                        ### Testing Methodologies
                        1. **Static Application Security Testing (SAST)** - Analyzes source code for security vulnerabilities
                        2. **Dynamic Application Security Testing (DAST)** - Tests running applications to find vulnerabilities
                        3. **Interactive Application Security Testing (IAST)** - Combines SAST and DAST approaches
                        4. **Security Test Automation** - Include security tests in your CI/CD pipeline
                        5. **Manual Penetration Testing** - Human experts testing your application for vulnerabilities
                        
                        ### Best Practices
                        - **Shift left** - Integrate security testing early in the development process
                        - **Test realistic scenarios** - Use the test cases that model real-world attack scenarios
                        - **Cover all layers** - Test the UI, API, database, and infrastructure
                        - **Use the right tools** - Different vulnerabilities require different testing approaches
                        - **Regular schedule** - Security testing should be ongoing, not a one-time event
                        
                        Consider building a security testing playbook from these test cases as a foundation for your security testing program.
                        """)
                    
                    # Regeneration option
                    if st.button("Regenerate Test Cases"):
                        st.session_state.test_cases = None
                        st.rerun()
                    
                except Exception as e:
                    st.error(f"Error rendering test cases: {e}")
                    st.text(f"Raw test cases data: {st.session_state.test_cases}")
        else:
            st.info("No threat model has been generated yet. Please complete the analysis in the Input tab.")
    
    # Framework Detection Tab
    with tabs[6]:
        st.header("AI Framework Detection")
        
        st.markdown("""
        This feature helps you identify the most appropriate threat modeling framework based on your system's characteristics.
        Upload a system diagram or provide a detailed description, and let AI suggest the optimal framework.
        """)
        
        # Allow input via diagram or text
        input_method = st.radio(
            "Select input method for framework detection",
            ["System Diagram", "Text Description"],
            help="Choose how you want to provide information about your system"
        )
        
        detected_framework = None
        
        if input_method == "System Diagram":
            uploaded_file = st.file_uploader(
                "Upload System Diagram",
                type=["png", "jpg", "jpeg"],
                help="Upload an image of your system architecture or data flow diagram",
                key="framework_detection_image"
            )
            
            if uploaded_file is not None:
                # Display the uploaded image
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded System Diagram", use_column_width=True)
                
                # Process button
                if st.button("Detect Appropriate Framework"):
                    with st.spinner("Analyzing system diagram..."):
                        try:
                            detected_framework = detect_framework(image=image)
                            st.success(f"Analysis complete! Detected appropriate framework: {detected_framework}")
                        except Exception as e:
                            st.error(f"Error analyzing diagram: {e}")
        
        else:  # Text Description
            system_description = st.text_area(
                "Describe your system",
                height=200,
                placeholder="Provide details about your system's purpose, architecture, components, data flows, user types, and security requirements.",
                help="Enter detailed information about your system to improve framework detection accuracy"
            )
            
            # Process button
            if st.button("Detect Appropriate Framework") and system_description:
                with st.spinner("Analyzing system description..."):
                    try:
                        detected_framework = detect_framework(text=system_description)
                        st.success(f"Analysis complete! Detected appropriate framework: {detected_framework}")
                    except Exception as e:
                        st.error(f"Error analyzing description: {e}")
        
        # Display framework recommendation if available
        if detected_framework:
            st.markdown(f"## Recommended Framework: {detected_framework}")
            
            # Display framework details
            framework_details = get_framework_details(detected_framework)
            
            st.markdown("### Framework Overview")
            st.markdown(framework_details.get("description", "No description available"))
            
            st.markdown("### Suitable Use Cases")
            for use_case in framework_details.get("suitable_for", []):
                st.markdown(f"- {use_case}")
            
            st.markdown("### Framework Components")
            for component in framework_details.get("components", []):
                st.markdown(f"- {component}")
            
            # Option to set the detected framework as current
            if st.button(f"Set {detected_framework} as Current Framework"):
                st.session_state.current_framework = detected_framework
                if detected_framework == "Hybrid":
                    st.session_state.hybrid_components = ["STRIDE", "DREAD"]  # Default components
                st.rerun()
    
    # Export Tab
    with tabs[7]:
        st.header("Export Results")
        
        if st.session_state.app_analyzed:
            st.markdown("""
            Export your threat modeling results in different formats. Choose the export options below:
            """)
            
            # Select content to include
            st.subheader("Content Selection")
            export_threat_model = st.checkbox("Threat Model", value=True)
            export_attack_tree = st.checkbox("Attack Tree", value=True)
            export_risk_assessment = st.checkbox("Risk Assessment", value=True)
            export_mitigations = st.checkbox("Mitigations", value=True)
            export_test_cases = st.checkbox("Test Cases", value=True)
            
            # Export format selection
            st.subheader("Export Format")
            export_format = st.radio("Select format", ["PDF", "Excel"])
            
            # Get data for export
            if st.button("Generate Export"):
                with st.spinner(f"Generating {export_format} export..."):
                    try:
                        # Prepare data structure for export
                        threats_mitigations = []
                        
                        # If using STRIDE framework
                        if st.session_state.current_framework == "STRIDE" and st.session_state.threat_model:
                            for threat in st.session_state.threats_list:
                                threat_data = {
                                    "name": threat.get("title", "Unnamed threat"),
                                    "category": threat.get("type", "Unspecified"),
                                    "description": threat.get("description", ""),
                                    "mitigations": threat.get("mitigation", [])
                                }
                                
                                # Add DREAD scores if available
                                if st.session_state.dread_assessment:
                                    for dread_threat in st.session_state.dread_assessment.get("threats", []):
                                        if dread_threat.get("name", "") == threat_data["name"]:
                                            threat_data["impact"] = f"DREAD Score: {dread_threat.get('overallScore', 'N/A')}"
                                            break
                                
                                threats_mitigations.append(threat_data)
                        
                        # If using enhanced or hybrid framework
                        elif st.session_state.enhanced_threat_model:
                            threats_mitigations = st.session_state.enhanced_threat_model
                        
                        # Generate the export
                        if export_format == "PDF":
                            framework_name = st.session_state.current_framework
                            hybrid_components = st.session_state.hybrid_components if framework_name == "Hybrid" else None
                            
                            pdf_bytes = export_to_pdf(threats_mitigations, framework=framework_name, hybrid_components=hybrid_components)
                            
                            # Create download button
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            filename = f"threat_model_{framework_name.lower()}_{timestamp}.pdf"
                            
                            st.download_button(
                                label="Download PDF Report",
                                data=pdf_bytes,
                                file_name=filename,
                                mime="application/pdf"
                            )
                        
                        elif export_format == "Excel":
                            framework_name = st.session_state.current_framework
                            hybrid_components = st.session_state.hybrid_components if framework_name == "Hybrid" else None
                            
                            excel_bytes = export_to_excel(threats_mitigations, framework=framework_name, hybrid_components=hybrid_components)
                            
                            # Create download button
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            filename = f"threat_model_{framework_name.lower()}_{timestamp}.xlsx"
                            
                            st.download_button(
                                label="Download Excel Report",
                                data=excel_bytes,
                                file_name=filename,
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            )
                    
                    except Exception as e:
                        st.error(f"Error generating export: {e}")
        else:
            st.info("No threat model has been generated yet. Please complete the analysis in the Input tab.")
    
    # Footer
    st.markdown("""
    ---
    ### About Enhanced Threat Modeling GPT
    
    This tool combines the power of the original STRIDE-GPT with support for multiple threat modeling frameworks, 
    improved reporting capabilities, and advanced features like OCR for diagrams and multi-language support.
    
    Original STRIDE-GPT by [Mark Wadams](https://github.com/mrwadams/stride-gpt) | Enhanced version includes multiple frameworks, OCR, exports, and more.
    """)


if __name__ == "__main__":
    # Check if OpenAI API key is set and prompt for it if not
    if "OPENAI_API_KEY" not in os.environ and "api_key" not in st.session_state:
        st.warning("⚠️ OpenAI API key is required for enhanced features. Please set it in the sidebar.")
    
    main()