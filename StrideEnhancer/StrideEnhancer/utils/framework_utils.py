import os
import json
from openai import OpenAI
import io
import numpy as np

def get_available_frameworks():
    """
    Get a list of available threat modeling frameworks.
    
    Returns:
        List of framework names
    """
    return [
        "STRIDE", "DREAD", "PASTA", "CVSS", "OCTAVE", "TRIKE", "LINDDUN", 
        "NIST 800-30", "FAIR", "ATT&CK", "Hybrid", "Custom"
    ]

def get_framework_details(framework_name):
    """
    Get detailed information about a specific framework.
    
    Args:
        framework_name: Name of the framework
        
    Returns:
        Dictionary with framework details
    """
    frameworks = {
        "STRIDE": {
            "full_name": "STRIDE",
            "description": "Microsoft's threat modeling methodology focusing on six threat categories: Spoofing, Tampering, Repudiation, Information disclosure, Denial of service, and Elevation of privilege.",
            "categories": ["Spoofing", "Tampering", "Repudiation", "Information disclosure", "Denial of service", "Elevation of privilege"],
            "suitable_for": ["Web Applications", "APIs", "Microservices", "Desktop Applications"],
            "focus_areas": ["Security Design", "Architecture Review"]
        },
        "DREAD": {
            "full_name": "DREAD Risk Assessment Model",
            "description": "Methodology for calculating risk scores based on Damage potential, Reproducibility, Exploitability, Affected users, and Discoverability.",
            "categories": ["Damage potential", "Reproducibility", "Exploitability", "Affected users", "Discoverability"],
            "suitable_for": ["Risk Assessment", "Vulnerability Prioritization"],
            "focus_areas": ["Qualitative Risk Measurement"]
        },
        "PASTA": {
            "full_name": "Process for Attack Simulation and Threat Analysis",
            "description": "Risk-centric methodology focused on identifying and analyzing threats and vulnerabilities from an attacker's perspective.",
            "categories": ["Asset Analysis", "Business Impact", "Threat Identification", "Vulnerability Analysis", "Attack Modeling", "Risk Analysis"],
            "suitable_for": ["Complex Enterprise Systems", "Security Strategy Development"],
            "focus_areas": ["Attack Surface Mapping", "Attack Simulation"]
        },
        "CVSS": {
            "full_name": "Common Vulnerability Scoring System",
            "description": "Framework for assessing the severity of computer system security vulnerabilities based on base, temporal, and environmental metrics.",
            "categories": ["Base Metrics", "Temporal Metrics", "Environmental Metrics"],
            "suitable_for": ["Vulnerability Management", "Patch Prioritization"],
            "focus_areas": ["Standardized Vulnerability Assessment"]
        },
        "OCTAVE": {
            "full_name": "Operationally Critical Threat, Asset, and Vulnerability Evaluation",
            "description": "Risk-based strategic assessment and planning technique for security focusing on organizational risk practices.",
            "categories": ["Critical Assets", "Threats to Assets", "Vulnerabilities", "Risk Analysis", "Protection Strategy"],
            "suitable_for": ["Enterprise Risk Management", "Organizational Security Planning"],
            "focus_areas": ["Asset-based Threat Analysis"]
        },
        "TRIKE": {
            "full_name": "TRIKE",
            "description": "Methodology focused on a risk-based approach with an emphasis on satisfying stakeholder security requirements.",
            "categories": ["Asset Identification", "Requirements Analysis", "Privilege Analysis", "Threat Modeling", "Risk Analysis"],
            "suitable_for": ["Security Auditing", "Compliance Verification"],
            "focus_areas": ["Requirements-driven Security Analysis"]
        },
        "LINDDUN": {
            "full_name": "Linkability, Identifiability, Non-repudiation, Detectability, Disclosure, Unawareness, Non-compliance",
            "description": "Privacy-focused threat modeling methodology that helps identify privacy threats in software systems.",
            "categories": ["Linkability", "Identifiability", "Non-repudiation", "Detectability", "Disclosure of information", "Unawareness", "Non-compliance"],
            "suitable_for": ["Privacy-Sensitive Applications", "Personal Data Processing Systems"],
            "focus_areas": ["Privacy Threat Analysis"]
        },
        "NIST 800-30": {
            "full_name": "NIST Special Publication 800-30",
            "description": "U.S. government standard for conducting risk assessments of federal information systems and organizations.",
            "categories": ["Threat Identification", "Vulnerability Identification", "Impact Analysis", "Likelihood Determination", "Risk Determination"],
            "suitable_for": ["Government Systems", "Critical Infrastructure", "Compliance-Driven Organizations"],
            "focus_areas": ["Systematic Risk Assessment"]
        },
        "FAIR": {
            "full_name": "Factor Analysis of Information Risk",
            "description": "Framework for understanding, analyzing, and measuring information risk with a focus on quantitative risk assessment.",
            "categories": ["Asset Value", "Threat Event Frequency", "Vulnerability", "Loss Magnitude"],
            "suitable_for": ["Financial Risk Assessment", "Security Investment Decisions"],
            "focus_areas": ["Quantitative Risk Analysis"]
        },
        "ATT&CK": {
            "full_name": "MITRE ATT&CK Framework",
            "description": "Knowledge base of adversary tactics and techniques based on real-world observations, used for threat modeling and security analysis.",
            "categories": ["Initial Access", "Execution", "Persistence", "Privilege Escalation", "Defense Evasion", "Credential Access", "Discovery", "Lateral Movement", "Collection", "Command and Control", "Exfiltration", "Impact"],
            "suitable_for": ["Threat Intelligence", "Security Operations", "Red/Blue Team Exercises"],
            "focus_areas": ["Adversary Behavior Modeling"]
        },
        "Hybrid": {
            "full_name": "Hybrid Framework",
            "description": "Custom approach combining elements from multiple frameworks to address specific organizational needs and threat landscapes.",
            "categories": ["Customizable based on selected frameworks"],
            "suitable_for": ["Complex Systems", "Organizations with Unique Requirements"],
            "focus_areas": ["Comprehensive Threat Coverage"]
        },
        "Custom": {
            "full_name": "Custom Framework",
            "description": "User-defined approach for specialized threat modeling needs that don't fit existing frameworks.",
            "categories": ["User-defined"],
            "suitable_for": ["Specialized Applications", "Novel Technology Stacks"],
            "focus_areas": ["Tailored Security Analysis"]
        }
    }
    
    return frameworks.get(framework_name, {
        "full_name": framework_name,
        "description": "Custom framework details not available.",
        "categories": [],
        "suitable_for": [],
        "focus_areas": []
    })

def detect_framework(application_description):
    """
    Automatically detect the most suitable threat modeling framework based on the application description.
    
    Args:
        application_description: Description of the application to be threat modeled
        
    Returns:
        Tuple of (recommended_framework, confidence_score, explanation)
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    
    if not api_key:
        # Default recommendation if API key is not available
        return ("STRIDE", 0.7, "STRIDE is recommended as a general-purpose framework for most applications.")
    
    client = OpenAI(api_key=api_key)
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",  # the newest OpenAI model is "gpt-4o" which was released May 13, 2024
            messages=[
                {
                    "role": "system",
                    "content": "You are a cybersecurity expert specializing in threat modeling. "
                               "Based on the application description, recommend the most suitable threat modeling framework. "
                               "Consider the application type, complexity, security requirements, and industry context."
                },
                {
                    "role": "user",
                    "content": f"Recommend the most suitable threat modeling framework for the following application:\n\n"
                               f"{application_description}\n\n"
                               f"Choose from: STRIDE, DREAD, PASTA, CVSS, OCTAVE, TRIKE, LINDDUN, NIST 800-30, FAIR, ATT&CK. "
                               f"You can also suggest a Hybrid approach combining multiple frameworks if appropriate. "
                               f"Provide your recommendation, a confidence score (0.0-1.0), and a brief explanation."
                }
            ],
            response_format={"type": "json_object"},
            max_tokens=500
        )
        
        result = json.loads(response.choices[0].message.content)
        
        # Extract data from response
        recommended_framework = result.get("recommended_framework", "STRIDE")
        confidence_score = float(result.get("confidence_score", 0.7))
        explanation = result.get("explanation", "No explanation provided.")
        
        return (recommended_framework, confidence_score, explanation)
    
    except Exception as e:
        # Fallback to default recommendation
        return ("STRIDE", 0.7, f"STRIDE is recommended as a general-purpose framework. Error in framework detection: {str(e)}")

def generate_hybrid_framework(components, application_description):
    """
    Generate a custom hybrid framework combining elements from multiple frameworks.
    
    Args:
        components: List of framework names to include in the hybrid
        application_description: Description of the application to be threat modeled
        
    Returns:
        Dictionary with hybrid framework details
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    
    if not api_key:
        # Default hybrid if API key is not available
        return {
            "name": "Hybrid Framework",
            "description": f"Custom hybrid approach combining elements from: {', '.join(components)}",
            "categories": [],
            "methodology": "Manual hybrid approach combining selected frameworks.",
            "components": components
        }
    
    client = OpenAI(api_key=api_key)
    
    try:
        # Get details for each component framework
        component_details = {}
        for framework in components:
            component_details[framework] = get_framework_details(framework)
        
        # Generate hybrid framework
        response = client.chat.completions.create(
            model="gpt-4o",  # the newest OpenAI model is "gpt-4o" which was released May 13, 2024
            messages=[
                {
                    "role": "system",
                    "content": "You are a cybersecurity expert specializing in threat modeling. "
                               "Create a hybrid framework combining elements from multiple frameworks "
                               "tailored to the specific application."
                },
                {
                    "role": "user",
                    "content": f"Create a hybrid threat modeling framework combining elements from: {', '.join(components)}.\n\n"
                               f"Consider how these frameworks can complement each other for the following application:\n\n"
                               f"{application_description}\n\n"
                               f"Provide a structured approach including: name for the hybrid framework, description, "
                               f"combined threat categories, step-by-step methodology, and key assessment factors."
                }
            ],
            response_format={"type": "json_object"},
            max_tokens=1000
        )
        
        result = json.loads(response.choices[0].message.content)
        
        # Ensure the result has the necessary fields
        if not all(key in result for key in ["name", "description", "categories", "methodology"]):
            raise ValueError("Generated hybrid framework is missing required fields")
        
        # Add the component frameworks to the result
        result["components"] = components
        
        return result
    
    except Exception as e:
        # Fallback to default hybrid
        return {
            "name": "Hybrid Framework",
            "description": f"Custom hybrid approach combining elements from: {', '.join(components)}",
            "categories": [],
            "methodology": f"Manual hybrid approach combining selected frameworks. Error generating detailed methodology: {str(e)}",
            "components": components
        }