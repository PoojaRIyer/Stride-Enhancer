def get_framework_description(framework_name):
    """
    Get a detailed description of a threat modeling framework.
    
    Args:
        framework_name: Name of the framework
        
    Returns:
        Markdown-formatted description string
    """
    descriptions = {
        "STRIDE": """
## STRIDE Framework

**STRIDE** is a threat modeling methodology developed by Microsoft. It categorizes threats into six types:

1. **Spoofing**: Impersonating something or someone else.
2. **Tampering**: Modifying data or code.
3. **Repudiation**: Claiming to have not performed an action.
4. **Information Disclosure**: Exposing information to unauthorized individuals.
5. **Denial of Service**: Denying or degrading service to valid users.
6. **Elevation of Privilege**: Gaining unauthorized capabilities.

### When to Use STRIDE

STRIDE is most effective when:
- Analyzing individual system components
- Conducting a comprehensive security review
- Working with well-defined technical systems
- You need a systematic approach to threat discovery

### Methodology

1. Identify assets and create a system model
2. For each component, evaluate all six STRIDE categories
3. Document threats using a consistent format
4. Prioritize threats based on risk
5. Develop mitigation strategies

### Key Strengths

- Comprehensive coverage of technical threats
- Well-established with extensive industry adoption
- Structured and methodical approach
- Clear categorization that maps to security properties
""",
        "DREAD": """
## DREAD Risk Assessment Model

**DREAD** is a risk assessment methodology used to calculate risk scores for identified threats. It evaluates five risk factors:

1. **Damage potential**: How bad would an attack be?
2. **Reproducibility**: How easy is it to reproduce the attack?
3. **Exploitability**: How much effort is required to mount the attack?
4. **Affected users**: How many users would be affected?
5. **Discoverability**: How easy is it to discover the vulnerability?

### When to Use DREAD

DREAD is particularly useful when:
- You need to prioritize multiple identified threats
- Quantitative risk assessment is required
- Communicating risk to non-technical stakeholders
- Making risk-based decisions on mitigation priorities

### Methodology

1. Identify and document threats through other methods
2. Score each threat on the five DREAD factors (usually 0-10)
3. Calculate an overall risk score (average or weighted)
4. Prioritize mitigation based on risk scores
5. Re-assess scores after implementing controls

### Key Strengths

- Simple, intuitive scoring system
- Facilitates consistent threat prioritization
- Supports risk-based decision making
- Helps communicate relative importance of different threats
""",
        "PASTA": """
## Process for Attack Simulation and Threat Analysis (PASTA)

**PASTA** is a risk-centric threat modeling methodology that focuses on identifying and analyzing threats and vulnerabilities from an attacker's perspective. It consists of seven stages:

1. **Define Objectives**: Identify business objectives and security requirements
2. **Define Technical Scope**: Map the application architecture
3. **Application Decomposition**: Identify application roles, assets, entry points, and trust levels
4. **Threat Analysis**: Identify threats that apply to your application
5. **Vulnerability Analysis**: Map threats to specific vulnerabilities 
6. **Attack Modeling**: Create attack trees and identify attack vectors
7. **Risk Analysis and Mitigation**: Calculate risk scores and develop countermeasures

### When to Use PASTA

PASTA is most appropriate when:
- A comprehensive risk-based approach is needed
- Business context and attacker motivation are important
- You have the resources for a thorough analysis
- Alignment with business objectives is critical

### Key Strengths

- Combines business, application, and risk perspectives
- Thorough, multi-stage approach
- Produces detailed, evidence-based attack scenarios
- Integrates threat intelligence and attack patterns
- Links technical vulnerabilities to business impact
""",
        "CVSS": """
## Common Vulnerability Scoring System (CVSS)

**CVSS** is a framework for assessing the severity of computer system security vulnerabilities. It provides a way to capture the principal characteristics of a vulnerability and produce a numerical score reflecting its severity, which can then be translated into a qualitative representation (such as low, medium, high, and critical).

### CVSS Metrics

CVSS uses three metric groups:

1. **Base Metrics**: Intrinsic and fundamental characteristics of a vulnerability that are constant over time
   - Exploitability metrics: Attack Vector, Attack Complexity, Privileges Required, User Interaction
   - Impact metrics: Confidentiality, Integrity, Availability

2. **Temporal Metrics**: Characteristics that evolve over the lifetime of a vulnerability
   - Exploit Code Maturity, Remediation Level, Report Confidence

3. **Environmental Metrics**: Characteristics unique to a specific user's environment
   - Modified Base Metrics, Security Requirements

### When to Use CVSS

CVSS is particularly useful when:
- Assessing the severity of known vulnerabilities
- Prioritizing vulnerability remediation efforts
- Communicating vulnerability severity to stakeholders
- Standardizing vulnerability assessment across different systems

### Key Strengths

- Industry-standard scoring system
- Consistent, objective measurement of vulnerability severity
- Customizable to specific environments
- Facilitates clear communication about security risks
- Supports risk-based decision making for vulnerability management
""",
        "OCTAVE": """
## Operationally Critical Threat, Asset, and Vulnerability Evaluation (OCTAVE)

**OCTAVE** is a risk-based strategic assessment and planning technique for security. It focuses on organizational risk and strategic, practice-related issues. There are several versions of OCTAVE, including the original OCTAVE method, OCTAVE-S (for smaller organizations), and OCTAVE Allegro (streamlined approach).

### OCTAVE Methodology

OCTAVE Allegro (the most recent version) consists of eight steps across four phases:

1. **Establish Drivers**
   - Develop risk measurement criteria
   - Develop organizational information asset profile

2. **Profile Assets**
   - Identify information asset containers
   - Identify areas of concern

3. **Identify Threats**
   - Identify and document threats
   - Identify and analyze risks

4. **Identify and Mitigate Risks**
   - Select mitigation approach
   - Develop mitigation strategy

### When to Use OCTAVE

OCTAVE is most appropriate when:
- An organizational-level risk assessment is needed
- Asset-focused risk management is required
- You need a structured, workshop-based approach
- Both technical and non-technical staff need to be involved
- Self-directed risk management is preferred over consultant-driven approaches

### Key Strengths

- Flexible and adaptable to different organization sizes
- Balances operational risk, security practices, and technology
- Emphasizes organizational and procedural aspects of security
- Produces a clear, documented security strategy
- Can be conducted without extensive technical expertise
""",
        "TRIKE": """
## TRIKE Framework

**TRIKE** is a security audit methodology and tool that focuses on satisfying the security auditing requirements from a risk management perspective. It uses a requirement-centered approach to security auditing.

### TRIKE Methodology

TRIKE's approach includes:

1. **Requirements Modeling**: Define "acceptable" use of the system
   - Map actors to assets, defining acceptable level of access
   - Define rules for authorization and validation

2. **Privilege Modeling**: Create a data flow diagram to understand how the system works
   - Track privilege through the system
   - Identify privilege elevation

3. **Threat Modeling**: Identify threats by looking at each asset/actor assignment
   - Denial of service
   - Elevation of privilege
   - Information disclosure

4. **Risk Analysis**: Assign risk value to each threat using a risk formula
   - Probability × Impact = Risk

5. **Risk Mitigation**: Define and implement controls to address identified risks

### When to Use TRIKE

TRIKE is particularly useful when:
- A requirements-driven approach is preferred
- You need to align security analysis with system requirements
- You want to automate parts of the threat modeling process
- A structured, formal approach to threat classification is needed

### Key Strengths

- Systematic and repeatable methodology
- Requirements-focused approach
- Clear mapping of actors to assets
- Risk-based prioritization
- Tools available to automate parts of the process
""",
        "LINDDUN": """
## LINDDUN Privacy Threat Modeling Framework

**LINDDUN** is a privacy-focused threat modeling methodology that helps identify privacy threats in software systems. The name is an acronym for the seven privacy threat categories it covers:

1. **Linkability**: Linking multiple items of interest without knowing the identity
2. **Identifiability**: Connecting an identity to an item of interest
3. **Non-repudiation**: Being unable to deny knowledge or actions
4. **Detectability**: The ability to distinguish whether an item of interest exists
5. **Disclosure of information**: Revealing information to unauthorized parties
6. **Unawareness**: Users being unaware of privacy implications 
7. **Non-compliance**: Failing to comply with privacy regulations or policies

### LINDDUN Methodology

The framework follows a systematic process:

1. Create a detailed data flow diagram (DFD) of the system
2. Map privacy threats to DFD elements
3. Identify threats based on threat trees for each LINDDUN category
4. Prioritize threats based on risk assessment
5. Select privacy-enhancing technologies (PETs) and strategies to mitigate threats
6. Implement and validate the solutions

### When to Use LINDDUN

LINDDUN is most appropriate when:
- Privacy is a primary concern for your application
- You're handling personal or sensitive user data
- Compliance with privacy regulations is required
- You need to conduct a thorough privacy impact assessment

### Key Strengths

- Comprehensive focus on privacy threats
- Methodical approach with supporting threat trees
- Complements security-focused frameworks like STRIDE
- Links identified threats to mitigation strategies
- Helps ensure regulatory compliance for privacy
""",
        "NIST 800-30": """
## NIST Special Publication 800-30 Risk Assessment Framework

**NIST 800-30** is a risk assessment methodology provided by the National Institute of Standards and Technology as part of their special publications on computer security. It provides guidance for conducting risk assessments of federal information systems and organizations.

### NIST 800-30 Methodology

The framework consists of four steps:

1. **Prepare for Assessment**
   - Define purpose, scope, assumptions, and constraints
   - Identify sources of information
   - Determine risk model and analytic approach

2. **Conduct Assessment**
   - Identify threats and vulnerabilities
   - Determine likelihood and impact
   - Determine risk

3. **Communicate Results**
   - Document results
   - Share results with stakeholders

4. **Maintain Assessment**
   - Monitor risk factors
   - Update assessment

### When to Use NIST 800-30

NIST 800-30 is particularly useful when:
- Regulatory compliance is required, especially for government systems
- A well-documented, standardized approach is needed
- Comprehensive risk assessment beyond just technical threats is desired
- You need to integrate with other NIST frameworks (e.g., Cybersecurity Framework)

### Key Strengths

- Comprehensive and well-documented methodology
- Recognized standard, especially for government systems
- Adaptable to different organizational contexts
- Strong focus on organizational risk management
- Integrates with other NIST security frameworks
""",
        "FAIR": """
## Factor Analysis of Information Risk (FAIR)

**FAIR** is a framework for understanding, analyzing, and measuring information risk. It provides a model for quantitative risk assessment, helping organizations make more informed decisions about security investments.

### FAIR Taxonomy

FAIR breaks down risk into key components:

1. **Loss Event Frequency (LEF)**
   - Threat Event Frequency (TEF): How often threats occur
   - Vulnerability (Vuln): Probability that a threat will result in a loss

2. **Loss Magnitude (LM)**
   - Primary Loss: Direct loss from an event
   - Secondary Loss: Indirect loss from an event

### FAIR Methodology

The FAIR process includes:

1. Identify risk scenarios to analyze
2. Evaluate Loss Event Frequency factors
3. Evaluate Loss Magnitude factors
4. Calculate and articulate risk
5. Identify options for risk treatment
6. Perform cost-benefit analysis for risk treatments

### When to Use FAIR

FAIR is most appropriate when:
- Quantitative risk analysis is required
- Financial justification for security controls is needed
- Comparing different security investment options
- Communicating risk in business terms
- Need to move beyond qualitative risk assessments

### Key Strengths

- Quantitative approach to risk assessment
- Financial expression of risk
- Common language between security and business stakeholders
- Supports cost-benefit analysis of security controls
- Enables comparison of different risk scenarios
""",
        "ATT&CK": """
## MITRE ATT&CK Framework

**ATT&CK** (Adversarial Tactics, Techniques, and Common Knowledge) is a globally-accessible knowledge base of adversary tactics and techniques based on real-world observations. It provides a common language for describing adversary behaviors.

### ATT&CK Matrix

The ATT&CK framework is organized as a matrix of tactics and techniques:

1. **Tactics**: The "why" of an attack technique (tactical goal)
   - Initial Access, Execution, Persistence, Privilege Escalation, Defense Evasion, Credential Access, Discovery, Lateral Movement, Collection, Command and Control, Exfiltration, Impact

2. **Techniques**: The "how" of an attack (methods used to achieve tactical goals)
   - Each tactic contains multiple techniques and sub-techniques

3. **Procedures**: Specific implementations by threat actors

### When to Use ATT&CK

ATT&CK is particularly useful when:
- Threat intelligence is a key driver for security
- Understanding attacker behavior is critical
- Assessing security control coverage
- Building detection and response capabilities
- Red team/blue team exercises

### Key Strengths

- Based on real-world attack observations
- Comprehensive coverage of attack techniques
- Common language for security teams
- Supports mapping of defenses to specific threats
- Enables threat-informed defense strategy
- Regular updates based on evolving threats
""",
        "Hybrid": """
## Hybrid Framework Approach

A **Hybrid Framework** approach combines elements from multiple threat modeling frameworks to create a customized methodology that addresses specific organizational needs. This approach recognizes that no single framework is perfect for all scenarios.

### Benefits of a Hybrid Approach

1. **Comprehensive Coverage**: Combines the strengths of multiple frameworks
2. **Tailored to Specific Needs**: Adaptable to unique organizational requirements
3. **Balanced Perspective**: Integrates technical, process, and business considerations
4. **Flexible Implementation**: Can be scaled based on available resources and time
5. **Evolving Methodology**: Can incorporate new frameworks and approaches as they emerge

### Common Hybrid Combinations

- **STRIDE + DREAD**: Technical threat identification with risk prioritization
- **STRIDE + ATT&CK**: System-focused threats combined with attacker techniques
- **PASTA + FAIR**: Risk-centric approach with quantitative risk analysis
- **STRIDE + LINDDUN**: Security and privacy threat modeling together
- **ATT&CK + CVSS**: Threat scenarios with standardized vulnerability scoring

### Implementation Approach

1. Define security objectives and scope
2. Select relevant frameworks based on specific needs
3. Map complementary elements across frameworks
4. Create a unified process incorporating selected elements
5. Document the custom approach for consistency
6. Iterate and refine based on effectiveness

### When to Use a Hybrid Approach

A hybrid approach is most appropriate when:
- Standard frameworks don't address all your requirements
- You have diverse systems with different threat profiles
- Both technical and non-technical aspects need consideration
- You want to balance comprehensiveness with practicality
- Your organization has the maturity to adapt and customize methodologies
""",
        "Custom": """
## Custom Threat Modeling Framework

A **Custom Framework** is a specialized threat modeling approach developed specifically for your organization or application. It's designed to address unique requirements that aren't adequately covered by standard frameworks.

### Developing a Custom Framework

1. **Analyze Requirements**: Identify specific needs and gaps in existing frameworks
2. **Define Scope and Objectives**: Clarify what the framework should achieve
3. **Design Methodology**: Create processes, templates, and supporting materials
4. **Build Taxonomy**: Develop categorization for threats relevant to your context
5. **Create Evaluation Criteria**: Establish how to assess and prioritize threats
6. **Document Framework**: Create comprehensive documentation to ensure consistent application
7. **Test and Refine**: Apply to pilot projects and iterate based on feedback

### When to Develop a Custom Framework

Creating a custom framework is appropriate when:
- Your technology stack or architecture has unique security considerations
- Industry-specific threats aren't addressed by general frameworks
- Existing frameworks are too complex or too simple for your needs
- You have organizational constraints that require a specialized approach
- You need to integrate threat modeling with proprietary development processes

### Key Considerations

- **Build on Existing Knowledge**: Don't reinvent the wheel—incorporate proven elements
- **Focus on Usability**: Ensure the framework is practical and applicable
- **Maintain Consistency**: Establish clear guidelines for application
- **Plan for Evolution**: Create mechanisms to update the framework as threats evolve
- **Balance Depth and Breadth**: Cover key threats thoroughly while maintaining reasonable scope
- **Measure Effectiveness**: Track metrics to validate framework performance

### Getting Started

Begin by identifying key stakeholders and conducting workshops to understand requirements. Review existing frameworks to identify useful components, and create a draft methodology for pilot testing before full implementation.
"""
    }
    
    return descriptions.get(framework_name, "Detailed description not available for this framework.")