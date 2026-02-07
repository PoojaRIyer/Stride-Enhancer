import io
import pandas as pd
import tempfile
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch 

def export_to_pdf(threats_mitigations, framework="STRIDE", hybrid_components=None):
    """
    Export the threat modeling results to a PDF file.
    
    Args:
        threats_mitigations: List of threat dictionaries
        framework: The framework used for threat modeling
        hybrid_components: List of frameworks if using a hybrid approach
    
    Returns:
        PDF file as bytes
    """
    # Create a file-like object to receive PDF data
    buffer = io.BytesIO()
    
    # Create the PDF document
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []
    
    # Add title and header
    if framework == "Hybrid" and hybrid_components:
        title = f"Threat Model Report - Hybrid Framework ({', '.join(hybrid_components)})"
    else:
        title = f"Threat Model Report - {framework} Framework"
    
    elements.append(Paragraph(title, styles["Title"]))
    elements.append(Spacer(1, 0.25*inch))
    
    # Add threats and mitigations
    table_data = [["ID", "Threat", "Category", "Description", "Mitigations"]]
    
    for i, threat in enumerate(threats_mitigations):
        threat_id = f"T{i+1}"
        threat_name = threat.get("name", "Unnamed Threat")
        category = threat.get("category", "Uncategorized")
        description = threat.get("description", "No description provided")
        mitigations = threat.get("mitigations", [])
        
        # Format mitigations as bullet points
        mitigations_text = ""
        for j, mitigation in enumerate(mitigations):
            if isinstance(mitigation, dict):
                mitigation_text = mitigation.get('description', str(mitigation))
            else:
                mitigation_text = str(mitigation)
            mitigations_text += f"• {mitigation_text}\n"
        
        table_data.append([threat_id, threat_name, category, description, mitigations_text])
    
    # Create the table
    table = Table(table_data, repeatRows=1)
    
    # Style the table
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.white),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 10),
        ('WORDWRAP', (0, 1), (-1, -1), True),
    ]))
    
    elements.append(table)
    
    # Build the PDF document
    doc.build(elements)
    
    # Get the value of the file-like object
    pdf_value = buffer.getvalue()
    buffer.close()
    
    return pdf_value

def export_to_excel(threats_mitigations, framework="STRIDE", hybrid_components=None):
    """
    Export the threat modeling results to an Excel file.
    
    Args:
        threats_mitigations: List of threat dictionaries
        framework: The framework used for threat modeling
        hybrid_components: List of frameworks if using a hybrid approach
    
    Returns:
        Excel file as bytes
    """
    # Create a pandas DataFrame
    data = []
    
    for i, threat in enumerate(threats_mitigations):
        threat_id = f"T{i+1}"
        threat_name = threat.get("name", "Unnamed Threat")
        category = threat.get("category", "Uncategorized")
        description = threat.get("description", "No description provided")
        risk = threat.get("risk", "Not Assessed")
        
        # Format mitigations as bullet points
        mitigations = threat.get("mitigations", [])
        mitigations_text = ""
        
        for j, mitigation in enumerate(mitigations):
            if isinstance(mitigation, dict):
                mitigation_text = mitigation.get('description', str(mitigation))
            else:
                mitigation_text = str(mitigation)
            mitigations_text += f"• {mitigation_text}\n"
        
        # Add row to data
        data.append({
            "ID": threat_id,
            "Threat": threat_name,
            "Category": category,
            "Description": description,
            "Risk": risk,
            "Mitigations": mitigations_text
        })
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Create Excel file
    excel_buffer = io.BytesIO()
    
    with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Threats')
        
        # Auto-adjust column widths
        for column in df:
            column_width = max(df[column].astype(str).map(len).max(), len(column)) + 2
            col_idx = df.columns.get_loc(column)
            writer.sheets['Threats'].column_dimensions[chr(65 + col_idx)].width = min(column_width, 50)
    
    excel_data = excel_buffer.getvalue()
    excel_buffer.close()
    
    return excel_data