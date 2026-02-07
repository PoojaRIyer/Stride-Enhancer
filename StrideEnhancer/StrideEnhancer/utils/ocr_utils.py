import cv2
import numpy as np
import pytesseract
from PIL import Image

def preprocess_image(image):
    """
    Preprocess the image to improve OCR accuracy.
    
    Args:
        image: PIL Image object
        
    Returns:
        Preprocessed image as numpy array
    """
    # Convert to numpy array if it's a PIL Image
    if isinstance(image, Image.Image):
        img = np.array(image)
    else:
        img = image
    
    # Convert to grayscale
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img
    
    # Apply threshold to get black and white image
    _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    # Apply noise removal
    denoised = cv2.fastNlMeansDenoising(binary, None, 10, 7, 21)
    
    return denoised

def extract_text_from_image(image):
    """
    Extract text from an image using OCR with enhanced accuracy.
    
    Args:
        image: PIL Image object
        
    Returns:
        Extracted text as string
    """
    # Preprocess image
    processed_img = preprocess_image(image)
    
    # Convert preprocessed image back to PIL Image for pytesseract
    pil_img = Image.fromarray(processed_img)
    
    # Apply OCR with optimized configuration
    custom_config = r'--oem 3 --psm 6 -l eng'
    text = pytesseract.image_to_string(pil_img, config=custom_config)
    
    return text

def extract_diagram_elements(image):
    """
    Extract diagram elements (boxes, lines, shapes) from an image.
    This function attempts to identify components and connections in a system diagram.
    
    Args:
        image: PIL Image object
        
    Returns:
        Dictionary of identified elements and their positions
    """
    # Convert PIL image to numpy array
    if isinstance(image, Image.Image):
        img = np.array(image)
    else:
        img = image
        
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if len(img.shape) == 3 else img
    
    # Apply threshold
    _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Extract shapes
    shapes = []
    for cnt in contours:
        # Calculate contour properties
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        x, y, w, h = cv2.boundingRect(cnt)
        
        # Filter out noise and small contours
        if area > 500:  # Adjust threshold as needed
            # Determine if it's a rectangle, circle, or another shape
            approx = cv2.approxPolyDP(cnt, 0.03 * perimeter, True)
            shape_type = len(approx)
            
            # Add to shapes list
            shapes.append({
                'type': 'box' if shape_type == 4 else 'ellipse' if shape_type > 8 else 'polygon',
                'position': (x, y),
                'size': (w, h),
                'bounds': (x, y, x+w, y+h)
            })
    
    # Find lines - used for connections in diagrams
    lines = cv2.HoughLinesP(binary, 1, np.pi/180, threshold=50, minLineLength=50, maxLineGap=10)
    
    connections = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            connections.append({
                'type': 'line',
                'start': (x1, y1),
                'end': (x2, y2)
            })
    
    # Extract text areas - typically inside shapes
    text_areas = []
    for shape in shapes:
        x, y, w, h = shape['position'][0], shape['position'][1], shape['size'][0], shape['size'][1]
        roi = gray[y:y+h, x:x+w]
        
        # Check if the region might contain text using histogram analysis
        if roi.size > 0:  # Ensure the region is not empty
            hist = cv2.calcHist([roi], [0], None, [256], [0, 256])
            hist_variation = np.std(hist)
            
            if hist_variation > 10:  # Threshold for text vs. non-text
                text_areas.append({
                    'position': (x, y),
                    'size': (w, h),
                    'parent_shape': shapes.index(shape)
                })
    
    return {
        'shapes': shapes,
        'connections': connections,
        'text_areas': text_areas
    }