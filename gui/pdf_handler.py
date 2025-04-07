"""
PDF Handler Module

Provides robust PDF text extraction with multiple fallback methods.
"""

import os
import logging
import tempfile
from typing import Tuple, Optional

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_text_from_pdf(file_data, filename: str) -> Tuple[str, bool]:
    """
    Extract text from a PDF file using multiple methods.
    
    Args:
        file_data: The file data (either a file-like object or bytes)
        filename: Name of the file for logging
        
    Returns:
        Tuple of (extracted_text, success_flag)
    """
    # Save to temporary file for more reliable processing
    temp_path = None
    extracted_text = ""
    success = False
    
    try:
        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            if hasattr(file_data, 'read'):
                # If it's a file-like object
                file_data.seek(0)
                temp_file.write(file_data.read())
                file_data.seek(0)  # Reset position
            else:
                # If it's bytes
                temp_file.write(file_data)
            temp_path = temp_file.name
            
        logger.info(f"Saved PDF to temporary file: {temp_path}")
        
        # Try multiple methods for extraction
        methods = [
            extract_with_pypdf,
            extract_with_pdfminer,
            extract_with_pdfplumber
        ]
        
        for method in methods:
            try:
                text, method_success = method(temp_path)
                if method_success and text.strip():
                    extracted_text = text
                    success = True
                    logger.info(f"Successfully extracted text with {method.__name__}")
                    break
            except Exception as e:
                logger.warning(f"Method {method.__name__} failed: {str(e)}")
                continue
        
        # If all methods failed, log a warning
        if not success:
            logger.warning(f"All PDF extraction methods failed for {filename}")
            
    except Exception as e:
        logger.error(f"Error in extract_text_from_pdf: {str(e)}")
    finally:
        # Clean up temp file
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
                logger.info(f"Removed temporary file: {temp_path}")
            except Exception as e:
                logger.warning(f"Failed to remove temporary file: {str(e)}")
    
    return extracted_text, success

def extract_with_pypdf(filepath: str) -> Tuple[str, bool]:
    """Extract text using PyPDF2"""
    try:
        import PyPDF2
        
        with open(filepath, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            
            # Check if PDF is encrypted
            if reader.is_encrypted:
                try:
                    reader.decrypt('')  # Try empty password
                except:
                    logger.warning("PDF is encrypted and could not be decrypted")
                    return "", False
            
            # Extract text from each page
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n\n"
            
            return text, bool(text.strip())
    except Exception as e:
        logger.warning(f"PyPDF2 extraction failed: {str(e)}")
        return "", False

def extract_with_pdfminer(filepath: str) -> Tuple[str, bool]:
    """Extract text using pdfminer.six"""
    try:
        from pdfminer.high_level import extract_text
        text = extract_text(filepath)
        return text, bool(text.strip())
    except ImportError:
        logger.warning("pdfminer.six not installed")
        return "", False
    except Exception as e:
        logger.warning(f"pdfminer extraction failed: {str(e)}")
        return "", False

def extract_with_pdfplumber(filepath: str) -> Tuple[str, bool]:
    """Extract text using pdfplumber"""
    try:
        import pdfplumber
        
        with pdfplumber.open(filepath) as pdf:
            text = ""
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n\n"
            
            return text, bool(text.strip())
    except ImportError:
        logger.warning("pdfplumber not installed")
        return "", False
    except Exception as e:
        logger.warning(f"pdfplumber extraction failed: {str(e)}")
        return "", False
