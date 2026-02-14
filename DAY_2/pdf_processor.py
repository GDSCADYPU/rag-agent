"""
Day 2 - Exercise 2: PDF Processing and Text Extraction
======================================================

Welcome to real-world document processing! 📄

The Challenge:
Most valuable information lives in PDFs - research papers, manuals, 
reports, guidelines. To build a useful RAG system, we need to extract 
text from these documents!

What you'll learn:
✓ Extracting text from PDF files
✓ Handling multi-page documents
✓ Extracting metadata (title, author, page count)
✓ Processing entire directories of PDFs
✓ Dealing with real-world messy data

Real-world applications:
- Legal document analysis
- Research paper Q&A systems
- Company knowledge bases
- Student study assistants
"""

import PyPDF2
from pathlib import Path
from typing import List, Dict
import sys

class PDFProcessor:
    """
    A powerful PDF text extraction system! 📄➡️📝
    
    Think of this as a super-fast reader who can go through hundreds
    of PDF pages in seconds and extract all the text for processing.
    
    Fun fact: PDFs were invented by Adobe in 1993 to share documents
    across different computers and operating systems!
    """
    
    def __init__(self):
        """Initialize the PDF processor"""
        print("✅ PDF Processor initialized and ready!")
        print("   Supported: Text extraction, metadata, batch processing")
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extract all text from a PDF file.
        
        This is the core function - it reads every page and combines
        the text into one string.
        
        Args:
            pdf_path (str): Path to the PDF file
            
        Returns:
            str: All extracted text
            
        Example:
            >>> processor = PDFProcessor()
            >>> text = processor.extract_text_from_pdf('guidelines.pdf')
            >>> print(f"Extracted {len(text)} characters")
        """
        try:
            with open(pdf_path, 'rb') as file:
                # Create PDF reader object
                pdf_reader = PyPDF2.PdfReader(file)
                
                # Get number of pages
                num_pages = len(pdf_reader.pages)
                
                print(f"📖 Reading '{Path(pdf_path).name}'...")
                print(f"   Pages: {num_pages}")
                
                # Extract text from all pages
                text = ""
                for page_num in range(num_pages):
                    page = pdf_reader.pages[page_num]
                    page_text = page.extract_text()
                    text += page_text + "\n"
                    
                    # Progress indicator for large PDFs
                    if (page_num + 1) % 10 == 0:
                        print(f"   Progress: {page_num + 1}/{num_pages} pages...")
                
                print(f"✅ Extracted {len(text)} characters from {pdf_path}")
                return text
        
        except FileNotFoundError:
            error_msg = f"❌ File not found: {pdf_path}"
            print(error_msg)
            return ""
        
        except Exception as e:
            error_msg = f"❌ Error processing {pdf_path}: {str(e)}"
            print(error_msg)
            return ""
    
    def extract_with_metadata(self, pdf_path: str) -> Dict:
        """
        Extract text AND metadata from PDF.
        
        Metadata includes:
        - Title, Author, Subject
        - Creation date
        - Number of pages
        - Page-by-page text
        
        This is super useful for:
        - Citing sources ("According to page 5 of document X...")
        - Filtering by document type
        - Tracking document provenance
        
        Args:
            pdf_path (str): Path to PDF file
            
        Returns:
            dict: Complete document information
        """
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                # Extract metadata (if available)
                metadata = pdf_reader.metadata if pdf_reader.metadata else {}
                
                # Extract text from all pages (with page numbers)
                pages = []
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    pages.append({
                        'page_number': page_num + 1,
                        'text': page.extract_text()
                    })
                
                # Compile full document
                full_text = '\n'.join([p['text'] for p in pages])
                
                return {
                    'filename': Path(pdf_path).name,
                    'path': pdf_path,
                    'num_pages': len(pdf_reader.pages),
                    'metadata': {
                        'title': metadata.get('/Title', 'Unknown'),
                        'author': metadata.get('/Author', 'Unknown'),
                        'subject': metadata.get('/Subject', 'Unknown'),
                        'creator': metadata.get('/Creator', 'Unknown'),
                    },
                    'pages': pages,
                    'full_text': full_text,
                    'char_count': len(full_text),
                    'word_count': len(full_text.split())
                }
        
        except Exception as e:
            print(f"❌ Error extracting metadata from {pdf_path}: {str(e)}")
            return {
                'filename': Path(pdf_path).name,
                'error': str(e)
            }
    
    def process_directory(self, directory_path: str) -> List[Dict]:
        """
        Process ALL PDFs in a directory.
        
        This is incredibly powerful for building knowledge bases!
        Point it at a folder of documents and it processes everything.
        
        Args:
            directory_path (str): Path to directory containing PDFs
            
        Returns:
            list: List of document dictionaries
            
        Example:
            >>> processor = PDFProcessor()
            >>> docs = processor.process_directory('./company_docs')
            >>> print(f"Processed {len(docs)} documents")
        """
        directory = Path(directory_path)
        
        # Find all PDF files
        pdf_files = list(directory.glob('*.pdf'))
        
        if not pdf_files:
            print(f"⚠️  No PDF files found in {directory_path}")
            return []
        
        print(f"\n📁 Found {len(pdf_files)} PDF files in '{directory_path}'")
        print("=" * 70)
        
        documents = []
        
        for i, pdf_file in enumerate(pdf_files, 1):
            print(f"\n[{i}/{len(pdf_files)}] Processing: {pdf_file.name}")
            doc = self.extract_with_metadata(str(pdf_file))
            
            if 'error' not in doc:
                documents.append(doc)
                print(f"   ✅ Success: {doc['num_pages']} pages, {doc['word_count']} words")
            else:
                print(f"   ❌ Failed: {doc['error']}")
        
        print("\n" + "=" * 70)
        print(f"✅ Successfully processed {len(documents)}/{len(pdf_files)} PDFs")
        
        return documents
    
    def search_in_document(self, doc: Dict, search_term: str) -> List[Dict]:
        """
        Search for a term within a document and return matching pages.
        
        Useful for: "Which page mentions 'registration'?"
        
        Args:
            doc (dict): Document dictionary from extract_with_metadata
            search_term (str): Term to search for
            
        Returns:
            list: Pages containing the search term
        """
        if 'pages' not in doc:
            return []
        
        search_term_lower = search_term.lower()
        matching_pages = []
        
        for page in doc['pages']:
            if search_term_lower in page['text'].lower():
                matching_pages.append({
                    'page_number': page['page_number'],
                    'preview': page['text'][:200] + '...'
                })
        
        return matching_pages


# ============================================================================
# DEMO: Let's process some PDFs! 🚀
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("PDF PROCESSOR DEMONSTRATION - Unlocking Document Knowledge!")
    print("=" * 70 + "\n")
    
    processor = PDFProcessor()
    
    print("This tool can:")
    print("  ✓ Extract text from any PDF")
    print("  ✓ Get metadata (title, author, etc.)")
    print("  ✓ Process entire directories")
    print("  ✓ Search within documents")
    print()
    
    print("-" * 70)
    print("📚 USAGE EXAMPLES:")
    print("-" * 70)
    print("""
# Example 1: Extract text from a single PDF
text = processor.extract_text_from_pdf('document.pdf')

# Example 2: Get full metadata
doc = processor.extract_with_metadata('document.pdf')
print(f"Title: {doc['metadata']['title']}")
print(f"Pages: {doc['num_pages']}")

# Example 3: Process entire directory
docs = processor.process_directory('./my_pdfs')
for doc in docs:
    print(f"{doc['filename']}: {doc['word_count']} words")

# Example 4: Search in document
results = processor.search_in_document(doc, 'Python')
for result in results:
    print(f"Found on page {result['page_number']}")
    """)
    
    print("\n" + "-" * 70)
    print("🎯 NEXT STEPS:")
    print("-" * 70)
    print("""
1. Try processing your own PDFs!
2. Combine with chunking_utility to break PDFs into chunks
3. Tomorrow: Store these chunks in a vector database!

Pro tip: Keep your PDFs in a dedicated folder for easy batch processing.
    """)
    
    print("=" * 70)
    print("✨ PDF processing mastered!")
    print("Next up: Knowledge Base - Building a vector database!")
    print("=" * 70 + "\n")