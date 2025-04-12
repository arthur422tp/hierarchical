"""
This is for chunking
"""
from langchain.text_splitter import RecursiveCharacterTextSplitter

class RagChunking():
    """
    This is for chunking
    """
    def __init__(self, text):
        self.text = text

    def text_chunking(self, chunk_size:int, chunk_overlap:int):
        """
        This is for chunking
        """
        text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "●"],
                                                       chunk_size=chunk_size,
                                                       chunk_overlap=chunk_overlap)
        chunking_text = text_splitter.split_text(self.text)
        
        return chunking_text