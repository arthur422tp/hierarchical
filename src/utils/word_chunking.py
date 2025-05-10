"""
蒐集做chunking的函式
"""

from langchain.text_splitter import RecursiveCharacterTextSplitter


class RagChunking:
    """
    做chunking的class
    """

    def __init__(self, text):
        self.text = text

    def text_chunking(self, chunk_size: int, chunk_overlap: int):
        """
        做chunking的函式
        """
        text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "●"], chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        chunking_text = text_splitter.split_text(self.text)

        return chunking_text
