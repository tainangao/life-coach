from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter


def get_chunker(strategy_name: str):
    if strategy_name == "Markdown":
        # Useful if the user uploads structured notes
        return MarkdownHeaderTextSplitter(headers_to_split_on=[("#", "Header 1"), ("##", "Header 2")])
    elif strategy_name == "Semantic":
        # You could implement SemanticChunker here, 
        # but Recursive is safer/faster for a 'lightweight' app.
        return RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    else:
        return RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
