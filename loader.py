from tabula import read_pdf
from langchain.document_loaders import PyMuPDFLoader
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import re
    
def replace_pattern(patterns, text): 
    _text = text
    for pattern in patterns:
        _text = re.sub(pattern["target_str"], pattern.get("replace_str", ""), _text)
    
    return _text


def load_process_split_doc(
    filename, 
    directory_path="data/tax_etc",
    page_ranges=[],
    table_pages_range='',
    keyword="",
    effective_year=None,
    replace_string=[]
    ):

    docs = []
    pdf_file = f"{directory_path}/{filename}.pdf"
    
    # pdf loader
    try:
        text_loader = PyMuPDFLoader(pdf_file)
        texts = text_loader.load()

        for i, text in enumerate(texts):
            text.metadata["page"] = i + 1      

        texts = [
            text for text in texts
            if any(start <= text.metadata.get("page", 0) <= end for start, end in page_ranges)
        ]
    except Exception as e:
        texts = []
        
    # pdf loader - 테이블
    try:
        tables = read_pdf(pdf_file, pages=table_pages_range, stream=True, multiple_tables=True)
        table_texts = [table.to_string(index=False, header=True) for table in tables]
    except Exception as e:
        table_texts = []
    
    text_docs = [
        Document(page_content=text.page_content, metadata={
            "title": filename, 
            "keyword": keyword,
            "effective_year": effective_year, 
            "source": pdf_file
        })
        for text in texts
    ]
        
    table_docs = [
        Document(page_content=text, metadata={
            "title": filename, 
            "keyword": keyword,
            "effective_year": effective_year, 
            "source": f"{pdf_file} - Table {i + 1}"
        })
        for i, text in enumerate(table_texts)
    ]

    # 문서결합 (텍스트 문서 + 표 문서)
    docs.extend(text_docs + table_docs)

    update_docs = []
    
    for doc in docs:
        if doc.page_content:
            edit_content = replace_pattern(replace_string, doc.page_content)
        else:
            edit_content = ""
        
        updated_docs = Document(page_content=edit_content, metadata=doc.metadata)   
        update_docs.append(updated_docs)
        
    # split
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        model_name="gpt-4o",
        chunk_size=2000,
        chunk_overlap=100,
    )
    split_docs = splitter.split_documents(update_docs)

    return split_docs

def load_process_split_doc_law(file_name):
    
    """
    PDF 파일들을 처리하여 임베딩을 Chroma Vector Store에 저장합니다.
    """

    file_path = f"data/tax_law/{file_name}.pdf"

    loader = PyMuPDFLoader(file_path)
    load_document = loader.load()

    # 전처리 - 반복 텍스트 삭제
    delete_pattern_1 = rf"법제처\s*\d+\s*국가법령정보센터\n{file_name.replace('_', ' ')}\n" 
    delete_pattern_2 = r'\[[\s\S]*?\]'
    delete_pattern_3 = r'<[\s\S]*?>'
    
    full_text = " ".join([document.page_content for document in load_document])
    
    full_text = re.sub(delete_pattern_1, "", full_text)
    full_text=re.sub(delete_pattern_2, '', full_text)
    full_text=re.sub(delete_pattern_3, '', full_text)
    
    # 전처리 - split
    split_pattern = r"\s*\n(제\d+조(?:의\d+)?(?:\([^)]*\))?)(?=\s|$)"
    chunks = re.split(split_pattern, full_text)

    chunk_docs = []  # 최종 return 할 list
    connected_chunks = [] 
    current_chunk = ""
    is_buchik_section = False 
    
    # 전처리 - 일반 조항과 부칙의 조항과 구별하기 위해 접두어 '부칙-'을 넣음.
    for chunk in chunks:
        if re.search(r"\n\s*부칙", chunk): 
            is_buchik_section = True

        if chunk.startswith("제") and "조" in chunk: 
            if is_buchik_section:
                chunk = "부칙-" + chunk

            if current_chunk:
                connected_chunks.append(current_chunk.strip())
            current_chunk = chunk 
        else:
            current_chunk += f" {chunk}" 

    if current_chunk:
        connected_chunks.append(current_chunk.strip())

    for chunk in connected_chunks:
        pattern =  r"^(?:부칙-)?제\d+조(?:의\d*)?(?:\([^)]*\))?"

        keyword = ''
        keyword += file_name
        
        match = re.search(pattern, chunk)
        if match:
            word = match.group()
            word = re.sub(r'\s+', ' ', word)
            keyword += word 
            
        doc = Document(metadata={"title": file_name, "keyword":keyword, "effective_year": 2025 }, page_content=chunk),
        
        chunk_docs.extend(doc)
        
    return chunk_docs