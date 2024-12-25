import re
from tabula import read_pdf
from langchain.document_loaders import PyMuPDFLoader
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
load_dotenv()


def load_연말정산신고안내():
    docs = []
    pdf_file = "data/tax_etc/연말정산_신고안내.pdf"
    # Tabula로 PDF에서 표 데이터 읽기
    try:
        tables = read_pdf(pdf_file, pages="17-425", stream=True, multiple_tables=True) # pages="19-44,47-71,75-161"
        table_texts = [table.to_string(index=False, header=True) for table in tables]
    except Exception as e:
        print(f"Tabula Error with {pdf_file}: {e}")
        table_texts = []

    # PyMuPDFLoader로 텍스트 데이터 로드 및 페이지 범위 필터링링
    try:
        text_loader = PyMuPDFLoader(pdf_file)
        texts = text_loader.load()

        for i, text in enumerate(texts):
            text.metadata["page"] = i + 1      

        page_ranges = [(17, 426)]
        texts = [
            text for text in texts
            if any(start <= text.metadata.get("page", 0) <= end for start, end in page_ranges)
        ]
    except Exception as e:
        print(f"PyMuPDFLoader Error with {pdf_file}: {e}")
        texts = []

    # 표 텍스트를 Document 형식으로 변환
    table_docs = [
        Document(page_content=text, metadata={"source": f"{pdf_file} - Table {i + 1}"})
        for i, text in enumerate(table_texts)
    ]

        # 텍스트를 Document 형식으로 변환
    text_docs = [
        Document(page_content=text.page_content, metadata={"source": pdf_file})
        for text in texts
    ]

        # 텍스트 문서와 표 문서를 결합
    docs.extend(text_docs + table_docs)

    # 정규식 패턴화
    update_docs = []

    pattern1 = (
        r"01. 2024년 귀속 연말정산 개정세법 요약\n\d+|"  
        r"II. 2024년 귀속 연말정산 주요 일정\n\d+|"
        r"III\. 원천징수의무자의 연말정산 중점 확인사항\n\d+|"
        r"원천징수의무자를 위한 \n2024년 연말정산 신고안내\n\d+|"
        r"Ⅰ\. 근로소득\n\d+|"
        r"II\. 근로소득 원천징수 및 연말정산\n\d+|"
        r"III\. 근로소득공제, 인적공제, 연금보험료공제\n\d+|"
        r"IV\. 특별소득공제(소법 §52)\n\d+|"
        r"V\. 그 밖의 소득공제(조특법)\n\d+|"
        r"VI\. 세액감면(공제) 및 농어촌특별세\n\d+|"
        r"I\. 2024년 귀속 연말정산 종합사례\n\d+|"
        r"II\. 근로소득 원천징수영수증(지급명세서) 작성요령\n\d+|"
        r"IV\. 수정 원천징수이행상황신고서 작성사례(과다공제)\n\d+|"
        r"VI\. 홈택스를 이용한 연말정산 신고(근로소득 지급명세서 제출)\n\d+|"
        r"I\. 사업소득 연말정산\n\d+|"
        r"II\. 연금소득 연말정산\n\d+|"
        r"I\. 종교인소득이란?\n\d+|"
        r"IV\. 종교인소득(기타소득)에 대한 연말정산\n\d+|"
        r"부록1\. 연말정산 관련 서비스\n\d+|"
        r"부록2\. 연말정산간소화 서비스\n\d+|"
        r"부록5\. 연말정산 주요 용어 설명\n\d+|"
        r"부록6\. 소득·세액공제신고서 첨부서류\n\d+|"
        r"\bNaN\b"
    )
    pattern2 =r"([\uAC00-\uD7A3])\n+([\uAC00-\uD7A3])"
    pattern3 = r"\s+"

    for doc in docs:
        if doc.page_content:
            edit_content = re.sub(pattern1, "", doc.page_content)
            edit_content = re.sub(pattern2, r"\1\2" , edit_content)
            edit_content = re.sub(pattern3, " ", edit_content)
        else:
            edit_content = ""
        
        updated_docs = Document(page_content=edit_content, metadata=doc.metadata)   
        update_docs.append(updated_docs)
        
    # 텍스트를 청크로 분리
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        model_name="gpt-4o",
        chunk_size=2000,
        chunk_overlap=100,
    )

    split_docs = splitter.split_documents(update_docs)

    return split_docs
