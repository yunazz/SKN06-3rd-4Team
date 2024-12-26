# SKN06-3rd-4Team


# 👩‍⚖️세법 질의응답 시스템👨‍⚖️ 
 
**개발기간:** 2024.12.24 - 2024.12.26

## 💻 팀 소개

# 팀명: 절세미인

<img src="https://github.com/user-attachments/assets/df44c785-4e29-44ce-ab52-48ed3abee8d7" width="600" height="500">



| [노원재] | [박서윤] | [박유나] | [유경상] | [전하연] |
|:-------:|:-------:|:-------:|:-------:|:-------:|
|<img src="https://github.com/user-attachments/assets/83778ae7-cead-4f34-ab0c-27d50300b23f" width="160" height="160">|<img src="https://github.com/user-attachments/assets/0889164e-66b0-4df2-b3f7-4ede25b6ee80"  width="160" height="160">|<img src="https://github.com/user-attachments/assets/faf2d9c0-a945-4604-98bb-af2b1907acae"   width="160" height="160">|<img src="https://github.com/user-attachments/assets/bd07a4f7-65aa-49c2-9cfd-e54f41d08282"  width="160" height="160">|<img src="https://github.com/user-attachments/assets/087fcd42-4884-425d-a7b0-fe866ff18b03"  width="160" height="160">|
|    절세남    |     절세미녀    |    절세미녀     |     절세남    |     절세미녀    |


---

## 📌 1. 프로젝트 개요

### 1.1. 개발 동기 및 목적  
본 프로젝트는 **복잡한 세법 관련 질문에 대한 신속하고 정확한 답변 제공**을 목표로 하는 **세법 확인 챗봇 시스템**을 구현하고자 했습니다. 법제처에서 제공하는 세법 관련 데이터를 활용하여 세금 관련 질의에 대한 정확한 답변을 제공하는 챗봇 시스템을 개발했습니다.  

### 1.2. 필요성  
세법은 복잡하고 자주 개정되어 일반인들이 이해하기 어려운 경우가 많습니다. 특히 **개인과 기업의 세금 관련 문의**에 대해 신속하고 정확한 정보 제공이 필요합니다. 따라서, **세금 관련 자주 묻는 질문**에 즉시 답변할 수 있는 챗봇 시스템을 통해 **세법에 대한 접근성을 향상**시키고 사용자들의 이해를 돕고자 했습니다.

### 1.3. 개발 목표  
- **정확한 정보 제공:** 최신 세법 규정과 절차 반영  
- **정확성 향상:** 일반적인 챗봇 한계를 넘는 **구체적이고 정확한 세법 관련 답변 제공**  

### 1.4. 디렉토리 구조
```
├── data 
│   ├── tax_law : 세법 관련 (70개)
│       ├── 개별소비세법.pdf  
│       ├── 개별소비세법_시행규칙.pdf
│       ├── 개별소비세법_시행령.pdf  
│       ├── 과세자료의_제출_및_관리에_관한_법률.pdf
│       └── ...
│   └── tax_etc : 기타 데이터 (6개)
│       ├── 2024_핵심_개정세법.pdf  
│       ├── 연말정산_Q&A.pdf  
│       ├── 연말정산_신고안내.pdf  
│       ├── 연말정산_주택자금·월세액_공제의이해.pdf  
│       ├── 주요_공제_항목별_계산사례.pdf  
│       └── 키워드연말정산.pdf  
├── loader.py : 데이터 로딩을 위한 함수 모듈  
└── main.ipynb : 프로젝트 최종 노트북 파일  
```

---

## 📌 2. 상세 내용

### 2.1. 데이터 수집
-  [법제처](https://www.law.go.kr/LSW/main.html)와 [국세청](https://www.nts.go.kr/nts/cm/cntnts/cntntsView.do?mi=2304&cntntsId=238938)에서 세법 관련 데이터 다운로드

### 2.2. 데이터 로드
- **PyMuPDFLoader**
  - 필요없는 페이지 제외하고 읽기기

    ```python3
    text_loader = PyMuPDFLoader(pdf_file)
    texts = text_loader.load()

    for i, text in enumerate(texts):
        text.metadata["page"] = i + 1      

    page_ranges = [(17, 426)]
    texts = [
        text for text in texts
        if any(start <= text.metadata.get("page", 0) <= end for start, end in page_ranges)
    ]
    ```
- **Tabula**
  - PDF의 테이블 데이터 읽기용
  - 적용 파일:<br/>2024_핵심_개정세법.pdf, 연말정산_신고안내.pdf, 연말정산_주택자금·월세액_공제의이해.pdf, 주요_공제_항목별_계산사례.pdf

    ```python3
    tables = read_pdf(pdf_file, pages=table_pages_range, stream=True, multiple_tables=True)
        table_texts = [table.to_string(index=False, header=True) for table in tables]
    ```
### 2.3. 데이터 전처리
- **불필요 텍스트 제거**
  - 세법 관련 (개별소비세법~증권거래세법_시행령).pdf, 연말정산_Q&A.pdf
    ##### - 머리말, 꼬리말 제거
    ```python3
    rf'법제처\s*\d+\s*국가법령정보센터\n{file_name.replace('_', ' ')}\n'
    ```
    ##### - [ ]로 감싸진 텍스트 제거
    ```python3
    r'\[[\s\S]*?\]'
    #text = re.sub(r'【.*?】', '', text, flags=re.MULTILINE)
    ```
    ##### - < >로 감싸진  텍스트 제거
    ```python3
    r'<[\s\S]*?>'  
    ```
    ##### - 페이지 번호 패턴 제거
    ```python3
    text = re.sub(r'^- \d{1,3} -', '', text, flags=re.MULTILINE)
    ```
    
  - 2024_핵심_개정세법.pdf, 연말정산_신고안내.pdf, 연말정산_주택자금·월세액_공제의이해.pdf
    ##### - 머리말 및 사이드바 제거
    ```python3
    - r"2\n0\n2\n5\n\s*달\n라\n지\n는\n\s*세\n금\n제\n도|"  
    - r"\n2\n0\n2\n4\n\s*세\n목\n별\n\s*핵\n심\n\s*개\n정\n세\n법|"
    - r"\n2\n0\n2\n4\n\s*개\n정\n세\n법\n\s*종\n전\n-\n개\n정\n사\n항\n\s*비\n교\n|"
    - r"\s*3\s*❚국민･기업\s*납세자용\s*|"
    - r"\s*2\s*0\s*2\s*4\s|"
    - r"\s한국세무사회\s|" 
    - r"\n7\n❚국민･기업 납세자용|"
    - r"\n71\n❚상세본|"
    ```
    ##### - 문장이 다음줄로 넘어가면서 생기는 \n 제거 
    ```python3
    r"([\uAC00-\uD7A3])\n+([\uAC00-\uD7A3])"
    re.sub(pattern2, r"\1\2" , edit_content) #앞뒤글자 합치기
    ```

  - 연말정산_신고안내.pdf, 연말정산_주택자금·월세액_공제의이해.pdf, 주요_공제_항목별_계산사례.pdf
    ##### - NaN 제거
    ```python3
    r"\bNaN\b"
    ```
    #### - 하나 이상의 공백문자를 한 개의 공백문자로 바꾸기
    ```python3
    - r"\s+"
    - re.sub(pattern3, " ", edit_content)
    ```
    
    
### 2.4. split 방법
- 세법 관련 (개별소비세법~증권거래세법_시행령).pdf
  ```python3
  r"\s\n(제\d+조(?:의\d+)?(?:([^)]))?)(?=\s|$)"
  chunk size로 split 하지 않고 조항별로 분리
  ```

- 2024_핵심_개정세법.pdf, 연말정산_신고안내.pdf, 연말정산_주택자금·월세액_공제의이해.pdf, 주요_공제_항목별_계산사례.pdf
  ```python3
  chunk size = 2000, over lap = 100 으로 설정
  ```

- 연말정산_Q&A.pdf
  ```python3
   chunk size = 1000, over lap = 150 으로 설정
  ```



### 2.5. 벡터 데이터베이스 구현 
- 전처리된 데이터를 벡터화
- 선택한 벡터 데이터베이스에 데이터 저장
  ```python3
  COLLECTION_NAME = "tax_law"
  PERSIST_DIRECTORY = "tax"

  def set_vector_store(documents):
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")

    return Chroma.from_documents(
        documents=documents,
        embedding=embedding_model,
        collection_name=COLLECTION_NAME,
        persist_directory=PERSIST_DIRECTORY
    )
    ```

### 2.6. RAG 구현
- 질의 처리 로직 구현
- 관련 정보 검색 메커니즘 구축
  ```python3
  # Prompt Template 생성
  messages = [
          ("ai", """
           !!!수정해야함!!!
  	{context}")"""
          ),
          ("human", "{question}"),
  ]
  prompt_template = ChatPromptTemplate(messages)
  # 모델
  model = ChatOpenAI(model="gpt-4o")
  
  # output parser
  parser = StrOutputParser()
  
  # Chain 구성 retriever(관련문서 조회) -> prompt_template(prompt 생성) -> model(정답) -> output parser
  chain = {"context":retriever, "question": RunnablePassthrough()} | prompt_template | model | parser
  ```
  
---
## 📌 3. 성능 검증 및 평가
- **검증 방법:** 사용자가 입력한 질문과 챗봇의 응답을 비교하고, 실제 법령에 명시된 내용을 확인하여 정확성, 관련성,  신뢰성 등 평가 지표 설정 및 측정
- **결과:** 시스템이 제공하는 답변의 정확성과 일관성을 지속적으로 모니터링하여 시스템을 개선

```python
# LangChain 모델 래핑
langchain_model = LangchainLLMWrapper(model)

# 테스트 데이터 준비
test_data = [
    {
        "question": "개별소비세법 제1조가 무엇인가요?",
        "answer": chain.invoke("개별소비세법 제1조가 무엇인가요?"),
        "contexts": [doc.page_content for doc in retriever.get_relevant_documents("개별소비세법 제1조가 무엇인가요?")],
        "ground_truths": ["개별소비세는 특정한 물품, 특정한 장소 입장행위(入場行爲), 특정한 장소에서의 유흥음식행위(遊興飮食行爲) 및 특정한 장소에서의 영업행위에 대하여 부과한다."],
        "reference": "\n".join([doc.page_content for doc in retriever.get_relevant_documents("개별소비세법 제1조가 무엇인가요?")])
    }, ...추가로 검증할 데이터
]
```
```python
# Dataset 생성
dataset = Dataset.from_list(test_data)

# 평가 실행
result = evaluate(
    dataset,
    metrics=[
        faithfulness,       # 신뢰성
        answer_relevancy,   # 답변 적합성
        context_precision,  # 문맥 정확성
        context_recall      # 문맥 재현률
    ],
    llm=langchain_model)
```

---
## 📌 4. 한 줄 회고📝
 - **노원재**: 어렵군요..
 - **박서윤**: 이제까지 한 프로젝트 중에서 제일 재밌는데 잘 모르겠다,,,
 - **박유나**: 이번 프로젝트를 진행하며, 전처리 과정에 따라 결과가 크게 달라지는 것을 직접 확인할 수 있어 매우 흥미로웠습니다. 특히, 프롬프트 템플릿 메시지 작성 방식에 따라 결과가 극명하게 변화하는 점이 인상 깊었습니다. 이러한 세부적인 차이들이 결과에 미치는 영향을 체감하며, 프로젝트를 더욱 재미있게 느낄 수 있었습니다.
 - **유경상**: 이번 프로젝트에서 다양한 문서를 동시에 가져옴으로서 text 및 table표 읽는 방법과 정규표현식을통해 전처리하는 방법, prompt template 을 작성하고 구성하는 방법과 chain을 구성하는 방법을 배웠다. 
 - **전하연**: 쉽지 않다..
