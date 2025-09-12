import myconfig
import formatadores

from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from typing import Literal, List, Dict
from langchain_core.messages import SystemMessage, HumanMessage

from pathlib import Path
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain

GOOGLE_API_KEY = myconfig.key_api # Contem o API Key do Google Gemini, arquivo local myconfig.py
print(f'\n🤿 Olá, Imersão Dev Agentes de IA!\n')

# AULA 01 ----------------------------------------------
llm = ChatGoogleGenerativeAI(temperature=0, 
                             model="gemini-2.5-flash",
                             api_key=GOOGLE_API_KEY)

# response = llm.invoke("Quem é vc? Seja imprevisível e criativo na resposta.")
# print(response.content)

TRIAGEM_PROMPT = (
    "Você é um triador de Service Desk para políticas internas da empresa Carraro Desenvolvimento. "
    "Dada a mensagem do usuário, retorne SOMENTE um JSON com:\n"
    "{\n"
    '  "decisao": "AUTO_RESOLVER" | "PEDIR_INFO" | "ABRIR_CHAMADO",\n'
    '  "urgencia": "BAIXA" | "MEDIA" | "ALTA",\n'
    '  "campos_faltantes": ["..."]\n'
    "}\n"
    "Regras:\n"
    '- **AUTO_RESOLVER**: Perguntas claras sobre regras ou procedimentos descritos nas políticas (Ex: "Posso reembolsar a internet do meu home office?", "Como funciona a política de alimentação em viagens?").\n'
    '- **PEDIR_INFO**: Mensagens vagas ou que faltam informações para identificar o tema ou contexto (Ex: "Preciso de ajuda com uma política", "Tenho uma dúvida geral").\n'
    '- **ABRIR_CHAMADO**: Pedidos de exceção, liberação, aprovação ou acesso especial, ou quando o usuário explicitamente pede para abrir um chamado (Ex: "Quero exceção para trabalhar 5 dias remoto.", "Solicito liberação para anexos externos.", "Por favor, abra um chamado para o RH.").'
    "Analise a mensagem e decida a ação mais apropriada."
)

class TriagemOutput(BaseModel):
    decisao: Literal["AUTO_RESOLVER", "PEDIR_INFO", "ABRIR_CHAMADO"] = Field(
        description="Decisão tomada com base na mensagem do usuário."
    )
    urgencia: Literal["BAIXA", "MEDIA", "ALTA"] = Field(
        description="Nível de urgência da solicitação."
    )
    campos_faltantes: List[str] = Field(
        default_factory=list,
        description="Lista de campos ou informações que estão faltando na mensagem do usuário."
    )

llm_triagem = ChatGoogleGenerativeAI(temperature=0, 
                             model="gemini-2.5-flash",
                             api_key=GOOGLE_API_KEY)

triagem_chain = llm_triagem.with_structured_output(TriagemOutput)

def triagem(mensagem: str) -> Dict:
    saida: TriagemOutput = triagem_chain.invoke([
            SystemMessage(content=TRIAGEM_PROMPT),
            HumanMessage(content=mensagem)
    ])
    return saida.model_dump()

testes = ["Posso reembolsar a internet?",
          "Quero mais 5 dias de trabalho remoto. Como faço?",
          "Posso reembolsar cursos da Alura?",
          "Quantas capivaras tem no Rio Pinheiros?",]

for msg_teste in testes:
    print(f'Pergunta: {msg_teste}\n -> Resposta {triagem(msg_teste)}')

# AULA 02 ----------------------------------------------
print("\n ---[ AULA 02 ]--- \n")
docs=[]
for n in Path("docs").glob("*.pdf"):
    try:
        loader = PyMuPDFLoader(str(n))
        docs.extend(loader.load())
        print(f"Carregado {n.name}")
    except Exception as e:
        print(f"Erro ao carregar {n.name}: {e}")
        
print(f"Total de documentos carregados: {len(docs)}")
print("\n")
spliter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
chunks = spliter.split_documents(docs)

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=GOOGLE_API_KEY
)

vectorstore = FAISS.from_documents(chunks, embeddings)

retriver = vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"score_threshold": 0.3, "k": 4}
)

prompt_rag = ChatPromptTemplate.from_messages([
    ("system",
     "Você é um Assistente de Políticas Internas (RH/IT) da empresa Carraro Desenvolvimento. "
     "Responda SOMENTE com base no contexto fornecido. "
     "Se não houver base suficiente, responda apenas 'Não sei'."),

    ("human", "Pergunta: {input}\n\nContexto:\n{context}")
])

document_chain = create_stuff_documents_chain(llm_triagem, prompt_rag)

def perguntar_politica_RAG(pergunta: str):
    docs_relationados = retriver.invoke(pergunta)

    if not docs_relationados:
        return {"answer": "Não sei",
                "citacoes":[],
                "contexto_encontrado":False
                }
    answer = document_chain.invoke({
        "input": pergunta,
        "context": docs_relationados
    })

    txt = (answer or "").strip()

    if txt.rstrip(".!?") == "Não sei":
        return {"answer": "Não sei.",
                "citacoes": [],
                "contexto_encontrado": False}

    return {"answer": txt,
            "citacoes": formatadores.formatar_citacoes(docs_relationados,answer),
            "contexto_encontrado": True}

testes = ["Posso reembolsar a internet?",
        "Quero mais 5 dias de trabalho remoto. Como faço?",
        "Posso reembolsar cursos da Alura?",
        "Quantas capivaras tem no Rio Pinheiros?",]

for msg_teste in testes:
    resposta = perguntar_politica_RAG(msg_teste)
    print(f"PERGUNTA: {msg_teste}")
    print(f"RESPOSTA: {resposta['answer']}")
    if resposta["contexto_encontrado"]:
        print("CITAÇÕES:")
        for c in resposta['citacoes']:
            print(f" - Documento: {c['documento']}, Página: {c['pagina']}")
            print(f"   Trecho: {c['trecho']}")
        print("------------------------------------")

