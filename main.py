import myconfig
from langchain_google_genai import ChatGoogleGenerativeAI

from pydantic import BaseModel, Field
from typing import Literal, List, Dict

from langchain_core.messages import SystemMessage, HumanMessage

GOOGLE_API_KEY = myconfig.key_api # Contem o API Key do Google Gemini, arquivo local myconfig.py

print(f'🤿 Olá, Imersão Dev Agentes de IA!')
print("\n")
llm = ChatGoogleGenerativeAI(temperature=0, 
                             model="gemini-2.5-flash",
                             api_key=GOOGLE_API_KEY)

# response = llm.invoke("Quem é vc? Seja imprevisível e criativo na resposta.")
# print(response.content)
print("\n")
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

