
from pydantic import BaseModel, Field
from typing import Literal, List

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