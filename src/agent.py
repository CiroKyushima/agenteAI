import os
import logging
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("llama_index").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("openai._base_client").setLevel(logging.WARNING)
from dotenv import load_dotenv
from llama_index.llms.openai import OpenAI
from llama_index.core import Settings

load_dotenv()

# ReActAgent import varia por versão -> fallback
try:
    from llama_index.core.agent.workflow import ReActAgent
except Exception:
    from llama_index.core.agent import ReActAgent

from agent_tools import TOOLS



def get_agent():
    # LlamaIndex usa Settings.llm (padrão)
    Settings.llm = OpenAI(
        model="gpt-4o-mini",
        api_key=os.getenv("OPENAI_API_KEY"),
        temperature=0.1,
    )
    
    system_prompt = """
Você é um Analista de IA Sênior especializado em vendas.

Regras:
- Seja objetivo e claro, em português.
- Se o usuário perguntar qual ferramenta foi utilizada, você DEVE informar o nome exato da função que chamou (ex: tool_produtos_mais_vendidos).
- Use ferramentas específicas para cálculos comuns.
- Se a pergunta for complexa ou envolver cruzamentos de dados não previstos, use a ferramenta 'consulta_geral'.
- ao usar a ferramenta 'consulta_geral', indique que foi por ela digitando [IA] logo no inicio da resposta
- Você tem acesso total aos dados do arquivo sales_clean.csv através dessas ferramentas.
- Não invente números.
- Use a ferramenta 'processar_e_limpar_vendas' se o usuário pedir para organizar ou limpar a base.
- Use a 'consulta_geral' para cálculos e perguntas sobre o conteúdo.
- Sempre confirme quando uma limpeza for realizada com sucesso.
""".strip()

    agent = ReActAgent(
        tools=TOOLS,
        llm=Settings.llm,
        system_prompt=system_prompt,
    )

    return agent
