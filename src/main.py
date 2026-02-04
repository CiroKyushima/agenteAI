import asyncio
from llama_index.core.workflow import Context
from agent import get_agent

agent = get_agent()

# ✅ Contexto mantém histórico/sessão entre perguntas
ctx = Context(agent)

async def ask(pergunta: str) -> str:
    # ReActAgent workflow usa run()
    handler = agent.run(
        pergunta,
        ctx=ctx,
        max_iterations=60,
        early_stopping_method="generate",
    )  
    response = await handler
            
    return str(response)

print("🤖 Chat iniciado! Digite 'sair' para encerrar.\n")

async def main_loop():
    while True:
        pergunta = input("Você: ").strip()

        if pergunta.lower() == "sair":
            print("Encerrando...")
            break

        if not pergunta:
            continue

        try:
            resposta = await ask(pergunta)
            print(f"\nGPT: {resposta}\n")
        except Exception as e:
            print(f"\nErro ao processar pergunta: {e}\n")

if __name__ == "__main__":
    try:
        asyncio.run(main_loop())
    except RuntimeError:
        # Fallback para ambientes com event loop já rodando
        loop = asyncio.get_event_loop()
        loop.run_until_complete(main_loop())
