# Agent-Based Sales Data Analyzer
## Visão Geral
Este projeto implementa um Agente de IA orientado a dados, projetado para analisar e sintetizar informações a partir de um dataset estruturado de vendas (sales.csv). O sistema combina raciocínio de modelos de linguagem com ferramentas analíticas determinísticas, garantindo que as respostas sejam baseadas em cálculos reais e não em suposições geradas pelo modelo. O agente atua como um Analista de Dados de Vendas Automatizado, capaz de interpretar métricas de desempenho, planejamento, promoções, nível de serviço e risco operacional.

## caracteristicas 
* 📊 Análise de vendas baseada em CSV com cálculos reais via ferramentas
* 🧠 Arquitetura ReAct Agent com tomada de decisão orientada a tools
* 📄 Geração automática de Relatório Executivo em PDF
* 🎯 Analise de ferramentas analiticas utilizadas
* 🔍 Consulta inteligente para perguntas complexas sobre o dataset
* 🐳 Aplicação totalmente executável via Docker
* 🏗️ Estrutura modular com separação entre agente, tools e analytics

## Stack
* **LLM:** OpenAI GPT-4o-mini (via LlamaIndex)
* **Arquitetura de Agente:** ReActAgent (LlamaIndex)
* **Engine de Dados:** Pandas
* **Consulta Estruturada:** PandasQueryEngine
* **Geração de PDF:** ReportLab
* **Containerização:** Docker
* **Gestão de Ambiente:** Python-dotenv

## Fluxo de Funcionamento
* Pergunta do usuário é enviada ao agente
* ReAct Agent interpreta a intenção
* Seleção automática de ferramenta adequada
* Execução de cálculo real em Pandas
* Resultado estruturado é retornado
* Resposta em linguagem natural é gerada
#### Se necessário, o agente usa uma consulta genérica apartir do dataset para análises não previstas.

## como executar o projeto
##### é necessario uma key da openAI
#### instalação manual:
```bash
# 1️⃣ Clonar o repositório
git clone https://github.com/CiroKyushima/Agent-Based-Sales-Data-Analyzer
cd Agent-Based-Sales-Data-Analyzer

# 2️⃣ Instalar as dependências
pip install -r requirements.txt

# 3️⃣ Criar arquivo .env na pasta do projeto
coloque: OPENAI_API_KEY=sua_chave_aqui

#4️⃣ Executar o projeto:
python src/main.py
```
#### instalação via DOCKER:
```bash
# 1️⃣ Clonar o repositório
git clone https://github.com/CiroKyushima/Agent-Based-Sales-Data-Analyzer
cd Agent-Based-Sales-Data-Analyzer

# 2️⃣ Criar arquivo .env na pasta do projeto
coloque: OPENAI_API_KEY=sua_chave_aqui

# 3️⃣ Build da imagem
docker compose build

# 5️⃣ Rodar o container
docker run --env-file .env -it {nome_da_imagem}
```

