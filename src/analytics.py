# tools.py
import pandas as pd
pd.set_option('display.max_rows', 50) 
pd.set_option('display.max_columns', None)
# Carregando o dataset
df = pd.read_csv("data/sales.csv", sep=";", low_memory=False)
df['date'] = pd.to_datetime(df['date'], dayfirst=True)

from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.lib import colors

# função para formatação dos numeros
def formatar_grandeza(valor):
    if valor >= 1_000_000_000:
        return f"{valor / 1_000_000_000:.2f} Bilhões"
    elif valor >= 1_000_000:
        return f"{valor / 1_000_000:.2f} Milhões"
    elif valor >= 1_000:
        return f"{valor / 1_000:.2f} Mil"
    return str(valor)

# 1 - analise de desempenho de vendas e acuracia de planejamento
def calcular_acuracia_planejamento(df):
    """
    Calcula a diferença percentual entre o planejado e o realizado.
    Útil para identificar erros de previsão de demanda.
    """
    df['variacao_quantidade'] = df['actual_quantity'] - df['planned_quantity']
    df['pct_desvio'] = (df['variacao_quantidade'] / df['planned_quantity']) * 100
    return df[['product_id', 'date', 'planned_quantity', 'actual_quantity', 'pct_desvio']]

def identificar_ruptura_ou_excesso(df, threshold=0.2):
    """
    Identifica casos onde a venda real foi muito abaixo (risco de excesso)
    ou muito acima (risco de ruptura/falta de estoque) do planejado.
    """
    df['razao_real_plan'] = df['actual_quantity'] / df['planned_quantity']
    alertas = df[(df['razao_real_plan'] < (1 - threshold)) | (df['razao_real_plan'] > (1 + threshold))]
    return alertas

#2 - Análise de Impacto de Promoções
def impacto_promocao_por_produto(df):
    """
    Compara o volume médio de vendas e o preço médio em dias com promoção vs dias sem.
    Responde: "A promoção aumentou o volume o suficiente para justificar o preço menor?"
    """
    analise = df.groupby(['product_id', 'promotion_type']).agg({
        'actual_quantity': 'mean',
        'actual_price': 'mean',
        'service_level': 'mean'
    }).reset_index()
    return analise

# 3 - Ranking e Curva ABC (Pareto)
def ranking_receita_por_local(df):
    """
    Calcula a receita real (quantidade * preço real) agrupada por local.
    """
    df['receita_real'] = df['actual_quantity'] * df['actual_price']
    ranking = df.groupby('local')['receita_real'].sum().sort_values(ascending=False)
    return ranking

def produtos_mais_vendidos(df, top_n=10):
    """
    Retorna os N produtos com maior volume de vendas real.
    """
    return df.groupby('product_id')['actual_quantity'].sum().nlargest(top_n)

# 4 - Análise de Nível de Serviço
def analisar_degradacao_servico(df, min_service_level=0.95):
    """
    Filtra transações onde o nível de serviço ficou abaixo da meta.
    Útil para correlacionar se promoções agressivas pioram o nível de serviço.
    """
    return df[df['service_level'] < min_service_level]

# 5 - perguntas do readme
def get_top_performing_entities(df, group_by_col='product_id', metric='actual_quantity', top_n=5):
    """
    Responde: 'Qual produto foi mais vendido?' ou 'Qual local teve maior volume?'
    """
    return df.groupby(group_by_col)[metric].sum().nlargest(top_n).to_dict()

def get_total_sales_period(df, start_date, end_date):
    """
    Responde: 'Qual foi o total de vendas em determinado período?'
    """
    df['date'] = pd.to_datetime(df['date'])
    mask = (df['date'] >= start_date) & (df['date'] <= end_date)
    total = df.loc[mask, 'actual_quantity'].sum()
    return {"periodo": f"{start_date} a {end_date}", "total_vendas": total}

def analyze_planning_gap(df):
    """
    Responde: 'Qual a diferença entre quantidade planejada e realizada?'
    Calcula o Bias (viés) e o MAPE (erro percentual) para o analista.
    """
    df['gap'] = df['actual_quantity'] - df['planned_quantity']
    df['abs_gap_pct'] = (abs(df['gap']) / df['planned_quantity']).replace([float('inf'), -float('inf')], 0)
    
    stats = {
        "gap_total": df['gap'].sum(),
        "mape_medio": f"{df['abs_gap_pct'].mean() * 100:.2f}%",
        "tendencia": "Subestimado" if df['gap'].sum() > 0 else "Superestimado"
    }
    return stats

# 6 - Elasticidade e Promoção
def analyze_promotion_impact(df):
    """
    Compara performance com e sem promoção.
    """
    report = df.groupby('promotion_type').agg({
        'actual_quantity': 'mean',
        'actual_price': 'mean',
        'service_level': 'mean'
    }).rename(columns={
        'actual_quantity': 'media_volume',
        'actual_price': 'preco_medio',
        'service_level': 'nivel_servico_medio'
    })
    return report.to_dict(orient='index')

# 7 - Saúde Logística
def check_service_risk(df, threshold=0.85):
    """
    Identifica locais ou produtos onde o nível de serviço está crítico.
    """
    criticos = df[df['service_level'] < threshold]
    return criticos.groupby(['local', 'product_id'])['service_level'].mean().sort_values().to_dict()




# =========================
# 8) Relatório executivo + PDF
# =========================
def gerar_relatorio_executivo(
    df: pd.DataFrame,
    top_n: int = 5,
    min_service_level: float = 0.95,
    service_risk_threshold: float = 0.85,
) -> str:
    """Gera um relatório executivo (texto) com os principais indicadores do dataset."""
    base = df.copy()

    # Período
    periodo = "N/A"
    if "date" in base.columns:
        base["date"] = pd.to_datetime(base["date"], errors="coerce")
        dt_min = base["date"].min()
        dt_max = base["date"].max()
        if pd.notna(dt_min) and pd.notna(dt_max):
            periodo = f"{dt_min.date()} a {dt_max.date()}"

    # Cobertura
    linhas = len(base)
    produtos = base["product_id"].nunique() if "product_id" in base.columns else None
    locais = base["local"].nunique() if "local" in base.columns else None

    # Volume / Receita
    total_qtd = float(base["actual_quantity"].sum()) if "actual_quantity" in base.columns else None
    total_plan = float(base["planned_quantity"].sum()) if "planned_quantity" in base.columns else None

    total_receita = None
    if {"actual_quantity", "actual_price"}.issubset(base.columns):
        total_receita = float((base["actual_quantity"] * base["actual_price"]).sum())

    # Planejamento
    gap_stats = None
    if {"planned_quantity", "actual_quantity"}.issubset(base.columns):
        gap_stats = analyze_planning_gap(base)

    # Serviço
    service_avg = float(base["service_level"].mean()) if "service_level" in base.columns else None
    service_baixo = int((base["service_level"] < min_service_level).sum()) if "service_level" in base.columns else None

    # Risco serviço (combinações local+produto)
    risk_count = None
    if {"local", "product_id", "service_level"}.issubset(base.columns):
        service_risk = check_service_risk(base, threshold=service_risk_threshold)
        try:
            risk_count = len(service_risk)
        except Exception:
            risk_count = None

    # Top produtos por volume
    top_produtos_txt = "N/A"
    if {"product_id", "actual_quantity"}.issubset(base.columns):
        top_prod = produtos_mais_vendidos(base, top_n=top_n)
        top_produtos_txt = top_prod.to_string(index=False)

    # Top locais por receita
    top_locais_txt = "N/A"
    if {"local", "actual_quantity", "actual_price"}.issubset(base.columns):
        top_loc = ranking_receita_por_local(base).head(top_n)
        top_locais_txt = "\n".join([f"- {idx}: {formatar_grandeza(val)}" for idx, val in top_loc.items()])

    # Promoção (amostra)
    promo_txt = "N/A"
    if "promotion_type" in base.columns:
        promo = analyze_promotion_impact(base)
        if isinstance(promo, dict) and promo:
            linhas_promo = []
            for k in list(promo.keys())[:5]:
                v = promo[k]
                if isinstance(v, dict):
                    mv = v.get("media_volume")
                    pm = v.get("preco_medio")
                    ns = v.get("nivel_servico_medio")
                    if isinstance(mv, (int, float)) and isinstance(pm, (int, float)) and isinstance(ns, (int, float)):
                        linhas_promo.append(
                            f"- {k}: media_volume={mv:.2f} | preco_medio={pm:.2f} | nivel_servico_medio={ns:.3f}"
                        )
                    else:
                        linhas_promo.append(f"- {k}: {v}")
                else:
                    linhas_promo.append(f"- {k}: {v}")
            promo_txt = "\n".join(linhas_promo)

    def fmt(x):
        if x is None:
            return "N/A"
        try:
            return formatar_grandeza(float(x))
        except Exception:
            return str(x)

    out = []
    out.append("RELATÓRIO EXECUTIVO (Dataset de Vendas)")
    out.append(f"Período: {periodo}")
    out.append(
        f"Cobertura: {linhas} linhas | produtos únicos: {produtos if produtos is not None else 'N/A'} | locais únicos: {locais if locais is not None else 'N/A'}"
    )
    out.append("")
    out.append("1) Volume & Receita")
    out.append(f"- Total vendido (actual_quantity): {fmt(total_qtd)}")
    if total_plan is not None:
        out.append(f"- Total planejado (planned_quantity): {fmt(total_plan)}")
    if total_receita is not None:
        out.append(f"- Receita total estimada (qtd * preço): {fmt(total_receita)}")
    out.append("")
    out.append("2) Planejamento")
    if isinstance(gap_stats, dict) and gap_stats:
        out.append(f"- Gap total (actual - planned): {fmt(gap_stats.get('gap_total'))}")
        out.append(f"- MAPE médio: {gap_stats.get('mape_medio')}")
        out.append(f"- Tendência: {gap_stats.get('tendencia')}")
    else:
        out.append("- N/A (colunas de planejamento não disponíveis)")
    out.append("")
    out.append("3) Nível de serviço")
    if service_avg is not None:
        out.append(f"- Service level médio: {service_avg:.3f}")
        if service_baixo is not None:
            out.append(f"- Linhas abaixo de {min_service_level:.2f}: {service_baixo}")
        if risk_count is not None:
            out.append(f"- Combinações local+produto abaixo de {service_risk_threshold:.2f}: {risk_count}")
    else:
        out.append("- N/A (coluna service_level não disponível)")
    out.append("")
    out.append(f"4) Top {top_n} produtos por volume")
    out.append(top_produtos_txt)
    out.append("")
    out.append(f"5) Top {top_n} locais por receita")
    out.append(top_locais_txt)
    out.append("")
    out.append("6) Promoções (amostra)")
    out.append(promo_txt)

    return "\n".join(out)


def salvar_relatorio_pdf(relatorio_texto: str, output_path: str) -> str:
    """Salva o texto do relatório em PDF e retorna o caminho."""


    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(
        str(out),
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
        title="Relatório Executivo",
    )

    styles = getSampleStyleSheet()
    title_style = styles["Title"]
    mono_style = ParagraphStyle(
        "Mono",
        parent=styles["BodyText"],
        fontName="Courier",
        fontSize=9,
        leading=11,
        alignment=TA_LEFT,
        textColor=colors.black,
    )

    story = []

    linhas = relatorio_texto.splitlines()
    if linhas:
        story.append(Paragraph(linhas[0], title_style))
        story.append(Spacer(1, 0.4 * cm))
        resto = linhas[1:]
    else:
        resto = []

    # corpo (preserva tabelas do to_string)
    for bloco in "\n".join(resto).split("\n\n"):
        bloco = bloco.strip()
        if not bloco:
            continue
        safe = bloco.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        story.append(Paragraph(safe.replace("\n", "<br/>"), mono_style))
        story.append(Spacer(1, 0.35 * cm))

    doc.build(story)
    return str(out)


def gerar_relatorio_pdf(
    df: pd.DataFrame,
    output_path: str = "reports/relatorio_executivo.pdf",
    top_n: int = 5,
) -> str:
    """Gera o relatório executivo e salva em PDF."""
    texto = gerar_relatorio_executivo(df, top_n=top_n)
    return salvar_relatorio_pdf(texto, output_path)
