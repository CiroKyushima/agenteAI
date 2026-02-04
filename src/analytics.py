import pandas as pd
import numpy as np
pd.set_option('display.max_rows', 50) 
pd.set_option('display.max_columns', None)
df = pd.read_csv("data/sales.csv", sep=";", low_memory=False)
df['date'] = pd.to_datetime(df['date'], dayfirst=True)

from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.lib import colors

def formatar_grandeza(valor):
    if valor >= 1_000_000_000:
        return f"{valor / 1_000_000_000:.2f} Bilhões"
    elif valor >= 1_000_000:
        return f"{valor / 1_000_000:.2f} Milhões"
    elif valor >= 1_000:
        return f"{valor / 1_000:.2f} Mil"
    return str(valor)

# =========================
# Helpers
# =========================
def _safe_div(numer, denom, fill_value=np.nan):
    """Divisão segura (evita inf/NaN por divisão por zero)."""
    denom = denom.replace(0, np.nan)
    out = numer / denom
    return out.fillna(fill_value)


# =========================
# 1) Acurácia de planejamento
# =========================
def calcular_acuracia_planejamento(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula a diferença percentual entre o planejado e o realizado.
    Corrigido: evita divisão por zero (planned_quantity = 0).
    """
    base = df.copy()
    base["variacao_quantidade"] = base["actual_quantity"] - base["planned_quantity"]


    base["pct_desvio"] = np.where(
        base["planned_quantity"] > 0,
        (base["variacao_quantidade"] / base["planned_quantity"]) * 100,
        np.nan,
    )

    return base[["product_id", "date", "planned_quantity", "actual_quantity", "pct_desvio"]]


def identificar_ruptura_ou_excesso(df: pd.DataFrame, threshold: float = 0.2) -> pd.DataFrame:
    """
    Identifica casos onde a venda real foi muito abaixo (risco de excesso)
    ou muito acima (risco de ruptura/falta de estoque) do planejado.
    Corrigido: planned_quantity = 0 vira NaN e não entra em alerta por razão.
    """
    base = df.copy()

    base["razao_real_plan"] = np.where(
        base["planned_quantity"] > 0,
        base["actual_quantity"] / base["planned_quantity"],
        np.nan,
    )

    alertas = base[
        (base["razao_real_plan"] < (1 - threshold)) | (base["razao_real_plan"] > (1 + threshold))
    ]
    return alertas


# =========================
# 2) Impacto de promoções
# =========================
def impacto_promocao_por_produto(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compara volume médio e preço médio com promoção vs sem promoção, por produto.
    Corrigido: trata NaN como 'Sem Promo' e entrega também o delta % (promo vs sem).
    """
    base = df.copy()
    base["promo_flag"] = np.where(base["promotion_type"].notna(), "Com Promo", "Sem Promo")

    agg = (
        base.groupby(["product_id", "promo_flag"])
        .agg(
            media_volume=("actual_quantity", "mean"),
            preco_medio=("actual_price", "mean"),
            nivel_servico_medio=("service_level", "mean"),
            n_linhas=("actual_quantity", "size"),
        )
        .reset_index()
    )

    piv_vol = agg.pivot(index="product_id", columns="promo_flag", values="media_volume")
    piv_pre = agg.pivot(index="product_id", columns="promo_flag", values="preco_medio")

    delta = pd.DataFrame({"product_id": piv_vol.index})
    delta["delta_volume_%"] = np.where(
        piv_vol.get("Sem Promo").notna() & (piv_vol.get("Sem Promo") != 0) & piv_vol.get("Com Promo").notna(),
        (piv_vol["Com Promo"] / piv_vol["Sem Promo"] - 1) * 100,
        np.nan,
    )
    delta["delta_preco_%"] = np.where(
        piv_pre.get("Sem Promo").notna() & (piv_pre.get("Sem Promo") != 0) & piv_pre.get("Com Promo").notna(),
        (piv_pre["Com Promo"] / piv_pre["Sem Promo"] - 1) * 100,
        np.nan,
    )

    out = agg.merge(delta, on="product_id", how="left")
    return out


# =========================
# 3) Ranking
# =========================
def ranking_receita_por_local(df: pd.DataFrame) -> pd.Series:
    """
    Calcula a receita real (quantidade * preço real) agrupada por local.
    (Mantido) só evitando mutação do df original.
    """
    base = df.copy()
    base["receita_real"] = base["actual_quantity"] * base["actual_price"]
    ranking = base.groupby("local")["receita_real"].sum().sort_values(ascending=False)
    return ranking


def produtos_mais_vendidos(df: pd.DataFrame, top_n: int = 10) -> pd.Series:
    """Retorna os N produtos com maior volume de vendas real."""
    return df.groupby("product_id")["actual_quantity"].sum().nlargest(top_n)


# =========================
# 4) Nível de serviço 
# =========================
def analisar_degradacao_servico(df: pd.DataFrame, min_service_level: float = 0.95) -> pd.DataFrame:
    """Filtra transações onde o nível de serviço ficou abaixo da meta."""
    return df[df["service_level"] < min_service_level].copy()


# =========================
# 5) Readme - top entities 
# =========================
def get_top_performing_entities(
    df: pd.DataFrame, group_by_col: str = "product_id", metric: str = "actual_quantity", top_n: int = 5
) -> dict:
    """Responde: 'Qual produto foi mais vendido?' ou 'Qual local teve maior volume?'"""
    return df.groupby(group_by_col)[metric].sum().nlargest(top_n).to_dict()


def get_total_sales_period(df: pd.DataFrame, start_date, end_date, metric: str = "revenue") -> dict:
    """
    Responde: 'Qual foi o total de vendas em determinado período?'
    Corrigido: por padrão retorna REVENUE (receita). Se quiser volume, passe metric="volume".
    """
    base = df.copy()
    base["date"] = pd.to_datetime(base["date"],dayfirst=True, errors="coerce")
    mask = (base["date"] >= pd.to_datetime(start_date)) & (base["date"] <= pd.to_datetime(end_date))

    if metric == "volume":
        total = float(base.loc[mask, "actual_quantity"].sum())
        return {"periodo": f"{start_date} a {end_date}", "total_volume": total}

    # metric == "revenue"
    total = float((base.loc[mask, "actual_quantity"] * base.loc[mask, "actual_price"]).sum())
    return {"periodo": f"{start_date} a {end_date}", "total_receita": total}


def analyze_planning_gap(df: pd.DataFrame) -> dict:
    """
    Responde: 'Qual a diferença entre quantidade planejada e realizada?'
    Corrigido: MAPE só considera linhas com planned_quantity > 0 (não zera infinito).
    """
    base = df.copy()
    base["gap"] = base["actual_quantity"] - base["planned_quantity"]

    validos = base[base["planned_quantity"] > 0].copy()
    if len(validos) > 0:
        validos["abs_gap_pct"] = (validos["gap"].abs() / validos["planned_quantity"])
        mape_medio = validos["abs_gap_pct"].mean() * 100
    else:
        mape_medio = np.nan

    gap_total = float(base["gap"].sum())

    stats = {
        "gap_total": gap_total,
        "mape_medio": f"{mape_medio:.2f}%" if pd.notna(mape_medio) else "N/A",
        "tendencia": "Subestimado" if gap_total > 0 else "Superestimado",
        "linhas_validas_mape": int(len(validos)),
    }
    return stats


# =========================
# 6) Promo impact
# =========================
def analyze_promotion_impact(df: pd.DataFrame) -> dict:
    """
    Compara performance com e sem promoção.
    Corrigido: cria promo_flag (Com Promo / Sem Promo) e calcula deltas percentuais.
    """
    base = df.copy()
    base["promo_flag"] = np.where(base["promotion_type"].notna(), "Com Promo", "Sem Promo")

    agg = (
        base.groupby("promo_flag")
        .agg(
            media_volume=("actual_quantity", "mean"),
            preco_medio=("actual_price", "mean"),
            nivel_servico_medio=("service_level", "mean"),
            n_linhas=("actual_quantity", "size"),
        )
        .to_dict(orient="index")
    )

    try:
        sem = agg["Sem Promo"]
        com = agg["Com Promo"]

        def _delta_pct(a, b):
    
            if b is None or b == 0 or a is None:
                return None
            return (a / b - 1) * 100

        agg["delta_com_vs_sem"] = {
            "delta_volume_%": _delta_pct(com["media_volume"], sem["media_volume"]),
            "delta_preco_%": _delta_pct(com["preco_medio"], sem["preco_medio"]),
            "delta_service_level_%": _delta_pct(com["nivel_servico_medio"], sem["nivel_servico_medio"]),
        }
    except KeyError:
        agg["delta_com_vs_sem"] = {}

    return agg

def get_promocao_share(df: pd.DataFrame) -> dict:
    """
    Retorna qual % das vendas ocorreu com promoção.
    IMPORTANTE: os campos *_pct já estão em PERCENTUAL (0 a 100).
    Também devolve versões formatadas em string com '%', para evitar o LLM multiplicar de novo.
    """
    base = df.copy()

    promo_mask = base["promotion_type"].notna()

    base["actual_quantity"] = pd.to_numeric(base["actual_quantity"], errors="coerce").fillna(0)
    base["actual_price"] = pd.to_numeric(base["actual_price"], errors="coerce").fillna(0)
    base["receita"] = base["actual_quantity"] * base["actual_price"]

    total_linhas = len(base)
    total_volume = float(base["actual_quantity"].sum())
    total_receita = float(base["receita"].sum())

    promo_linhas = int(promo_mask.sum())
    promo_volume = float(base.loc[promo_mask, "actual_quantity"].sum())
    promo_receita = float(base.loc[promo_mask, "receita"].sum())

    share_linhas_pct = (promo_linhas / total_linhas * 100) if total_linhas else 0.0
    share_volume_pct = (promo_volume / total_volume * 100) if total_volume else 0.0
    share_receita_pct = (promo_receita / total_receita * 100) if total_receita else 0.0

    return {
        "linhas_com_promo": promo_linhas,
        "linhas_total": total_linhas,

        # números (já em %)
        "share_linhas_pct": share_linhas_pct,
        "share_volume_pct": share_volume_pct,
        "share_receita_pct": share_receita_pct,

        # strings (para o agente só “copiar e colar”)
        "share_linhas_fmt": f"{share_linhas_pct:.4f}%",
        "share_volume_fmt": f"{share_volume_pct:.4f}%",
        "share_receita_fmt": f"{share_receita_pct:.4f}%",
    }


def get_preco_medio_geral(df: pd.DataFrame) -> dict:
    """
    Retorna o preço médio geral (actual_price).
    Corrigido: garante conversão para número e ignora NaN.
    """
    base = df.copy()
    base["actual_price"] = pd.to_numeric(base["actual_price"], errors="coerce")

    preco_medio = float(base["actual_price"].mean())
    return {"preco_medio_geral": round(preco_medio, 2)}


def get_produto_maior_receita(df: pd.DataFrame) -> dict:
    """
    Retorna o produto com maior receita total.
    Receita = soma(actual_quantity * actual_price) por produto.
    Corrigido: NÃO confunde com 'mais vendido' e NÃO usa preço fixo.
    """
    base = df.copy()
    base["actual_quantity"] = pd.to_numeric(base["actual_quantity"], errors="coerce").fillna(0)
    base["actual_price"] = pd.to_numeric(base["actual_price"], errors="coerce").fillna(0)

    base["receita"] = base["actual_quantity"] * base["actual_price"]

    receita_por_produto = base.groupby("product_id")["receita"].sum().sort_values(ascending=False)

    top_produto = str(receita_por_produto.index[0])
    top_receita = float(receita_por_produto.iloc[0])

    return {"product_id": top_produto, "receita_total": top_receita}

# =========================
# 7) Service risk
# =========================
def check_service_risk(df: pd.DataFrame, threshold: float = 0.85) -> dict:
    """
    Identifica combinações local+produto onde o nível de serviço médio está crítico.
    Corrigido: calcula média por (local, produto) e só então filtra < threshold.
    """
    base = df.copy()
    medias = (
        base.groupby(["local", "product_id"])["service_level"]
        .mean()
        .sort_values()
    )
    criticos = medias[medias < threshold]
    return criticos.to_dict()



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

        top_produtos_txt = "\n".join(
            [f"- {idx}: {formatar_grandeza(val)}" for idx, val in top_prod.items()]
        )

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

     # Promo share (%)
    promo_share_txt = "N/A"
    if "promotion_type" in base.columns:
        promo_share = get_promocao_share(base)
        promo_share_txt = (
            f"- % linhas com promoção: {promo_share['share_linhas_fmt']}\n"
            f"- % volume com promoção: {promo_share['share_volume_fmt']}\n"
            f"- % receita com promoção: {promo_share['share_receita_fmt']}"
)

    # Preço médio geral
    preco_medio_txt = "N/A"
    if "actual_price" in base.columns:
        preco_medio_txt = str(get_preco_medio_geral(base).get("preco_medio_geral"))

    # Produto maior receita
    top_receita_txt = "N/A"
    if {"product_id", "actual_quantity", "actual_price"}.issubset(base.columns):
        tr = get_produto_maior_receita(base)
        top_receita_txt = f"{tr['product_id']} | receita={formatar_grandeza(tr['receita_total'])}"

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

    out.append("")
    out.append("7) Promoções (share)")
    out.append(promo_share_txt)

    out.append("")
    out.append("8) Preço médio geral")
    out.append(f"- actual_price médio: {preco_medio_txt}")

    out.append("")
    out.append("9) Produto com maior receita")
    out.append(f"- {top_receita_txt}")

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
