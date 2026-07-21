"""Gera gráficos e o PDF acadêmico final da Entrega 2."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parent
FINAL_DIR = ROOT / "reports" / "final"
OUTPUT_PDF = ROOT.parent / "Entrega-2-Classificacao-Lesoes-Dermatologicas.pdf"


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


BASE_RESNET = load_json(ROOT / "reports" / "resnet18" / "metrics.json")
BASE_EFFICIENT = load_json(ROOT / "reports" / "efficientnet_b3" / "metrics.json")
TUNED = load_json(
    ROOT / "reports" / "experiments" / "efficientnet_b3_tuned" / "metrics.json"
)
ADVANCED = load_json(
    ROOT
    / "reports"
    / "experiments"
    / "efficientnet_b3_tuned"
    / "advanced"
    / "advanced_metrics.json"
)
CV = load_json(
    ROOT / "reports" / "experiments" / "cv_resnet18" / "summary.json"
)
ABLATION = load_json(
    ROOT / "reports" / "experiments" / "ablation_resnet18" / "summary.json"
)

METRICS = (
    "auc_roc_melanoma",
    "auc_roc_macro",
    "f1_macro",
    "balanced_accuracy",
)
METRIC_LABELS = (
    "AUC melanoma",
    "AUC macro",
    "F1 macro",
    "Acurácia balanceada",
)


def generate_figures() -> None:
    FINAL_DIR.mkdir(parents=True, exist_ok=True)

    models = {
        "ResNet18": BASE_RESNET,
        "EfficientNet-B3 original": BASE_EFFICIENT,
        "EfficientNet-B3 ajustada": TUNED,
    }
    positions = np.arange(len(METRICS))
    width = 0.25
    fig, axis = plt.subplots(figsize=(11, 6))
    for index, (label, result) in enumerate(models.items()):
        values = [100 * result[name] for name in METRICS]
        bars = axis.bar(positions + (index - 1) * width, values, width, label=label)
        axis.bar_label(bars, fmt="%.1f", fontsize=8, padding=2)
    axis.set_xticks(positions, METRIC_LABELS)
    axis.set_ylabel("Pontuação (%)")
    axis.set_ylim(0, 105)
    axis.set_title("Comparação final no conjunto de teste")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FINAL_DIR / "final_model_comparison.png", dpi=180)
    plt.close(fig)

    intervals = ADVANCED["bootstrap_95_ci"]
    estimates = np.array([100 * intervals[name]["estimate"] for name in METRICS])
    lower = estimates - np.array([100 * intervals[name]["lower_95"] for name in METRICS])
    upper = np.array([100 * intervals[name]["upper_95"] for name in METRICS]) - estimates
    fig, axis = plt.subplots(figsize=(10, 6))
    axis.errorbar(
        np.arange(len(METRICS)),
        estimates,
        yerr=np.vstack((lower, upper)),
        fmt="o",
        capsize=6,
        linewidth=2,
    )
    axis.set_xticks(np.arange(len(METRICS)), METRIC_LABELS)
    axis.set_ylabel("Pontuação (%)")
    axis.set_ylim(50, 102)
    axis.set_title("EfficientNet-B3 ajustada — IC 95% por bootstrap agrupado")
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FINAL_DIR / "bootstrap_confidence_intervals.png", dpi=180)
    plt.close(fig)

    folds = CV["folds"]
    fig, axis = plt.subplots(figsize=(10, 6))
    fold_positions = np.arange(1, 6)
    axis.bar(
        fold_positions - 0.18,
        [100 * fold["auc_roc_melanoma"] for fold in folds],
        0.36,
        label="AUC melanoma",
    )
    axis.bar(
        fold_positions + 0.18,
        [100 * fold["f1_macro"] for fold in folds],
        0.36,
        label="F1 macro",
    )
    axis.axhline(85, linestyle="--", color="black", linewidth=1, label="Meta AUC melanoma")
    axis.set(xlabel="Fold", ylabel="Pontuação (%)", title="Validação cruzada agrupada — ResNet18")
    axis.set_xticks(fold_positions)
    axis.set_ylim(0, 100)
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FINAL_DIR / "cross_validation.png", dpi=180)
    plt.close(fig)

    strategies = {
        "Loss ponderada": ABLATION["weighted_loss"],
        "Sampler ponderado": ABLATION["weighted_sampler"],
        "Focal loss": ABLATION["focal_loss"],
    }
    positions = np.arange(len(METRICS))
    width = 0.25
    fig, axis = plt.subplots(figsize=(11, 6))
    for index, (label, result) in enumerate(strategies.items()):
        values = [100 * result[name] for name in METRICS]
        axis.bar(positions + (index - 1) * width, values, width, label=label)
    axis.set_xticks(positions, METRIC_LABELS)
    axis.set_ylabel("Pontuação (%)")
    axis.set_ylim(0, 105)
    axis.set_title("Ablation study — tratamento do desbalanceamento")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FINAL_DIR / "imbalance_ablation.png", dpi=180)
    plt.close(fig)


def page_number(canvas, document) -> None:
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#555555"))
    canvas.drawString(2 * cm, 1.1 * cm, "Classificação de Lesões Dermatológicas — Entrega 2")
    canvas.drawRightString(19 * cm, 1.1 * cm, f"Página {document.page}")
    canvas.restoreState()


def build_styles():
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="ReportTitle",
            parent=styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=20,
            leading=24,
            alignment=TA_CENTER,
            spaceAfter=14,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Section",
            parent=styles["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=15,
            leading=18,
            textColor=colors.HexColor("#163A5F"),
            spaceBefore=10,
            spaceAfter=8,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Subsection",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=11.5,
            leading=14,
            spaceBefore=8,
            spaceAfter=5,
        )
    )
    styles.add(
        ParagraphStyle(
            name="BodyJustified",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=9.5,
            leading=13,
            alignment=TA_JUSTIFY,
            spaceAfter=6,
        )
    )
    styles.add(
        ParagraphStyle(
            name="SmallCaption",
            parent=styles["BodyText"],
            fontName="Helvetica-Oblique",
            fontSize=8,
            leading=10,
            textColor=colors.HexColor("#555555"),
            alignment=TA_CENTER,
            spaceAfter=8,
        )
    )
    return styles


def styled_table(data, widths=None, font_size=8):
    table = Table(data, colWidths=widths, repeatRows=1, hAlign="CENTER")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#DCE6F1")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#163A5F")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), font_size),
                ("LEADING", (0, 0), (-1, -1), font_size + 2),
                ("ALIGN", (1, 1), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#AAB7C4")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F6F8FA")]),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    return table


def add_image(story, path: Path, caption: str, styles, width=16.5 * cm):
    story.append(Image(str(path), width=width, height=width * 0.56))
    story.append(Paragraph(caption, styles["SmallCaption"]))


def bullet(story, text: str, styles):
    story.append(Paragraph(f"• {text}", styles["BodyJustified"]))


def generate_pdf() -> Path:
    styles = build_styles()
    document = SimpleDocTemplate(
        str(OUTPUT_PDF),
        pagesize=A4,
        rightMargin=1.8 * cm,
        leftMargin=1.8 * cm,
        topMargin=1.6 * cm,
        bottomMargin=1.8 * cm,
        title="Classificação de Lesões Dermatológicas — Entrega 2",
        author="Equipe Residência em TIC 44",
    )
    story = []

    story.extend(
        [
            Spacer(1, 1.7 * cm),
            Paragraph("Classificação de Lesões Dermatológicas com Deep Learning", styles["ReportTitle"]),
            Paragraph("Entrega 2 — Relatório Final", styles["Title"]),
            Spacer(1, 0.7 * cm),
            Paragraph(
                "Residência em TIC 44 (CTE-IA) · Projeto de Inteligência Artificial",
                styles["BodyJustified"],
            ),
            Spacer(1, 0.8 * cm),
            styled_table(
                [
                    ["Integrante", "Papel principal"],
                    ["Marcos Vinicius de Vasconcelos Marques", "Liderança"],
                    ["Bruno Takazono", "Dados e apresentação"],
                    ["Marcelo de Araujo", "Modelagem"],
                    ["Douglas Leonard", "Engenharia"],
                    ["Todos", "Avaliação e documentação"],
                ],
                widths=[10.5 * cm, 5.5 * cm],
                font_size=9,
            ),
            Spacer(1, 1.0 * cm),
            Paragraph(
                "<b>Resultado central:</b> a EfficientNet-B3 ajustada alcançou AUC-ROC de "
                "melanoma 0,909, AUC macro 0,959, F1 macro 0,704 e acurácia balanceada "
                "0,749, superando todas as metas e o baseline ResNet18.",
                styles["BodyJustified"],
            ),
            Spacer(1, 2.0 * cm),
            Paragraph("21 de julho de 2026", styles["SmallCaption"]),
            PageBreak(),
        ]
    )

    story.append(Paragraph("1. Introdução", styles["Section"]))
    story.append(Paragraph("1.1 Contexto e motivação", styles["Subsection"]))
    story.append(
        Paragraph(
            "O câncer de pele é o câncer mais frequente no Brasil. A detecção precoce do "
            "melanoma está associada a prognóstico substancialmente melhor. Este projeto "
            "investiga classificação automatizada de imagens dermatoscópicas como apoio "
            "à triagem, com explicações visuais Grad-CAM. O sistema é experimental e não "
            "substitui avaliação dermatológica.",
            styles["BodyJustified"],
        )
    )
    story.append(Paragraph("1.2 Pergunta e hipótese", styles["Subsection"]))
    story.append(
        Paragraph(
            "Pergunta: é possível atingir AUC-ROC de melanoma igual ou superior a 0,85 "
            "no HAM10000? A hipótese previa que a EfficientNet-B3 ajustada superaria a "
            "ResNet18. Após tuning controlado, a hipótese comparativa foi confirmada por "
            "margem pequena em AUC e maior vantagem em F1 macro.",
            styles["BodyJustified"],
        )
    )
    story.append(Paragraph("1.3 Objetivos realizados", styles["Subsection"]))
    for text in (
        "Pipeline reproduzível de aquisição, validação, split e pré-processamento.",
        "Comparação entre ResNet18 e EfficientNet-B3 com transfer learning.",
        "Avaliação por métricas robustas ao desbalanceamento e IC 95%.",
        "Calibração, limiar de triagem, análise por subgrupos e Grad-CAM++.",
        "Validação cruzada e ablation study das estratégias de desbalanceamento.",
    ):
        bullet(story, text, styles)

    story.append(Paragraph("2. Dados", styles["Section"]))
    story.append(
        Paragraph(
            "Foi utilizado o HAM10000, sob licença CC BY-NC 4.0, com 10.015 imagens JPEG "
            "e metadados clínicos anonimizados. As sete classes são: akiec, bcc, bkl, df, "
            "mel, nv e vasc. Nevos representam 6.705 imagens (67%), enquanto df e vasc "
            "somam menos de 3%, caracterizando forte desbalanceamento.",
            styles["BodyJustified"],
        )
    )
    story.append(Paragraph("2.1 Split e prevenção de vazamento", styles["Subsection"]))
    story.append(
        Paragraph(
            "A divisão 70%/15%/15% foi estratificada por diagnóstico no nível de "
            "lesion_id. Todas as imagens da mesma lesão permanecem no mesmo conjunto. "
            "O teste, com 1.481 imagens, foi mantido isolado durante treinamento, "
            "calibração, seleção de checkpoint e ajuste do limiar.",
            styles["BodyJustified"],
        )
    )
    story.append(Paragraph("2.2 Pré-processamento", styles["Subsection"]))
    story.append(
        Paragraph(
            "Treino: RandomResizedCrop, espelhamentos horizontal e vertical, rotação e "
            "ColorJitter moderado. Validação/teste: resize determinístico. Todas as "
            "imagens foram normalizadas com estatísticas ImageNet. O tratamento padrão "
            "usou CrossEntropyLoss com pesos inversos à frequência.",
            styles["BodyJustified"],
        )
    )
    story.append(PageBreak())

    story.append(Paragraph("3. Metodologia", styles["Section"]))
    method_data = [
        ["Componente", "Configuração final"],
        ["Framework", "PyTorch 2.13.0 + CUDA 12.6; timm"],
        ["Hardware", "NVIDIA Quadro P6000, 24 GB"],
        ["ResNet18", "224×224; AdamW; LR 3e-4; 20 épocas"],
        ["EfficientNet-B3 final", "300×300; LR 1e-4; WD 5e-4; dropout 0,3"],
        ["Unfreezing final", "classificador → blocos finais → backbone completo"],
        ["Seleção final", "maior AUC-ROC de melanoma na validação"],
        ["Semente", "42"],
    ]
    story.append(styled_table(method_data, widths=[5 * cm, 11 * cm]))
    story.append(Paragraph("3.1 Validação estatística", styles["Subsection"]))
    story.append(
        Paragraph(
            "Foram executados 2.000 reamostramentos bootstrap no nível de lesion_id para "
            "IC 95%. A ResNet18 também foi avaliada por StratifiedGroupKFold de cinco "
            "folds no conjunto de desenvolvimento. O teste original não participou dos folds.",
            styles["BodyJustified"],
        )
    )
    story.append(Paragraph("3.2 Calibração e limiar", styles["Subsection"]))
    story.append(
        Paragraph(
            "Temperature scaling foi ajustado apenas na validação. O limiar de melanoma "
            "foi escolhido para sensibilidade alvo de 90% na validação e então congelado "
            "para o teste. A calibração foi avaliada por ECE, Brier score e reliability diagram.",
            styles["BodyJustified"],
        )
    )
    story.append(Paragraph("3.3 Ablation study", styles["Subsection"]))
    story.append(
        Paragraph(
            "Foram comparados loss ponderada, WeightedRandomSampler e focal loss, com a "
            "mesma arquitetura ResNet18, split, augmentations e orçamento de dez épocas. "
            "Isso isola o efeito da estratégia de desbalanceamento.",
            styles["BodyJustified"],
        )
    )

    story.append(Paragraph("4. Resultados", styles["Section"]))
    results_table = [["Modelo", *METRIC_LABELS]]
    for label, result in (
        ("ResNet18", BASE_RESNET),
        ("EfficientNet-B3 original", BASE_EFFICIENT),
        ("EfficientNet-B3 ajustada", TUNED),
    ):
        results_table.append([label, *[f"{result[name]:.3f}" for name in METRICS]])
    results_table.append(["Meta", "0,850", "0,800", "0,550", "0,650"])
    story.append(styled_table(results_table, widths=[4.2 * cm] + [3 * cm] * 4, font_size=7.5))
    add_image(
        story,
        FINAL_DIR / "final_model_comparison.png",
        "Figura 1 — Comparação final no conjunto de teste.",
        styles,
    )
    story.append(PageBreak())

    story.append(Paragraph("4.1 Intervalos de confiança", styles["Subsection"]))
    ci_rows = [["Métrica", "Estimativa", "IC 95% inferior", "IC 95% superior"]]
    for label, name in zip(METRIC_LABELS, METRICS):
        interval = ADVANCED["bootstrap_95_ci"][name]
        ci_rows.append(
            [
                label,
                f"{interval['estimate']:.3f}",
                f"{interval['lower_95']:.3f}",
                f"{interval['upper_95']:.3f}",
            ]
        )
    story.append(styled_table(ci_rows, widths=[5 * cm, 3.5 * cm, 3.5 * cm, 3.5 * cm]))
    add_image(
        story,
        FINAL_DIR / "bootstrap_confidence_intervals.png",
        "Figura 2 — IC 95% por bootstrap agrupado, 2.000 reamostragens.",
        styles,
    )

    story.append(Paragraph("4.2 Validação cruzada", styles["Subsection"]))
    story.append(
        Paragraph(
            f"A ResNet18 obteve AUC melanoma média {CV['mean']['auc_roc_melanoma']:.3f} "
            f"± {CV['std']['auc_roc_melanoma']:.3f}; AUC macro "
            f"{CV['mean']['auc_roc_macro']:.3f} ± {CV['std']['auc_roc_macro']:.3f}; "
            f"F1 macro {CV['mean']['f1_macro']:.3f} ± {CV['std']['f1_macro']:.3f}. "
            "Todos os folds superaram a meta de AUC de melanoma.",
            styles["BodyJustified"],
        )
    )
    add_image(
        story,
        FINAL_DIR / "cross_validation.png",
        "Figura 3 — Resultados por fold com agrupamento por lesion_id.",
        styles,
    )
    story.append(PageBreak())

    story.append(Paragraph("4.3 Desbalanceamento", styles["Subsection"]))
    ablation_rows = [["Estratégia", *METRIC_LABELS]]
    for label, key in (
        ("Loss ponderada", "weighted_loss"),
        ("Sampler ponderado", "weighted_sampler"),
        ("Focal loss", "focal_loss"),
    ):
        result = ABLATION[key]
        ablation_rows.append([label, *[f"{result[name]:.3f}" for name in METRICS]])
    story.append(styled_table(ablation_rows, widths=[4.2 * cm] + [3 * cm] * 4, font_size=7.5))
    story.append(
        Paragraph(
            "O sampler ponderado produziu o melhor F1 macro (0,676) e acurácia balanceada "
            "(0,715). A focal loss, com a parametrização testada, elevou recall de melanoma "
            "mas degradou fortemente precisão, F1 macro e acurácia, não sendo recomendada.",
            styles["BodyJustified"],
        )
    )
    add_image(
        story,
        FINAL_DIR / "imbalance_ablation.png",
        "Figura 4 — Ablation study com orçamento de treinamento controlado.",
        styles,
    )

    story.append(Paragraph("4.4 Calibração e limiar de triagem", styles["Subsection"]))
    calibration = ADVANCED["calibration"]
    threshold = ADVANCED["threshold_test"]
    story.append(
        Paragraph(
            f"Temperature scaling reduziu ECE de {calibration['ece_before']:.3f} para "
            f"{calibration['ece_after']:.3f} e Brier de {calibration['brier_before']:.3f} "
            f"para {calibration['brier_after']:.3f}. No limiar congelado, a sensibilidade "
            f"de melanoma foi {threshold['sensitivity']:.3f}, especificidade "
            f"{threshold['specificity']:.3f} e precisão {threshold['precision']:.3f}. "
            "A alta sensibilidade vem acompanhada por 404 falsos positivos, reforçando o "
            "caráter de triagem e não de diagnóstico.",
            styles["BodyJustified"],
        )
    )
    add_image(
        story,
        ROOT / "reports" / "experiments" / "efficientnet_b3_tuned" / "advanced" / "calibration.png",
        "Figura 5 — Reliability diagram de melanoma antes e após calibração.",
        styles,
    )
    story.append(PageBreak())

    story.append(Paragraph("5. Análise de erros e equidade", styles["Section"]))
    story.append(
        Paragraph(
            "Com o limiar de triagem, a sensibilidade foi 0,929 em homens e 0,943 em "
            "mulheres. Por idade, variou de 0,667 em menores de 40 anos (apenas nove "
            "melanomas) a 0,975 entre 60 e 79 anos. Esses resultados são descritivos; "
            "subgrupos pequenos têm grande incerteza e não sustentam alegações de equidade.",
            styles["BodyJustified"],
        )
    )
    add_image(
        story,
        FINAL_DIR / "subgroup_performance.png",
        "Figura 6 — Sensibilidade e especificidade por sexo e faixa etária.",
        styles,
    )
    story.append(
        Paragraph(
            "Foram separados 26 casos críticos (todos os 11 falsos negativos e os 15 "
            "falsos positivos mais confiantes) com Grad-CAM++. A ficha cega foi gerada, "
            "mas não preenchida: revisão por especialista não é apresentada como concluída.",
            styles["BodyJustified"],
        )
    )

    story.append(Paragraph("6. Discussão", styles["Section"]))
    story.append(Paragraph("6.1 Comparação com a hipótese", styles["Subsection"]))
    story.append(
        Paragraph(
            "A hipótese foi confirmada após tuning. A EfficientNet-B3 original não superou "
            "a ResNet18, mas a configuração com LR menor, regularização adicional, "
            "unfreezing gradual e checkpoint por AUC melanoma alcançou AUC 0,909 contra "
            "0,908 da ResNet18. A diferença de AUC é pequena e não demonstra superioridade "
            "estatística; a vantagem mais relevante ocorreu em F1 macro (0,704 contra 0,676) "
            "e acurácia balanceada (0,749 contra 0,730).",
            styles["BodyJustified"],
        )
    )
    story.append(Paragraph("6.2 Limitações", styles["Subsection"]))
    for text in (
        "HAM10000 representa majoritariamente pessoas de pele clara e não contém fototipo.",
        "Classes raras possuem poucos exemplos, especialmente no teste.",
        "Não houve validação externa por ausência de segunda base independente.",
        "A revisão Grad-CAM por dermatologista foi preparada, mas não executada.",
        "Ajustes e comparações compartilham o mesmo teste; os ICs quantificam amostragem, não todos os vieses.",
        "Grad-CAM é pós-hoc e não comprova causalidade ou validade clínica.",
    ):
        bullet(story, text, styles)
    story.append(PageBreak())

    story.append(Paragraph("7. Ética, validação externa e uso pretendido", styles["Section"]))
    story.append(
        Paragraph(
            "O modelo deve ser interpretado como protótipo acadêmico de triagem. Não deve "
            "emitir diagnóstico, orientar tratamento ou ser aplicado em populações novas "
            "sem validação. Falsos negativos podem atrasar atendimento; falsos positivos "
            "podem causar ansiedade e procedimentos desnecessários.",
            styles["BodyJustified"],
        )
    )
    story.append(
        Paragraph(
            "Foi preparado um protocolo congelado de validação externa: checkpoint, "
            "calibração e limiar não poderão ser ajustados na base externa; a análise "
            "deverá usar bootstrap por paciente e estratificação por centro e fototipo. "
            "Essa etapa permanece trabalho futuro.",
            styles["BodyJustified"],
        )
    )

    story.append(Paragraph("8. Conclusão", styles["Section"]))
    for text in (
        "A meta de AUC-ROC de melanoma foi superada de forma consistente.",
        "A EfficientNet-B3 ajustada foi o melhor modelo agregado.",
        "A validação cruzada confirmou estabilidade razoável da ResNet18.",
        "Temperature scaling melhorou substancialmente a calibração.",
        "O limiar de triagem atingiu 93,3% de sensibilidade, com baixa precisão.",
        "Sampler ponderado foi preferível à focal loss nesta ablação.",
    ):
        bullet(story, text, styles)
    story.append(
        Paragraph(
            "Com mais seis semanas, as prioridades são validação externa multicêntrica, "
            "revisão cega por dermatologistas, coleta de fototipo, teste estatístico entre "
            "modelos e estudo prospectivo de fluxo clínico.",
            styles["BodyJustified"],
        )
    )

    story.append(Paragraph("9. Reprodutibilidade", styles["Section"]))
    reproducibility = [
        ["Item", "Valor"],
        ["Python", "3.14.3"],
        ["PyTorch", "2.13.0+cu126"],
        ["GPU", "NVIDIA Quadro P6000 24 GB"],
        ["Semente", "42"],
        ["Dados", "HAM10000, 10.015 imagens"],
        ["Split", "70/15/15 por lesion_id"],
        ["Checkpoint final", "models/experiments/efficientnet_b3_tuned/best.pt"],
        ["Métricas finais", "reports/experiments/efficientnet_b3_tuned/"],
        ["Protocolos pendentes", "docs/protocols/"],
    ]
    story.append(styled_table(reproducibility, widths=[5 * cm, 11 * cm], font_size=8.5))
    story.append(Spacer(1, 0.3 * cm))
    story.append(
        Paragraph(
            "Execução principal: <font name='Courier'>python final_validation.py all --device cuda</font>. "
            "Geração deste relatório: <font name='Courier'>python final_report.py</font>.",
            styles["BodyJustified"],
        )
    )

    document.build(story, onFirstPage=page_number, onLaterPages=page_number)
    return OUTPUT_PDF


def main() -> None:
    generate_figures()
    print(generate_pdf())


if __name__ == "__main__":
    main()
