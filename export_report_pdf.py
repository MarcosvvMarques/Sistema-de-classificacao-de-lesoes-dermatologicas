"""Exporta a análise visual dos resultados como PDF."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / "reports"
DEFAULT_OUTPUT = Path.home() / "Downloads" / "analise-resultados-dermatologia.pdf"


def _summary_page(pdf: PdfPages) -> None:
    fig = plt.figure(figsize=(11.69, 8.27))
    fig.suptitle(
        "Resultados do classificador dermatológico",
        fontsize=22,
        fontweight="bold",
        y=0.94,
    )
    fig.text(
        0.08,
        0.86,
        "HAM10000 · ResNet18 versus EfficientNet-B3",
        fontsize=13,
        color="dimgray",
    )
    fig.text(
        0.08,
        0.76,
        "Conclusão principal",
        fontsize=15,
        fontweight="bold",
    )
    fig.text(
        0.08,
        0.69,
        "Os dois modelos superaram todas as metas, mas a hipótese comparativa\n"
        "não foi confirmada: a ResNet18 teve melhor desempenho agregado.",
        fontsize=13,
        linespacing=1.5,
    )

    metrics = [
        ("AUC-ROC melanoma", "0,908", "0,886", "0,850"),
        ("AUC-ROC macro", "0,957", "0,948", "0,800"),
        ("F1 macro", "0,676", "0,613", "0,550"),
        ("Acurácia balanceada", "0,730", "0,704", "0,650"),
    ]
    axis = fig.add_axes((0.08, 0.31, 0.84, 0.30))
    axis.axis("off")
    table = axis.table(
        cellText=metrics,
        colLabels=("Métrica", "ResNet18", "EfficientNet-B3", "Meta"),
        cellLoc="center",
        colLoc="center",
        bbox=(0, 0, 1, 1),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.6)

    fig.text(
        0.08,
        0.20,
        "Melanoma — ResNet18: recall 70,3% e precisão 46,4%.",
        fontsize=12,
        fontweight="bold",
    )
    fig.text(
        0.08,
        0.14,
        "Interpretação: maior sensibilidade para triagem, acompanhada de falsos positivos.\n"
        "Uso clínico exige calibração, definição de limiar e validação externa.",
        fontsize=11,
        linespacing=1.4,
    )
    pdf.savefig(fig)
    plt.close(fig)


def _image_page(pdf: PdfPages, image_path: Path, title: str, caption: str) -> None:
    image = plt.imread(image_path)
    fig = plt.figure(figsize=(11.69, 8.27))
    fig.suptitle(title, fontsize=18, fontweight="bold", y=0.96)
    axis = fig.add_axes((0.04, 0.10, 0.92, 0.80))
    axis.imshow(image)
    axis.axis("off")
    fig.text(0.05, 0.045, caption, fontsize=9, color="dimgray")
    pdf.savefig(fig)
    plt.close(fig)


def _recommendations_page(pdf: PdfPages) -> None:
    fig = plt.figure(figsize=(11.69, 8.27))
    fig.suptitle(
        "Melhorias recomendadas para a Entrega 2",
        fontsize=20,
        fontweight="bold",
        y=0.94,
    )
    recommendations = [
        "1. Calcular intervalos de confiança por bootstrap para AUC, F1 e sensibilidade.",
        "2. Aplicar validação cruzada estratificada e agrupada por lesion_id.",
        "3. Definir limiar de melanoma conforme uma meta explícita de sensibilidade.",
        "4. Avaliar calibração com reliability diagram, ECE e Brier score.",
        "5. Comparar loss ponderada, sampler ponderado e focal loss.",
        "6. Ajustar a EfficientNet-B3 com learning rate menor e unfreezing gradual.",
        "7. Selecionar checkpoint por AUC de melanoma ou AUC macro.",
        "8. Analisar falsos positivos e negativos por idade, sexo e localização.",
        "9. Validar em uma base externa e mais diversa.",
        "10. Revisar mapas Grad-CAM com especialista.",
    ]
    y = 0.84
    for recommendation in recommendations:
        fig.text(0.08, y, recommendation, fontsize=12)
        y -= 0.065

    fig.text(
        0.08,
        0.14,
        "Limitações a registrar",
        fontsize=14,
        fontweight="bold",
    )
    fig.text(
        0.08,
        0.075,
        "Split único; poucas amostras raras; predomínio de peles claras; ausência de fototipo;\n"
        "avaliação retrospectiva; Grad-CAM não demonstra causalidade; sem validação clínica.",
        fontsize=11,
        linespacing=1.4,
    )
    pdf.savefig(fig)
    plt.close(fig)


def export_pdf(output_path: Path = DEFAULT_OUTPUT) -> Path:
    comparison = REPORTS / "comparison"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(output_path) as pdf:
        _summary_page(pdf)
        _image_page(
            pdf,
            comparison / "model_comparison.png",
            "Comparação das métricas no teste",
            "Fonte: reports/*/metrics.json · teste com 1.481 imagens.",
        )
        _image_page(
            pdf,
            comparison / "training_curves.png",
            "Convergência e generalização",
            "Fonte: models/*/history.json · ResNet18: 20 épocas; EfficientNet-B3: early stopping na época 10.",
        )
        _image_page(
            pdf,
            comparison / "per_class_f1.png",
            "F1-score por classe",
            "Classes raras têm suporte reduzido: dermatofibroma=20 e vascular=19 no teste.",
        )
        _recommendations_page(pdf)
    return output_path


if __name__ == "__main__":
    print(export_pdf())
