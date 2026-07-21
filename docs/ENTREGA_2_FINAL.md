# Classificação de Lesões Dermatológicas — Entrega 2

## Resumo

O pipeline classificou 10.015 imagens HAM10000 em sete classes, sem vazamento
entre lesões. A EfficientNet-B3 ajustada foi o melhor modelo: AUC-ROC melanoma
0,909, AUC macro 0,959, F1 macro 0,704 e acurácia balanceada 0,749. Todas as
metas foram superadas.

## Dados e pré-processamento

- Split 70%/15%/15%, estratificado e agrupado por `lesion_id`.
- Augmentation somente no treino: crop, flips, rotação e ColorJitter.
- Normalização ImageNet e resize determinístico na validação/teste.
- Desbalanceamento padrão tratado por loss ponderada.

## Modelos efetivamente avaliados

- ResNet18 baseline, 20 épocas.
- EfficientNet-B3 original, early stopping na época 10.
- EfficientNet-B3 ajustada: LR 1e-4, weight decay 5e-4, dropout 0,3,
  unfreezing gradual e checkpoint por AUC melanoma.
- Ablação ResNet18: loss ponderada, sampler ponderado e focal loss.

## Resultados principais

ResNet18:

- AUC melanoma 0,908.
- AUC macro 0,957.
- F1 macro 0,676.
- Acurácia balanceada 0,730.

EfficientNet-B3 ajustada:

- AUC melanoma 0,909.
- AUC macro 0,959.
- F1 macro 0,704.
- Acurácia balanceada 0,749.
- Acurácia global 0,791.

IC 95% por bootstrap agrupado para a EfficientNet-B3:

- AUC melanoma: 0,875–0,927.
- AUC macro: 0,950–0,968.
- F1 macro: 0,651–0,744.
- Acurácia balanceada: 0,696–0,799.

## Validação cruzada

A ResNet18 obteve, em cinco folds agrupados:

- AUC melanoma 0,881 ± 0,015.
- AUC macro 0,931 ± 0,012.
- F1 macro 0,599 ± 0,030.
- Acurácia balanceada 0,683 ± 0,047.

Todos os folds superaram a meta de AUC melanoma 0,85.

## Calibração e limiar

Temperature scaling reduziu ECE de 0,096 para 0,030 e Brier de 0,315 para
0,298. O limiar selecionado somente na validação produziu no teste:

- Sensibilidade de melanoma 0,933.
- Especificidade 0,693.
- Precisão 0,276.
- 11 falsos negativos e 404 falsos positivos.

O resultado é compatível com triagem sensível, não com diagnóstico autônomo.

## Ablação

O sampler ponderado apresentou melhor F1 macro (0,676) e acurácia balanceada
(0,715). A focal loss testada degradou F1 macro para 0,448 e não foi adotada.

## Análise de erros

A sensibilidade foi 0,929 em homens e 0,943 em mulheres. Por idade, variou de
0,667 em menores de 40 anos a 0,975 entre 60 e 79 anos. O primeiro grupo contém
apenas nove melanomas, portanto a estimativa é incerta.

Foram preparados 26 casos críticos com Grad-CAM++ para revisão cega. A revisão
por dermatologista não foi executada e consta como trabalho futuro.

## Hipótese

A hipótese foi confirmada após tuning. A EfficientNet-B3 ajustada superou a
ResNet18 por margem pequena em AUC melanoma, mas apresentou vantagens maiores
em F1 macro e acurácia balanceada. O intervalo de confiança não constitui teste
pareado de superioridade entre modelos.

## Limitações

- Ausência de validação externa.
- Ausência de revisão por especialista.
- Predomínio de pele clara e ausência de fototipo.
- Poucas amostras de classes raras.
- Avaliação retrospectiva.
- Grad-CAM não demonstra causalidade.

## Reprodutibilidade

- Python 3.14.3.
- PyTorch 2.13.0+cu126.
- NVIDIA Quadro P6000 24 GB.
- Semente 42.
- Checkpoint final: `models/experiments/efficientnet_b3_tuned/best.pt`.
- Relatórios: `reports/experiments/` e `reports/final/`.
- Protocolos pendentes: `docs/protocols/`.

O relatório diagramado está em
`../Entrega-2-Classificacao-Lesoes-Dermatologicas.pdf`.
