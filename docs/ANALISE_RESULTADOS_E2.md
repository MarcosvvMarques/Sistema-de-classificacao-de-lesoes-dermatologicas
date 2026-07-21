# Análise dos resultados para a Entrega 2

## Síntese executiva

Os dois modelos superaram todas as metas definidas na Entrega 1. Entretanto, a
hipótese de que a EfficientNet-B3 superaria a ResNet18 não foi confirmada neste
experimento. A ResNet18 apresentou os melhores resultados agregados e melhor
capacidade de identificar melanoma.

Resultados da ResNet18 no teste:

- AUC-ROC melanoma: 0,908 (meta: 0,85).
- AUC-ROC macro: 0,957 (meta: 0,80).
- F1 macro: 0,676 (meta: 0,55).
- Acurácia balanceada: 0,730 (meta: 0,65).
- Acurácia global: 0,783.

Resultados da EfficientNet-B3 no teste:

- AUC-ROC melanoma: 0,886.
- AUC-ROC macro: 0,948.
- F1 macro: 0,613.
- Acurácia balanceada: 0,704.
- Acurácia global: 0,752.

## Evidências para incluir no documento

### Seção 2.5 — Pré-processamento aplicado

O pipeline indexou 10.015 imagens e criou splits de 70%/15%/15% no nível de
`lesion_id`, estratificados por diagnóstico. Nenhuma lesão aparece em mais de
um split. As imagens de treino receberam crop redimensionado aleatório,
espelhamentos, rotação e variações moderadas de cor. Validação e teste usam
resize determinístico. Todas as entradas foram normalizadas com estatísticas
do ImageNet.

O desbalanceamento foi tratado com pesos inversamente proporcionais à
frequência das classes na `CrossEntropyLoss`. O conjunto de teste manteve a
distribuição natural: 992 das 1.481 imagens são nevos.

### Seção 3.4 — Pipeline

Fluxo efetivamente executado:

1. Download e validação do HAM10000.
2. Indexação de imagens e metadados.
3. Split estratificado por lesão.
4. Augmentation e normalização.
5. Transfer learning com pesos ImageNet.
6. Seleção do checkpoint pela menor loss de validação.
7. Avaliação no teste mantido isolado.
8. Geração de Grad-CAM e Grad-CAM++.

### Seção 3.5 — Modelos comparados

A ResNet18 foi treinada por 20 épocas com batch 128. A EfficientNet-B3 usou
batch 64 e foi encerrada por early stopping na época 10. Ambas usaram AdamW,
learning rate inicial de 3e-4, weight decay de 1e-4, scheduler por plateau,
AMP e fine-tuning completo após uma época de congelamento do backbone.

### Seção 5.2 — Gráficos

Inserir as figuras:

- `reports/comparison/training_curves.png`.
- `reports/comparison/model_comparison.png`.
- `reports/comparison/per_class_f1.png`.
- `reports/resnet18/confusion_matrix.png`.
- `reports/resnet18/roc_curves.png`.
- `reports/resnet18/pr_curves.png`.
- Exemplos de `reports/resnet18/explanations/`.

### Seção 5.3 — Análise de erros

Na ResNet18, melanoma obteve recall de 0,703, mas precisão de apenas 0,464.
Isso indica uma quantidade relevante de falsos positivos, comportamento que
pode ser aceitável em triagem, mas exige avaliação do custo clínico e ajuste
de limiar. A EfficientNet-B3 teve recall de melanoma inferior, 0,570.

As classes minoritárias têm estimativas mais incertas. Dermatofibroma e lesão
vascular possuem apenas 20 e 19 exemplos no teste. O F1 da EfficientNet-B3
para dermatofibroma foi 0,378. Resultados altos nessas classes não devem ser
interpretados sem intervalos de confiança.

A EfficientNet-B3 mostrou sobreajuste mais intenso: na época 10, alcançou
94,7% no treino e 78,1% na validação, enquanto a loss de validação já havia
atingido seu melhor valor na época 5. A ResNet18 apresentou convergência mais
estável e menor diferença entre treino e validação.

### Seção 5.4 — Comparação com a hipótese

A hipótese foi parcialmente confirmada. A meta absoluta de AUC-ROC de melanoma
foi superada pelos dois modelos, mas a EfficientNet-B3 não superou o baseline.
A ResNet18 obteve vantagem de 0,022 em AUC-ROC de melanoma, 0,063 em F1 macro
e 0,026 em acurácia balanceada. Portanto, os resultados favorecem a arquitetura
mais simples para este protocolo.

### Seção 6.2 — Limitações

- Uso de uma única divisão treino/validação/teste.
- Ausência de intervalos de confiança e teste estatístico entre modelos.
- Poucos exemplos de classes raras no teste.
- Dataset majoritariamente composto por peles claras.
- Ausência de metadados de fototipo.
- Avaliação retrospectiva, sem validação clínica ou externa.
- Grad-CAM oferece evidência visual, mas não prova raciocínio causal.
- O sistema não deve ser usado como diagnóstico autônomo.

## Melhorias prioritárias para novos experimentos

1. Aplicar validação cruzada estratificada e agrupada por `lesion_id`.
2. Calcular intervalos de confiança por bootstrap para AUC, F1 e sensibilidade.
3. Otimizar o limiar de melanoma conforme uma meta explícita de sensibilidade.
4. Avaliar calibração com reliability diagram, ECE e Brier score.
5. Comparar loss ponderada, sampler ponderado e focal loss em ablation study.
6. Ajustar a EfficientNet-B3 com learning rate menor, unfreezing gradual e
   regularização mais forte.
7. Selecionar checkpoints por AUC macro ou AUC de melanoma, não apenas por loss.
8. Analisar falsos positivos e falsos negativos por idade, sexo e localização.
9. Validar em uma base externa e mais diversa antes de qualquer uso aplicado.
10. Fazer avaliação qualitativa dos mapas Grad-CAM com especialista.

## Reprodutibilidade a registrar

- Semente global: 42.
- Hardware: NVIDIA Quadro P6000 de 24 GB.
- PyTorch: 2.13.0 com CUDA 12.6.
- Split: 70% treino, 15% validação e 15% teste por `lesion_id`.
- ResNet18: 20 épocas; melhor loss de validação na época 18.
- EfficientNet-B3: early stopping na época 10; melhor loss na época 5.
