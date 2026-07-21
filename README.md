# Classificação de lesões dermatológicas

MVP acadêmico de classificação das sete classes do HAM10000 com ResNet18,
EfficientNet-B3, métricas multiclasse e explicações Grad-CAM/Grad-CAM++.

> Este projeto é experimental e não deve ser usado para diagnóstico médico.
> O HAM10000 tem limitações de representatividade, especialmente para fototipos
> de pele pouco presentes na base.

## Modelos e classes

- Baseline: ResNet18 pré-treinada no ImageNet, entrada 224 × 224.
- Modelo principal: EfficientNet-B3 pré-treinada, entrada 300 × 300.
- Classes: `akiec`, `bcc`, `bkl`, `df`, `mel`, `nv` e `vasc`.

O split é feito em 70%/15%/15% sobre `lesion_id` únicos e estratificado por
diagnóstico. Portanto, imagens da mesma lesão nunca aparecem em conjuntos
diferentes.

## Instalação

Requer Python 3.10 ou superior.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Para CUDA, instale a distribuição do PyTorch compatível com a sua GPU conforme
as instruções oficiais antes das demais dependências.

## Dados

Copie `.env.example` para `.env` e informe suas credenciais Kaggle, ou configure
`~/.kaggle/kaggle.json`.

```powershell
Copy-Item .env.example .env
python download_data.py
python pipeline.py prepare
```

O download deve resultar em 10.015 imagens. O comando `prepare` valida os
arquivos e grava `artifacts/splits.csv`.

## Treinamento

```powershell
python pipeline.py train --model resnet18
python pipeline.py train --model efficientnet_b3
```

Por padrão, a loss usa pesos inversamente proporcionais às frequências. Para
testar amostragem ponderada sem combinar as duas estratégias:

```powershell
python pipeline.py train --model resnet18 --imbalance-strategy sampler
```

Checkpoints e históricos são gravados em `models/<modelo>/`.

## Avaliação e Grad-CAM

```powershell
python pipeline.py evaluate --model resnet18
python pipeline.py explain --model resnet18 --method both --max-images 12
```

As métricas, predições, matrizes de confusão, curvas ROC/PR e mapas explicativos
ficam em `reports/<modelo>/`. As metas do projeto são:

- AUC-ROC melanoma ≥ 0,85
- AUC-ROC macro ≥ 0,80
- F1 macro ≥ 0,55
- Acurácia balanceada ≥ 0,65

Para preparar os dados, treinar ambos os modelos, avaliar e explicar:

```powershell
python pipeline.py all --epochs 20
```

## Validação final da Entrega 2

O fluxo avançado é retomável: checkpoints existentes são reutilizados.

```powershell
python final_validation.py advanced --device cuda
python final_validation.py cv --device cuda
python final_validation.py ablation --device cuda
python final_validation.py efficientnet --device cuda
python error_analysis.py
python final_report.py
```

Os comandos executam bootstrap agrupado, calibração, limiar de melanoma,
validação cruzada, ablação, tuning da EfficientNet-B3 e análise de subgrupos.
O PDF final é salvo um nível acima do repositório como
`Entrega-2-Classificacao-Lesoes-Dermatologicas.pdf`.

## Testes

Os testes não precisam do HAM10000 nem de pesos baixados:

```powershell
pytest -q
```

## Artefatos

- `artifacts/splits.csv`: índice das imagens e splits.
- `models/<modelo>/best.pt`: melhor checkpoint por loss de validação.
- `models/<modelo>/history.json`: histórico por época.
- `reports/<modelo>/metrics.json`: métricas e metas atingidas.
- `reports/<modelo>/predictions.csv`: probabilidades por imagem.
- `reports/<modelo>/*.png`: matriz de confusão e curvas.
- `reports/<modelo>/explanations/`: Grad-CAM e Grad-CAM++.
- `models/experiments/`: checkpoints da validação final.
- `reports/experiments/`: métricas de folds, ablações e modelo ajustado.
- `reports/final/`: gráficos consolidados e análise por subgrupos.
- `docs/protocols/`: validação externa e revisão clínica ainda pendentes.
