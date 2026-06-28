# Classificação de Lesões Dermatológicas com Deep Learning

> **Módulo 7 — Projetos de IA · PPI/SiDi · Abr–Jul/2026**
>
> Transfer learning EfficientNet-B3 vs. ResNet18 no HAM10000 com explicabilidade via Grad-CAM.

---

## Índice

1. [Introdução](#introdução)
2. [Dados](#dados)
3. [Metodologia](#metodologia)
4. [Resultados](#resultados)
5. [Conclusão](#conclusão)
6. [Reprodutibilidade](#reprodutibilidade)

---

## Introdução

### Problema

Classificar imagens dermatoscópicas em 7 categorias de lesões de pele — melanoma, nevo, carcinoma basocelular, ceratose actínica, ceratose benigna, dermatofibroma e lesão vascular — para apoio ao diagnóstico precoce, especialmente relevante em regiões brasileiras com escassez de dermatologistas.

**Pergunta de pesquisa:** É possível atingir AUC-ROC ≥ 0,85 para melanoma com EfficientNet-B3 fine-tuned sobre o HAM10000, superando o baseline ResNet18?

**Hipótese:** Transfer learning com EfficientNet-B3 + balanceamento de classes atingirá AUC-ROC ≥ 0,85 para melanoma e F1-macro ≥ 0,55, superando o ResNet18 baseline.

### Contexto e motivação

O câncer de pele é o tipo mais comum de câncer no Brasil. Melanomas detectados em estágio inicial têm sobrevida em 5 anos superior a 98%; em estágio avançado, menos de 30%. Modelos de deep learning já demonstraram desempenho comparável a dermatologistas especializados (Esteva et al., *Nature* 2017). Este projeto constrói um classificador auditável: cada predição vem acompanhada de um mapa Grad-CAM que evidencia quais regiões anatômicas da imagem motivaram a decisão.

### Objetivos

1. Construir pipeline reprodutível de download, organização e pré-processamento do HAM10000.
2. Conduzir EDA rigorosa identificando desbalanceamento e vieses.
3. Treinar ResNet18 (baseline) e EfficientNet-B3 (modelo principal) com comparação de métricas.
4. Implementar GradCAM e GradCAM++ para explicabilidade das predições.
5. Documentar o projeto de forma reprodutível end-to-end.

---

## Dados

### Fonte e licença

| Campo | Valor |
|-------|-------|
| Dataset | HAM10000 (Human Against Machine with 10000 training images) |
| Fonte | Kaggle: `kmader/skin-cancer-mnist-ham10000` |
| Licença | **CC BY-NC 4.0** — uso acadêmico permitido, uso comercial proibido |
| Acesso | Requer conta Kaggle + credenciais (ver `.env.example`) |

### Volume e formato

| Campo | Valor |
|-------|-------|
| Imagens totais | 10.015 |
| Lesões únicas | ~7.470 |
| Formato | JPEG, RGB, 600×450 px originais |
| Tamanho | ~3 GB |
| Classes | 7 (mel, nv, bcc, akiec, bkl, df, vasc) |

### Variáveis principais

| Variável | Tipo | Descrição |
|----------|------|-----------|
| `image_id` | string | ID único da imagem ISIC |
| `lesion_id` | string | ID da lesão (base do split patient-level) |
| `dx` | categórica | Diagnóstico: mel / nv / bcc / akiec / bkl / df / vasc |
| `dx_type` | categórica | Método: histo, follow_up, consensus, confocal |
| `age` | numérica | Idade do paciente |
| `sex` | categórica | Sexo: male / female |
| `localization` | categórica | Localização da lesão no corpo |

### Desbalanceamento e riscos

O dataset é severamente desbalanceado: a classe `nv` (nevo) representa ~67% das imagens; `df` e `vasc` somam menos de 2%. Estratégias de mitigação:

- **Split:** ao nível de `lesion_id` (sem leakage entre splits)
- **Treino:** `WeightedRandomSampler` para balancear batches
- **Loss:** `CrossEntropyLoss` com pesos de classe inversamente proporcionais à frequência

**Viés de representatividade:** HAM10000 é composto majoritariamente por pacientes de pele clara (europeu). O modelo não deve ser usado em populações com fototipos muito distintos sem validação adicional.

### Pré-processamento

**Treino:**
- Resize 224×224
- Flip horizontal e vertical (p=0,5)
- Rotação aleatória 90° (p=0,5)
- ShiftScaleRotate (shift=0.1, scale=0.15, rotate=45°, p=0,5)
- Distorção elástica/grid/ótica (p=0,3)
- Ruído gaussiano / blur (p=0,3)
- ColorJitter (brightness, contrast, saturation, hue)
- Normalização ImageNet (mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])

**Validação / Teste:**
- Resize 224×224
- Normalização ImageNet

---

## Metodologia

### Abordagem

Classificação supervisionada multiclasse com transfer learning. Pesos pré-treinados no ImageNet são fine-tuned sobre o HAM10000. O protocolo de validação é estratificado ao nível de `lesion_id` para evitar data leakage.

### Stack técnica

| Componente | Ferramenta |
|------------|-----------|
| Linguagem | Python 3.10+ |
| Deep learning | PyTorch ≥ 2.1, timm ≥ 0.9 |
| Augmentation | Albumentations ≥ 1.3 |
| Métricas | scikit-learn |
| Visualização | matplotlib, seaborn, OpenCV |
| Reprodutibilidade | semente global = 42, requirements.txt |

### Baselines

| Modelo | Descrição |
|--------|-----------|
| Majoritário | Sempre prediz `nv`; AUC-ROC = 0,50 |
| **ResNet18** | ResNet-18 ImageNet pré-treinado, fine-tuned com mesmo pipeline |

### Pipeline

```
download_data.py         # Kaggle → data/
    ↓
src/prepare_dataset.py   # data/ → data/organized/{train,val,test}/{classe}/
    ↓
src/eda.py               # EDA → docs/figs/
    ↓
src/train.py             # train → checkpoints/
    ↓
src/evaluate.py          # test → outputs/evaluation/
    ↓
src/compare_models.py    # comparação → outputs/comparison/
    ↓
src/gradcam.py           # Grad-CAM++ → outputs/gradcam/
```

### Modelos comparados

| Modelo | Parâmetros | Backbone | Observação |
|--------|-----------|----------|------------|
| ResNet18 | ~11 M | ResNet-18 | Baseline da especificação |
| EfficientNet-B3 | ~12 M | EfficientNet-B3 | Modelo principal |

**Hiperparâmetros comuns:**
- Epochs: 50 (com early stopping, patience=10)
- Batch size: 32
- Optimizer: AdamW (lr=1e-4, weight_decay=1e-4)
- Scheduler: CosineAnnealingLR (T_max=50, eta_min=1e-6)
- Mixed precision: AMP (autocast + GradScaler)

### Protocolo de validação

| Parâmetro | Valor |
|-----------|-------|
| Split | 70% treino / 15% validação / 15% teste |
| Estratificação | Por classe, ao nível de `lesion_id` |
| Semente global | 42 |
| Vazamento | Impossível — lesão em apenas 1 split |

### Métricas e critérios de sucesso

| Métrica | Meta mínima |
|---------|-------------|
| AUC-ROC melanoma | **≥ 0,85** |
| AUC-ROC macro | ≥ 0,80 |
| F1-macro | ≥ 0,55 |
| Acurácia balanceada | ≥ 0,65 |

---

## Resultados

> [TODO Entrega 2] — preencher após treinamento e avaliação completos.

### Métricas obtidas

| Modelo | AUC-ROC mel | AUC-ROC macro | F1-macro | Acurácia |
|--------|-------------|---------------|----------|----------|
| Majoritário (baseline trivial) | 0,500 | — | — | ~67% |
| ResNet18 (baseline) | [TODO] | [TODO] | [TODO] | [TODO] |
| EfficientNet-B3 | [TODO] | [TODO] | [TODO] | [TODO] |

### Gráficos relevantes

Figuras salvas em `docs/figs/` e `outputs/`:
- `class_distribution.png` — distribuição de classes no dataset
- `metadata_distributions.png` — idade, sexo, localização
- `{model}_confusion_matrix.png` — matriz de confusão
- `{model}_roc_curves.png` — curvas ROC por classe
- `model_comparison.png` — comparação entre modelos
- `outputs/gradcam/` — mapas Grad-CAM++ por classe

### Análise de erros

[TODO Entrega 2]

### Comparação com a hipótese

[TODO Entrega 2]

---

## Conclusão

### Principais achados

[TODO Entrega 2]

### Limitações

- Dataset majoritariamente de pele clara (fototipos I–III); desempenho em fototipos V–VI requer validação.
- Modelo de apoio ao diagnóstico — não substitui avaliação médica qualificada.
- [TODO Entrega 2]

### Trabalhos futuros

[TODO Entrega 2]

---

## Reprodutibilidade

### Requisitos

- Python 3.10+
- CUDA 11.8+ (opcional; CPU funciona)
- Conta Kaggle com acesso ao dataset `kmader/skin-cancer-mnist-ham10000`
- Sistema operacional testado: Ubuntu 22.04
- Semente global: **42**

### Instalação

```bash
git clone <URL-do-repositório>
cd Sistema-de-classificacao-de-lesoes-dermatologicas

python -m venv venv
source venv/bin/activate        # Linux/macOS
# venv\Scripts\activate         # Windows

pip install -r requirements.txt
```

### Obter os dados

```bash
cp .env.example .env
# Edite .env com suas credenciais Kaggle:
# KAGGLE_USERNAME=seu_usuario
# KAGGLE_KEY=sua_chave_api

make download    # baixa o dataset (~3 GB)
make prepare     # organiza em train/val/test por classe
make eda         # gera figuras de EDA em docs/figs/
```

### Executar o pipeline

```bash
make train-baseline     # treina ResNet18 (baseline)
make train-main         # treina EfficientNet-B3 (modelo principal)
make evaluate-baseline  # avalia ResNet18 no conjunto de teste
make evaluate-main      # avalia EfficientNet-B3 no conjunto de teste
make compare            # tabela e gráfico comparativo
make gradcam            # mapas Grad-CAM++ por classe
```

**Pipeline completo de uma vez:**

```bash
make all
```

### Artefatos gerados

| Caminho | Conteúdo | Tamanho estimado |
|---------|----------|-----------------|
| `checkpoints/*.pth` | Pesos do melhor modelo por época | ~50–200 MB por modelo |
| `outputs/evaluation/` | Matriz de confusão + curvas ROC | ~2 MB |
| `outputs/gradcam/` | Imagens com overlay Grad-CAM++ | ~10 MB |
| `outputs/comparison/` | Gráfico comparativo de modelos | ~1 MB |
| `docs/figs/` | Figuras da EDA | ~5 MB |

---

*Projeto desenvolvido para a disciplina Projetos de IA — Módulo 7, PPI/SiDi. Dataset: HAM10000 (CC BY-NC 4.0). Instrutor: Nicksson Freitas.*
