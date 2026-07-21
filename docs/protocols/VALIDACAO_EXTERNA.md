# Protocolo de validação externa

## Estado

Protocolo preparado, mas não executado. O projeto não recebeu uma segunda base
independente e, portanto, não apresenta validação externa como resultado.

## Base candidata

Priorizar uma coleção pública ISIC com licença compatível com uso acadêmico,
metadados de proveniência e diagnósticos confirmados. Antes do download:

1. verificar licença e termos de redistribuição;
2. confirmar ausência de imagens também presentes no HAM10000;
3. documentar população, centros de origem e métodos diagnósticos;
4. mapear rótulos externos para as sete classes do projeto;
5. excluir classes sem correspondência semântica inequívoca.

## Protocolo congelado

- Usar o checkpoint final sem novo treinamento na base externa.
- Aplicar a mesma normalização e resolução.
- Manter o temperature scaling e o limiar definidos apenas no HAM10000.
- Reportar AUC-ROC de melanoma, sensibilidade, especificidade, AUC macro, F1
  macro, acurácia balanceada, ECE e Brier.
- Calcular intervalos de confiança de 95% por bootstrap no nível do paciente.
- Estratificar por centro, sexo, idade e fototipo quando disponíveis.
- Registrar imagens inválidas e rótulos não mapeáveis antes da avaliação.

## Critérios de aceitação

- AUC-ROC de melanoma com limite inferior do IC 95% acima de 0,80.
- Sensibilidade de melanoma igual ou superior a 0,85 no limiar congelado.
- Queda de AUC de melanoma inferior a 0,08 em relação ao teste interno.
- ECE igual ou inferior a 0,10.

Qualquer adaptação do modelo ou do limiar transforma a análise em adaptação de
domínio e exige uma nova partição externa mantida intocada.
