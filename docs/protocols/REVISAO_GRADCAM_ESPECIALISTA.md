# Protocolo de revisão dos mapas Grad-CAM

## Estado

Instrumentos e casos preparados; revisão clínica ainda não executada. Somente
um dermatologista qualificado pode preencher e assinar esta avaliação.

## Material

- `specialist_review_blinded.csv`: formulário sem diagnóstico real ou tipo de
  erro.
- `specialist_review_answer_key.csv`: gabarito restrito à equipe de avaliação.
- `reports/final/error_gradcam/`: mapas Grad-CAM++ dos casos selecionados.

Foram selecionados 15 falsos negativos mais confiantes e 15 falsos positivos
mais confiantes para melanoma. O especialista não deve acessar o gabarito antes
de concluir a revisão.

## Instruções

Para cada caso, avaliar:

1. plausibilidade anatômica do mapa, de 1 a 5;
2. se a ativação se concentra na lesão e não em pelos, réguas, bordas ou
   artefatos;
3. qualidade da imagem, de 1 a 5;
4. comentário livre sobre padrão dermatoscópico e possíveis fontes de erro.

## Síntese planejada

- Mediana e intervalo interquartil da plausibilidade.
- Percentual de mapas focados na lesão.
- Comparação descritiva entre falsos positivos e falsos negativos.
- Lista dos artefatos visuais mais frequentes.
- Concordância entre dois especialistas, caso haja dois revisores.

Grad-CAM é uma explicação pós-hoc. Mesmo uma região visualmente plausível não
prova que o modelo aprendeu um mecanismo causal ou clinicamente válido.
