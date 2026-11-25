# Avaliação dos resultados para o artigo

## Veredito

Os resultados originais são adequados como piloto, mas insuficientes como experimento final: comparavam somente o modelo estático com o treinamento no lote imediatamente anterior, não implementavam a alternativa cumulativa ou o ensemble e estavam marcados como exploratórios.

O experimento final corrige essa lacuna com cinco políticas temporais avaliadas em B2–B10, sempre sem usar dados futuros. O resultado permite responder à pergunta de pesquisa de forma objetiva e cabe em um artigo de oito páginas.

## Resultado central

| Estratégia | Macro-F1 média | Desvio | Pior lote | Vitórias/empates |
|---|---:|---:|---:|---:|
| Cumulativo | 0,724 | 0,079 | 0,606 | 7 |
| Janela de 3 lotes | 0,705 | 0,072 | 0,599 | 2 |
| Ensemble por recência | 0,616 | 0,159 | 0,341 | 2 |
| Último lote | 0,612 | 0,208 | 0,250 | 3 |
| Estático B1 | 0,353 | 0,100 | 0,287 | 1 |

O cumulativo melhorou a macro-F1 média em 0,371 sobre o estático e em 0,111 sobre o último lote. Também apresentou o melhor pior caso e menor variação que as estratégias de curto prazo. As comparações pareadas exploratórias permaneceram abaixo de 0,05 após correção de Holm, mas devem ser interpretadas com cautela porque existem somente nove lotes de teste.

## Interpretação recomendada

- O modelo estático degrada depois de B2, fornecendo evidência preditiva compatível com mudança temporal.
- Usar apenas o lote mais recente é instável, sobretudo quando o lote de treino é pequeno ou omite classes.
- A janela de três lotes já recupera grande parte do desempenho.
- O cumulativo oferece o melhor compromisso entre aquisição e retenção neste conjunto de dados.
- O ensemble simples por votação e recência melhora muito o estático, mas não supera a integração cumulativa dos dados.
- Classes 4 e 6 continuam mais difíceis; isso deve aparecer na discussão, e não ser ocultado pela acurácia global.

## Material a colocar nas 8 páginas

Use a figura `experimento_final_macro_f1.png` e uma versão compacta da tabela de resumo. A tabela completa por lote e o F1 por classe podem ser mencionados no texto ou disponibilizados como material suplementar.

Distribuição sugerida: 0,6 página para título/resumos; 0,8 para introdução; 0,8 para fundamentação; 1,2 para metodologia; 0,8 para protocolo; 1,4 para resultados; 0,8 para discussão; 0,6 para limitações/conclusão; 1,0 para referências.

## Limitações que precisam ser declaradas

- Apenas dez lotes agregados e nove momentos de teste.
- Ausência de timestamps individuais dentro dos lotes.
- Hiperparâmetros fixos, sem otimização temporal aninhada.
- Um único conjunto de dados e um único tipo de classificador.
- As quedas de desempenho não constituem, isoladamente, um detector estatístico formal de concept drift.
- SVC e o protocolo são determinísticos; por isso repetições com seeds não se aplicam neste desenho.
