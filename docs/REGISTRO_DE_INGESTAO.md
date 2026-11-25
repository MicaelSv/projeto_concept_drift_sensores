# Registro da ingestão dos lotes

Os dados foram fornecidos em dez arquivos denominados `batch1(1).dat` até `batch10(1).dat`.
Para operação do projeto, foram feitas **cópias binariamente idênticas** com nomes `batch1.dat` até `batch10.dat`.
A equivalência precisa ser conferida pelos hashes em `checksums/originais.sha256` e pelo arquivo original arquivado pelo pesquisador.

O projeto utiliza a sequência de 10 lotes temporais do artigo de Vergara et al. (2012), com 13.910 observações no total.
**Não há timestamps individuais por linha no formato LIBSVM desses arquivos**; `lote` e `linha_origem_0` são identificadores de proveniência, não datas ou horas.

Este registro não substitui a obtenção controlada e preservação da origem no repositório do usuário.
