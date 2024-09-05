# Depth Metrics

## Comportamento esperado

This repository/notebook aims to implement the affine invariant protocol for relative monocular depth estimation models

- Step 1: Load the data.

- Step 2: Pre-process: Transform into disparity, apply masks, deal with zero division.

- Step 3: Align the prediction with groundtruth with least squares algorithm

- Step 4: Calculate the Metrics (AbsRel, delta, RMSE)

- Step 5: Output the results in pandas dataframe and figures.  

##  Coisas pra fazer:

1) **Zero Division**: Modificar cálculos para evitar divisão por zero  :white_check_mark:

2) **Métricas**: implementar o código para outras métricas (delta123, RMSE) ![](https://geps.dev/progress/80)
  
3) **Loop de teste**: fazer rodar para todas as imagens do dataset   

4) **Saída**: salvar em dataframe  
   1) Dataframe com delta, absrel e...

5) **Realizar testes**: Teste no código de teste
   1) Imagens sintéticas com erros conhecidos para testar
   
6) Rodar nos outros datasets
   1) Codar individualmente o teste para cada um
   2) pensar numa maneira de isolar o carregamento dos dados
   
7) Exibir resultados de outras formas
   1) Codificar teste de erro por distância
   2) gerar imagens de erro
   3) pesquisar outras formas...