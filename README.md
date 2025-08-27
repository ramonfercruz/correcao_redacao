# Correção de Redações do ENEM com IA

Ferramenta simples que utiliza modelos da OpenAI por meio do LangChain
para avaliar textos conforme os cinco critérios da redação do ENEM.
O projeto lê redações de um diretório, aplica um *prompt* específico
para cada critério e gera um arquivo JSON com as notas e descrições
correspondentes.

## Requisitos

- Python 3.11+
- Chave de API da OpenAI definida na variável de ambiente `OPENAI_API_KEY`
- Bibliotecas listadas em `requirements.txt`

## Instalação

```bash
pip install -r requirements.txt
```

Crie um arquivo `.env` na raiz do projeto com sua chave da OpenAI:

```env
OPENAI_API_KEY=coloque_sua_chave_aqui
```

## Estrutura do Projeto

```
correcao_redacao/
├── main.py               # Script principal de avaliação
├── prompts/system/       # Prompts para cada competência do ENEM
├── redacao/              # Textos a serem avaliados (.txt)
├── descricao_score.json  # Descrição das notas por critério
└── resultado_nota.json   # Resultado gerado após a execução
```

## Uso

Após adicionar os textos no diretório `redacao/`, execute:

```bash
python main.py
```

O arquivo `resultado_nota.json` será criado com as avaliações
por critério para cada redação.

## Licença

Este projeto é distribuído sob a licença MIT. Veja o arquivo `LICENSE`
se houver.