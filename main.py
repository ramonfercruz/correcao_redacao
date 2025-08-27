import sys
from pathlib import Path
import json
import os

from langchain_core.output_parsers.string import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain.schema import OutputParserException
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

# --- Constantes e caminhos base ---
BASE_DIR = Path(__file__).resolve().parent
DIRETORIO_PROMPTS = BASE_DIR / "prompts" / "system"
DIRETORIO_REDACAO = BASE_DIR / "redacao"
ARQUIVO_SCORE = BASE_DIR / 'descricao_score.json'
ARQIVOS_PROMPTS = [  # mantido o nome original da variável
    "c1_escrita_formal.txt",
    "c2_tema.txt",
    "c3_argumentacao.txt",
    "c4_coesao.txt",
    "c5_intervencao.txt",
]


# -----------------------------
# Utilitários de leitura/escrita
# -----------------------------
def ler_scores() -> dict:
    """
    Lê o arquivo JSON com as descrições de score (critérios x notas).
    Retorna um dicionário {criterio: {nota_str: descricao}}.
    """
    if not ARQUIVO_SCORE.exists():
        raise FileNotFoundError(f"Arquivo de scores não encontrado: {ARQUIVO_SCORE}")
    with open(ARQUIVO_SCORE, "r", encoding="utf-8") as f:
        return json.load(f)


def ler_arquivo(path: os.PathLike | str) -> str:
    """
    Lê um arquivo de texto (UTF-8) e retorna seu conteúdo como string.
    Aceita Path ou str.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def escreve_json_score(data: dict | list, path: Path) -> None:
    """
    Escreve um dicionário/lista em JSON (UTF-8) com indentação.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def ler_system_prompt(path: str) -> str:
    """
    Carrega um system prompt a partir do nome do arquivo contido em DIRETORIO_PROMPTS.
    """
    full_path = DIRETORIO_PROMPTS / path
    prompts = ler_arquivo(full_path)
    return prompts


def ler_redacao() -> dict:
    """
    Lê todos os arquivos .txt do diretório de redações e
    retorna um dict {nome_arquivo: conteúdo}.
    """
    if not DIRETORIO_REDACAO.exists():
        raise FileNotFoundError(f"Diretório de redações não encontrado: {DIRETORIO_REDACAO}")
    redacoes: dict[str, str] = {}
    for arquivo in os.listdir(DIRETORIO_REDACAO):
        if arquivo.endswith(".txt"):
            caminho_completo = DIRETORIO_REDACAO / arquivo
            conteudo = ler_arquivo(caminho_completo)
            redacoes[arquivo] = conteudo
    if not redacoes:
        print(f"Aviso: nenhum .txt encontrado em {DIRETORIO_REDACAO}")
    return redacoes


# -----------------------------
# Parser para saída numérica
# -----------------------------
class NumberOutputParser(StrOutputParser):
    """Parseia a saída do LLM garantindo valor numérico (float)."""

    def parse(self, text: str) -> float:  # type: ignore[override]
        try:
            return float(text.strip())
        except ValueError as err:  # validação simples
            raise OutputParserException(
                f"Expected numeric output, got: {text}"
            ) from err


# -----------------------------
# Carregamento de prompts
# -----------------------------
def carrega_prompt() -> list[dict]:
    """
    Percorre a lista ARQIVOS_PROMPTS e monta uma lista de dicts:
    [{'prefixo': 'c1', 'prompt': '<conteúdo>'}, ...]
    O prefixo é inferido pelo trecho antes do primeiro '_'.
    """
    if not DIRETORIO_PROMPTS.exists():
        raise FileNotFoundError(f"Diretório de prompts não encontrado: {DIRETORIO_PROMPTS}")

    prompts: list[dict] = []
    for prompt_path in ARQIVOS_PROMPTS:
        _prompt: dict[str, str] = {}
        prefixo_prompt = prompt_path.split('_')[0]
        prompt = ler_system_prompt(prompt_path)
        _prompt['prefixo'] = prefixo_prompt
        _prompt['prompt'] = prompt
        prompts.append(_prompt)
    return prompts


# -----------------------------
# Avaliação via LLM
# -----------------------------
def avaliar_redacao(prompt: str, redacao: str) -> float:
    """
    Dado um system prompt e o texto da redação, invoca a cadeia LLM -> parser numérico.
    Retorna a nota (float). Em caso de falha de parsing, retorna 0.0.
    """
    prompt_tpl = ChatPromptTemplate.from_messages(
        [
            ("system", prompt),
            ("user", "{redacao}"),
        ]
    )

    # temperature=0 para respostas mais determinísticas;
    # parser numérico reforça que a saída seja somente número.
    chain = prompt_tpl | ChatOpenAI(model="gpt-5", temperature=0) | NumberOutputParser()

    try:
        score: float = chain.invoke({"redacao": redacao})
    except OutputParserException as e:
        # Em caso de saída não-numérica, faz fallback para 0.0 e loga o problema.
        print(f"Aviso: saída não-numérica do LLM. Erro: {e}")
        score = 0.0
    return score


# -----------------------------
# Execução principal
# -----------------------------
if __name__ == "__main__":
    # Checagens rápidas de pré-condições (diretórios/arquivos).
    if not DIRETORIO_PROMPTS.exists():
        print(f"Erro: diretório de prompts inexistente: {DIRETORIO_PROMPTS}")
        sys.exit(1)
    if not DIRETORIO_REDACAO.exists():
        print(f"Erro: diretório de redações inexistente: {DIRETORIO_REDACAO}")
        sys.exit(1)
    if not ARQUIVO_SCORE.exists():
        print(f"Erro: arquivo de descrição de score inexistente: {ARQUIVO_SCORE}")
        sys.exit(1)

    print("Iniciando avaliação de redações...")
    prompts = carrega_prompt()
    redacoes = ler_redacao()
    score_descricao = ler_scores()

    avaliacao_geral: list[dict] = []

    for redacao_nome, redacao in redacoes.items():
        # Monta o registro de avaliação para a redação atual
        avaliacao_redacao: dict = {}
        redacao_nome = redacao_nome.replace('.txt', '')
        avaliacao_redacao['redacao_nome'] = redacao_nome
        avaliacao_redacao['nota_criterio'] = 0.0  # soma das notas por critério
        avaliacao_redacao['avaliacoes'] = []

        # Avalia em cada critério/prompt
        for prompt in prompts:
            score = avaliar_redacao(prompt['prompt'], redacao)

            # Para mapear a descrição, a chave no JSON é string de inteiro (ex.: "0", "40", "80"...)
            _s_score = str(int(score))

            _avaliacao: dict = {}
            _avaliacao['criterio'] = prompt['prefixo']
            _avaliacao['nota'] = score
            avaliacao_redacao['nota_criterio'] += score

            # Busca a descrição correspondente; se não existir, coloca mensagem padrão.
            _descricao = score_descricao.get(prompt['prefixo'], {}).get(
                _s_score, f"Descrição não encontrada para {_s_score} em {prompt['prefixo']}"
            )
            _avaliacao['descricao'] = _descricao

            avaliacao_redacao['avaliacoes'].append(_avaliacao)

        # Adiciona o resultado desta redação ao agregado
        avaliacao_geral.append(avaliacao_redacao)

        # Log simples por redação (opcional)
        print(f"Avaliada: {avaliacao_redacao['redacao_nome']} | Soma dos critérios: {avaliacao_redacao['nota_criterio']}")

    # Exibe a última avaliação processada (mantido do código original)
    print(avaliacao_redacao)

    # Persiste o resultado completo em JSON
    escreve_json_score(data=avaliacao_geral, path=BASE_DIR / "resultado_nota.json")
    print(f"Arquivo salvo em: {BASE_DIR / 'resultado_nota.json'}")
