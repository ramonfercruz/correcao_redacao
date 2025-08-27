from __future__ import annotations

import sys
from pathlib import Path

from langchain.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain.schema import OutputParserException
from langchain_openai import ChatOpenAI

BASE_DIR = Path(__file__).resolve().parent
PROMPT_DIR = BASE_DIR / "prompts" / "system"
PROMPT_FILES = [
    "c1_escrita_formal.txt",
    "c2_tema.txt",
    "c3_argumentacao.txt",
    "c4_coesao.txt",
    "c5_intervencao.txt",
]


def read_file(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def build_system_prompt() -> str:
    prompts = [read_file(PROMPT_DIR / name) for name in PROMPT_FILES]
    return "\n\n".join(prompts)


class NumberOutputParser(StrOutputParser):
    """Parse the LLM output ensuring it is a numeric value."""

    def parse(self, text: str) -> float:  # type: ignore[override]
        try:
            return float(text.strip())
        except ValueError as err:  # pragma: no cover - simple validation
            raise OutputParserException(
                f"Expected numeric output, got: {text}"
            ) from err


def evaluate_essay(essay_path: str) -> None:
    essay = read_file(Path(essay_path))
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", build_system_prompt()),
            ("user", "{essay}"),
        ]
    )
    chain = prompt | ChatOpenAI(model="gpt-4o-mini") | NumberOutputParser()
    score = chain.invoke({"essay": essay})
    print(score)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Uso: python main_resumo.py caminho_para_redacao.txt")
        raise SystemExit(1)
    evaluate_essay(sys.argv[1])