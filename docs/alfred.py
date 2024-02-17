from __future__ import annotations

import os
from collections import namedtuple
import asyncio
from tqdm.asyncio import tqdm
import typing as t
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.language_models.chat_models import BaseChatModel
from langchain.prompts import ChatPromptTemplate

File = namedtuple("File", "name content")


def get_files(path: str, ext: str) -> list:
    return [os.path.join(path, f) for f in os.listdir(path) if f.endswith(ext)]


def load_docs(path: str) -> t.List[File]:
    files = [*get_files(path, ".md")]
    docs = []
    for file in files:
        with open(file, "r") as f:
            docs.append(File(file, f.read()))
    return docs


async def fix_doc_with_llm(doc: File, llm: BaseChatModel) -> File:
    prompt = """\
fix the following grammar and spelling mistakes in the following text. 
Please keep the markdown format intact when reformating it. 
Do not make any change to the parts of text that are for formating or additional metadata for the core text in markdown.
The target audience for this is developers so keep the tone serious and to the point without any marketing terms. 
The output text should me in .md format. 

text: {text}
"""
    fix_docs_prompt = ChatPromptTemplate.from_messages(
        [
            (prompt),
        ]
    )
    # get output
    fixed_doc = await llm.ainvoke(fix_docs_prompt.format_messages(text=doc.content))
    return File(doc.name, fixed_doc.content)


async def main(docs: t.List[File], llm: BaseChatModel):
    fix_doc_routines = [fix_doc_with_llm(doc, llm) for doc in docs]
    return await tqdm.gather(*fix_doc_routines)


if __name__ == "__main__":
    """
    Helpful assistant for documentation review and more (hopefully in the future).
    """
    gpt4 = ChatOpenAI(model="gpt-4")
    docs = load_docs("./getstarted/")
    fix_docs = asyncio.run(main(docs, gpt4))
    for doc in fix_docs:
        with open(doc.name, "w") as f:
            f.write(doc.content)
