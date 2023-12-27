"""Load html from files, clean up, split, ingest into Weaviate."""
import logging
import os
import re
from parser import langchain_docs_extractor

import weaviate
from bs4 import BeautifulSoup, SoupStrainer
from langchain.document_loaders import RecursiveUrlLoader, SitemapLoader, GitLoader
from langchain.indexes import SQLRecordManager
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.utils.html import (PREFIXES_TO_IGNORE_REGEX,
                                  SUFFIXES_TO_IGNORE_REGEX)
from langchain.vectorstores.weaviate import Weaviate

from _index import index
from chain import get_embeddings_model
from constants import WEAVIATE_DOCS_INDEX_NAME

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

WEAVIATE_URL = os.environ["WEAVIATE_URL"]
WEAVIATE_API_KEY = os.environ["WEAVIATE_API_KEY"]

# os.environ["RECORD_MANAGER_DB_URL"]
RECORD_MANAGER_DB_URL = "sqlite:///mydatabase.db"


def metadata_extractor(meta: dict, soup: BeautifulSoup) -> dict:
    title = soup.find("title")
    description = soup.find("meta", attrs={"name": "description"})
    html = soup.find("html")
    return {
        "source": meta["loc"],
        "title": title.get_text() if title else "",
        "description": description.get("content", "") if description else "",
        "language": html.get("lang", "") if html else "",
        **meta,
    }


def load_tbd_docs():
    return SitemapLoader(
        "https://developer.tbd.website/sitemap.xml",
        filter_urls=[
            r'https://developer\.tbd\.website/(?!blog|api|community).*'],
        # filter_urls=["https://developer.tbd.website/docs/tbdex/message-types"],
        parsing_function=langchain_docs_extractor,
        default_parser="lxml",
        bs_kwargs={
            "parse_only": SoupStrainer(
                name=("article", "title")
            ),
        },
        meta_function=metadata_extractor,
    ).load()


def load_web5_code():
    return GitLoader(
        "web5-js",
        clone_url="https://github.com/TBD54566975/web5-js.git"
    ).load()


def load_tbdex_api_docs():
    return RecursiveUrlLoader(
        url="https://tbd54566975.github.io/tbdex-js/",
        max_depth=8,
        extractor=simple_extractor,
        prevent_outside=True,
        use_async=True,
        timeout=600,
        # Drop trailing / to avoid duplicate pages.
        link_regex=(
            f"href=[\"']{PREFIXES_TO_IGNORE_REGEX}((?:{SUFFIXES_TO_IGNORE_REGEX}.)*?)"
            r"(?:[\#'\"]|\/[\#'\"])"
        ),
        check_response_status=True,
        # exclude_dirs=(
        #     "https://api.python.langchain.com/en/latest/_sources",
        #     "https://api.python.langchain.com/en/latest/_modules",
        # ),
    ).load()


def load_langchain_docs():
    return SitemapLoader(
        "https://python.langchain.com/sitemap.xml",
        filter_urls=["https://python.langchain.com/"],
        parsing_function=langchain_docs_extractor,
        default_parser="lxml",
        bs_kwargs={
            "parse_only": SoupStrainer(
                name=("article", "title", "html", "lang", "content")
            ),
        },
        meta_function=metadata_extractor,
    ).load()


def load_langsmith_docs():
    return RecursiveUrlLoader(
        url="https://docs.smith.langchain.com/",
        max_depth=8,
        extractor=simple_extractor,
        prevent_outside=True,
        use_async=True,
        timeout=600,
        # Drop trailing / to avoid duplicate pages.
        link_regex=(
            f"href=[\"']{PREFIXES_TO_IGNORE_REGEX}((?:{SUFFIXES_TO_IGNORE_REGEX}.)*?)"
            r"(?:[\#'\"]|\/[\#'\"])"
        ),
        check_response_status=True,
    ).load()


def simple_extractor(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    return re.sub(r"\n\n+", "\n\n", soup.text).strip()


def load_api_docs():
    return RecursiveUrlLoader(
        url="https://api.python.langchain.com/en/latest/",
        max_depth=8,
        extractor=simple_extractor,
        prevent_outside=True,
        use_async=True,
        timeout=600,
        # Drop trailing / to avoid duplicate pages.
        link_regex=(
            f"href=[\"']{PREFIXES_TO_IGNORE_REGEX}((?:{SUFFIXES_TO_IGNORE_REGEX}.)*?)"
            r"(?:[\#'\"]|\/[\#'\"])"
        ),
        check_response_status=True,
        exclude_dirs=(
            "https://api.python.langchain.com/en/latest/_sources",
            "https://api.python.langchain.com/en/latest/_modules",
        ),
    ).load()


def ingest_docs():
    # docs_from_documentation_langchain = load_langchain_docs()
    # logger.info(f"Loaded {len(docs_from_documentation_langchain)} docs from documentation")
    # docs_from_langchain_api = load_api_docs()
    # logger.info(f"Loaded {len(docs_from_langchain_api)} docs from API")
    # docs_from_langsmith = load_langsmith_docs()
    # logger.info(f"Loaded {len(docs_from_langsmith)} docs from Langsmith")
    docs_from_documentation = load_tbd_docs()
    logger.info(
        f"Loaded {len(docs_from_documentation)} docs from TBD documentation at developer.tbd.website")
    # docs_from_code = load_web5_code()
    # logger.info(
    #     f"Loaded {len(docs_from_code)} docs from the sourcecode at web5-js repo")
    # docs_from_api = load_tbdex_api_docs()
    # logger.info(f"Loaded {len(docs_from_api)} docs from tbDEX-js API docs at https://tbd54566975.github.io/tbdex-js")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=4000, chunk_overlap=200)

    # for doc in docs_from_code:
    #     doc.metadata["title"] = doc.metadata["source"]
    #     # todo: add github raw file path prefix
    #     # todo: change from main to a specific released tagged branch

    docs_transformed = text_splitter.split_documents(
        docs_from_documentation
        # docs_from_documentation + docs_from_api + docs_from_code
        # docs_from_documentation_langchain
        # docs_from_langchain_api
    )  # + docs_from_code

    # We try to return 'source' and 'title' metadata when querying vector store and
    # Weaviate will error at query time if one of the attributes is missing from a
    # retrieved document.
    with open("docs_transformed.txt", 'a') as file:
        for doc in docs_transformed:
            if "source" not in doc.metadata:
                print("!!! Missing source for doc")
                doc.metadata["source"] = ""
            if "title" not in doc.metadata:
                print("!!! Missing title for doc")
                doc.metadata["title"] = ""
                print(doc.metadata)
            print(doc, file=file)

    client = weaviate.Client(
        url=WEAVIATE_URL,
        # auth_client_secret=weaviate.AuthApiKey(api_key=WEAVIATE_API_KEY),
    )
    embedding = get_embeddings_model()
    vectorstore = Weaviate(
        client=client,
        index_name=WEAVIATE_DOCS_INDEX_NAME,
        text_key="text",
        embedding=embedding,
        by_text=False,
        attributes=["source", "title"],
    )

    record_manager = SQLRecordManager(
        f"weaviate/{WEAVIATE_DOCS_INDEX_NAME}", db_url=RECORD_MANAGER_DB_URL
    )
    record_manager.create_schema()

    indexing_stats = index(
        docs_transformed,
        record_manager,
        vectorstore,
        cleanup="full",
        source_id_key="source",
        force_update=(os.environ.get("FORCE_UPDATE")
                      or "false").lower() == "true",
    )

    logger.info(f"Indexing stats: {indexing_stats}")
    num_vecs = client.query.aggregate(
        WEAVIATE_DOCS_INDEX_NAME).with_meta_count().do()
    logger.info(
        f"Weaviate now has this many vectors: {num_vecs}",
    )


if __name__ == "__main__":
    ingest_docs()
