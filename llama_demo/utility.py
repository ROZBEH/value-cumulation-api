import os
import datetime
from sec_api import QueryApi, RenderApi
from llama_index import (
    download_loader,
    ServiceContext,
    GPTVectorStoreIndex,
    SimpleDirectoryReader,
    StringIterableReader,
    StorageContext,
    load_index_from_storage,
)
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


def index_txt_doc(path) -> GPTVectorStoreIndex:
    """
    Reads the contents of a text file and creates a GPTVectorStoreIndex from it.

    Args:
        file_path (str): The path to the text file to read.

    Returns:
        index (GPTVectorStoreIndex): The GPTVectorStoreIndex created from the text file.

    """
    documents = SimpleDirectoryReader(path).load_data()
    index = GPTVectorStoreIndex.from_documents(documents)

    return index


def index_html_doc(path) -> GPTVectorStoreIndex:
    """
    Reads the contents of an html file and creates a GPTVectorStoreIndex from it.

    Args:
        file_path (str): The path to the html file to read.

    Returns:
        index (GPTVectorStoreIndex): The GPTVectorStoreIndex created from the text file.

    """
    if not path.endswith(".html") and not path.endswith(".htm"):
        raise ValueError("The file must end in .html or .htm")
    service_context = ServiceContext.from_defaults(chunk_size_limit=512)
    if not Path("./storage").is_dir():
        print("creating index")
        UnstructuredReader = download_loader("UnstructuredReader", refresh_cache=True)
        loader = UnstructuredReader()
        document = loader.load_data(file=Path(path), split_documents=False)
        index = GPTVectorStoreIndex.from_documents(
            document, service_context=service_context
        )
        index.storage_context.persist()
    else:
        print("loading index")
        storage_context = StorageContext.from_defaults(persist_dir="./storage")
        # load index
        index = load_index_from_storage(storage_context)

    return index


# for the year argument, I would like it to be the year previous to the current by default
# write the code and have the default value be the year previous to the current year


def index_sec_url(report_type="10-K", ticker="AAPL", year=None) -> GPTVectorStoreIndex:
    """
    Reads the contents of an html file and creates a GPTVectorStoreIndex from it.

    Args:
        report_type (str): Type of report to download
        ticker (str): Ticker of the company to download

    Returns:
        index (GPTVectorStoreIndex): The GPTVectorStoreIndex created from the text file.

    """
    if year is None:
        year = (
            datetime.datetime.now().year - 2
            if datetime.datetime.now().month == 1
            else datetime.datetime.now().year - 1
        )
    SEC_API_KEY = os.getenv("SEC_API_KEY")
    queryApi = QueryApi(api_key=SEC_API_KEY)
    renderApi = RenderApi(api_key=SEC_API_KEY)

    query = {
        "query": {
            "query_string": {
                "query": f'ticker:{ticker} AND filedAt:{{{year}-01-01 TO {year}-12-31}} AND formType:"{report_type}"'
            }
        },
        "from": "0",
        "size": "10",
        "sort": [{"filedAt": {"order": "desc"}}],
    }

    filings = queryApi.get_filings(query)
    sec_url = filings["filings"][0]["linkToFilingDetails"]
    filing = renderApi.get_filing(sec_url)
    with open("tmp.html", "w") as f:
        f.write(filing)

    service_context = ServiceContext.from_defaults(chunk_size_limit=512)
    if not Path("./storage").is_dir():
        print("creating index")
        # reading the json html file approach
        UnstructuredReader = download_loader("UnstructuredReader", refresh_cache=True)
        loader = UnstructuredReader()
        document = loader.load_data(file=Path("tmp.html"), split_documents=False)
        # read the txt file approach
        # document = StringIterableReader().load_data(texts=[filing])
        index = GPTVectorStoreIndex.from_documents(
            document, service_context=service_context
        )
        index.storage_context.persist()
    else:
        print("loading index")
        storage_context = StorageContext.from_defaults(persist_dir="./storage")
        # load index
        index = load_index_from_storage(storage_context)

    return index
