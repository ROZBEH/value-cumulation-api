import re
import os
import glob
import datetime
from sec_api import QueryApi, RenderApi
from logger import custom_logger
from llama_index import (
    download_loader,
    VectorStoreIndex,
    ServiceContext,
    StorageContext,
    load_index_from_storage,
    ListIndex,
    LLMPredictor,
    load_graph_from_storage,
)
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.agents import initialize_agent
from llama_index.query_engine.transform_query_engine import TransformQueryEngine
from llama_index.indices.query.query_transform.base import DecomposeQueryTransform
from llama_index.langchain_helpers.agents import (
    LlamaToolkit,
    create_llama_chat_agent,
    IndexToolConfig,
)
from langchain import OpenAI
from llama_index.indices.composability import ComposableGraph
from pathlib import Path

from llama_index import (
    download_loader,
    ServiceContext,
    GPTVectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
    LLMPredictor,
)
from langchain.llms.openai import OpenAIChat

from langchain.chat_models import ChatOpenAI

from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
current_dir = os.path.dirname(os.path.abspath(__file__))


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
    llm_predictor = LLMPredictor(
        # llm=ChatOpenAI(model_name="gpt-4", max_tokens=512, temperature=0.1)
        llm=OpenAIChat(model_name="gpt-4", max_tokens=512, temperature=0.1)
    )
    service_context = ServiceContext.from_defaults(
        llm_predictor=llm_predictor, chunk_size_limit=512
    )
    if not Path("./storage").is_dir():
        print("creating index")
        UnstructuredReader = download_loader("UnstructuredReader")
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


def is_ticker_in_file(ticker, filename="indexList.txt"):
    filename = os.path.join(current_dir, filename)
    with open(filename, "r") as file:
        tickers = file.read().splitlines()
    return ticker in tickers


def fetch_sec_report(report_type="10-K", ticker="AAPL", year=None):
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
                "query": f'ticker:{ticker} AND filedAt:{{{year}-01-01 TO {year+1}-04-04}} AND formType:"{report_type}"'
            }
        },
        "from": "0",
        "size": "10",
        "sort": [{"filedAt": {"order": "desc"}}],
    }

    filings = queryApi.get_filings(query)
    sec_url = filings["filings"][0]["linkToFilingDetails"]
    filing = renderApi.get_filing(sec_url)
    html_file = os.path.join(current_dir, "tmp_" + ticker + ".html")
    with open(html_file, "w") as f:
        f.write(filing)
    index_list_file = os.path.join(current_dir, "indexList.txt")
    with open(index_list_file, "a") as f:
        f.write(ticker + "\n")
    return html_file


def index_sec_url(report_type="10-K", ticker="AAPL", year=None) -> GPTVectorStoreIndex:
    """
    Reads the contents of an html file and creates a GPTVectorStoreIndex from it.

    Args:
        report_type (str): Type of report to download
        ticker (str): Ticker of the company to download

    Returns:
        index (GPTVectorStoreIndex): The GPTVectorStoreIndex created from the text file.

    """
    html_file = None
    if not is_ticker_in_file(ticker):
        html_file = fetch_sec_report(report_type, ticker, year)

    llm_predictor = LLMPredictor(
        llm=ChatOpenAI(model_name="gpt-3.5-turbo", max_tokens=512, temperature=0.1)
        # llm=OpenAIChat(model_name="gpt-4", max_tokens=512, temperature=0.1)
    )

    service_context = ServiceContext.from_defaults(
        llm_predictor=llm_predictor, chunk_size_limit=512
    )

    storage_path = os.path.join(current_dir, "storage")
    if html_file:
        if Path(storage_path).is_dir():
            storage_context = StorageContext.from_defaults(persist_dir=storage_path)
            custom_logger.info("Index found. Loading ...")
            custom_logger.info(f"Adding {ticker} to the index")
            UnstructuredReader = download_loader("UnstructuredReader")
            loader = UnstructuredReader()
            document = loader.load_data(file=Path(html_file), split_documents=False)
            # load index
            index = load_index_from_storage(
                storage_context, service_context=service_context
            )

            index.insert(document[0])
            index.storage_context.persist(persist_dir=storage_path)
        else:
            custom_logger.info("No index found. Creating ...")
            # reading the json html file approach
            UnstructuredReader = download_loader("UnstructuredReader")
            loader = UnstructuredReader()
            document = loader.load_data(file=Path(html_file), split_documents=False)
            # read the txt file approach
            index = GPTVectorStoreIndex.from_documents(
                document,
                service_context=service_context,
            )
            index.storage_context.persist(persist_dir=storage_path)
    else:
        storage_context = StorageContext.from_defaults(persist_dir=storage_path)
        custom_logger.info(f"{ticker} is already in the index. Loading ...")
        UnstructuredReader = download_loader("UnstructuredReader")
        index = load_index_from_storage(
            storage_context, service_context=service_context
        )

    return index


def chat_bot_agent(load_index=True):
    storage_path = os.path.join(current_dir, "storage")
    service_context = ServiceContext.from_defaults(chunk_size=512)
    # find all the files inside data/sec10K folder
    html_files = glob.glob(os.path.join(current_dir, "data/sec10K/*.html"))

    UnstructuredReader = download_loader("UnstructuredReader", refresh_cache=True)

    loader = UnstructuredReader()
    doc_set = {}
    all_docs = []
    companies = []
    for html_file in html_files:
        pattern = r"tmp_(.+)\.html"
        company_name = re.search(pattern, html_file).group(1)
        companies.append(company_name)
        if not load_index:
            document = loader.load_data(file=html_file, split_documents=False)
            # insert company metadata into each company
            for d in document:
                d.extra_info = {"company": company_name}
            doc_set[company_name] = document
            all_docs.extend(document)

    # set up vector indices for each company
    service_context = ServiceContext.from_defaults(chunk_size=512)
    # index_set = {}
    # for company in companies:
    #     storage_context = StorageContext.from_defaults()
    #     cur_index = VectorStoreIndex.from_documents(
    #         doc_set[company],
    #         service_context=service_context,
    #         storage_context=storage_context,
    #     )
    #     index_set[company] = cur_index
    #     storage_context.persist(persist_dir=f"{storage_path}/{company}")

    # Load indices from disk
    index_set = {}
    for company in companies:
        storage_context = StorageContext.from_defaults(
            persist_dir=f"{storage_path}/{company}"
        )
        cur_index = load_index_from_storage(storage_context=storage_context)
        index_set[company] = cur_index

    # Composing a Graph to Synthesize Answers Across 10-K Filings
    # describe each index to help traversal of composed graph
    index_summaries = [f"10-k Filing for {company}" for company in companies]

    # define an LLMPredictor set number of output tokens
    llm_predictor = LLMPredictor(
        llm=OpenAI(temperature=0, max_tokens=512, model_name="gpt-3.5-turbo")
    )
    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)
    storage_context = StorageContext.from_defaults()

    # define a list index over the vector indices
    # allows us to synthesize information across each index
    graph = ComposableGraph.from_indices(
        ListIndex,
        [index_set[y] for y in companies],
        index_summaries=index_summaries,
        service_context=service_context,
        storage_context=storage_context,
    )
    root_id = graph.root_id

    # [optional] save to disk
    storage_context.persist(persist_dir=f"{storage_path}/root")

    # [optional] load from disk, so you don't need to build graph from scratch
    graph = load_graph_from_storage(
        root_id=root_id,
        service_context=service_context,
        storage_context=storage_context,
    )

    ### Setting up the Tools + Langchain Chatbot Agent
    decompose_transform = DecomposeQueryTransform(llm_predictor, verbose=True)
    # define custom retrievers
    custom_query_engines = {}
    for index in index_set.values():
        query_engine = index.as_query_engine()
        query_engine = TransformQueryEngine(
            query_engine,
            query_transform=decompose_transform,
            transform_extra_info={"index_summary": index.index_struct.summary},
        )
        custom_query_engines[index.index_id] = query_engine
    custom_query_engines[graph.root_id] = graph.root_index.as_query_engine(
        response_mode="tree_summarize",
        verbose=True,
    )
    # construct query engine
    graph_query_engine = graph.as_query_engine(
        custom_query_engines=custom_query_engines
    )

    # tool config
    graph_config = IndexToolConfig(
        query_engine=graph_query_engine,
        name=f"Graph Index",
        description="useful for when you want to answer queries that require analyzing multiple SEC 10-K documents for Companies.",
        tool_kwargs={"return_direct": True},
    )

    # define toolkit
    index_configs = []
    for company in companies:
        query_engine = index_set[company].as_query_engine(
            similarity_top_k=3,
        )
        tool_config = IndexToolConfig(
            query_engine=query_engine,
            name=f"Vector Index {company}",
            description=f"useful for when you want to answer queries about the {company} SEC 10-K",
            tool_kwargs={"return_direct": True},
        )
        index_configs.append(tool_config)

    toolkit = LlamaToolkit(
        index_configs=index_configs + [graph_config],
    )

    memory = ConversationBufferMemory(memory_key="chat_history")
    llm = OpenAI(temperature=0)
    agent_chain = create_llama_chat_agent(toolkit, llm, memory=memory, verbose=True)
    return agent_chain
