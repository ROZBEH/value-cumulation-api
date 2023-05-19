from llama_index import (
    GPTVectorStoreIndex,
    SimpleDirectoryReader,
    PromptHelper,
    ServiceContext,
    LLMPredictor,
)
from llama_index.node_parser import SimpleNodeParser

from dotenv import load_dotenv
from llama_index.readers import BeautifulSoupWebReader


from langchain import OpenAI

load_dotenv()
# create a BeautifulSoupWebReader object
web_reader = BeautifulSoupWebReader()
parser = SimpleNodeParser()

# define a list of URLs to scrape
urls = [
    "https://financialmodelingprep.com/api/v4/financial-reports-json?symbol=AAPL&year=2020&period=FY&apikey=30d0838215af7a980b24b41cab12410f"
]

# load data from the URLs
documents = web_reader.load_data(urls)

# print the text of the first document
# document = documents[0].text
llm_predictor = LLMPredictor(llm=OpenAI(temperature=0, model_name="text-davinci-003"))


# define prompt helper
# set maximum input size
max_input_size = 4096
# set number of output tokens
num_output = 256
# set maximum chunk overlap
max_chunk_overlap = 20
prompt_helper = PromptHelper(max_input_size, num_output, max_chunk_overlap)

service_context = ServiceContext.from_defaults(
    llm_predictor=llm_predictor, prompt_helper=prompt_helper
)


nodes = parser.get_nodes_from_documents(documents)

# index = GPTVectorStoreIndex.from_documents(documents, service_context=service_context)
# index = GPTVectorStoreIndex(nodes)

index = GPTVectorStoreIndex.build_index_from_nodes(
    nodes, service_context=service_context
)

# index = GPTVectorStoreIndex.from_documents(documents)
index.storage_context.persist()

query_engine = index.as_query_engine()

query_engine = index.as_query_engine()
response = query_engine.query("What is the document type?")
