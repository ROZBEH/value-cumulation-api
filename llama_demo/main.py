from llama_index import GPTVectorStoreIndex, SimpleDirectoryReader
from dotenv import load_dotenv

load_dotenv()

documents = SimpleDirectoryReader(
    "/Users/rouzbeh/value-cumulation/value-cumulation-api/llama_index/examples/paul_graham_essay/data"
).load_data()
index = GPTVectorStoreIndex.from_documents(documents)

query_engine = index.as_query_engine()

query_engine = index.as_query_engine()
response = query_engine.query("What did the author do growing up?")
print(response)
