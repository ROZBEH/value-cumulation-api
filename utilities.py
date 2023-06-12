import os
import pinecone
from dotenv import load_dotenv
from llama_index.vector_stores import PineconeVectorStore

load_dotenv()


def create_pinecone_index(
    index_name: str = "test-index",
    dimension: int = 768,
    metric: str = "euclidean",
    pod_type: str = "p1",
    environment: str = "us-west1-gcp",
):
    """
    Create a Pinecone index with the given name, dimension, metric, and pod_type.
    Args:
        index_name(str): The name of the Pinecone index to create.
        dimension(int): The dimension of the vectors to be indexed.
        metric(str): The distance metric to use for the index.
        pod_type(str): The type of Pinecone pod to use for the index.
        environment(str): The environment to create the index in.
    Returns:
        None
    """
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

    # Initialize Pinecone index

    pinecone.init(api_key=PINECONE_API_KEY, environment=environment)

    # create Pinecone index
    pinecone.create_index(
        index_name, dimension=dimension, metric=metric, pod_type=pod_type
    )
