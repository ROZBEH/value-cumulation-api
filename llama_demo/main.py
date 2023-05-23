import os
from utility import index_txt_doc, index_html_doc, index_sec_url

index = index_sec_url()
query_engine = index.as_query_engine()
# query = "Which company this report is about?"
# results = query_engine.query(query)
# print(results)
# query = "What is the document type?"
# results = query_engine.query(query)
# print(results)
query = "What is the product that Apple makes the most money out of?"
results = query_engine.query(query)
print(results)
query = "How about the least?"
results = query_engine.query(query)
print(results)
query = "What is the product that Apple makes the least money out of?"
results = query_engine.query(query)
print(results)
# query = "How much apple executives make?"
# results = query_engine.query(query)
# print(results)

query = "Who is on the BOD?"
results = query_engine.query(query)
print(results)
