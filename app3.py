from haystack.document_stores import InMemoryDocumentStore
from haystack.utils.preprocessing import convert_files_to_docs

document_store = InMemoryDocumentStore(use_bm25=True)

doc_dir="story"
dicts = convert_files_to_docs(doc_dir)
document_store.write_documents(dicts)

from haystack.nodes import BM25Retriever
retriever = BM25Retriever(document_store=document_store)

from haystack.nodes import FARMReader
reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2", use_gpu=True)

from haystack.pipelines import ExtractiveQAPipeline
pipe = ExtractiveQAPipeline(reader, retriever)
prediction = pipe.run(
    query="Who led the first expidition?"
)

from pprint import pprint
pprint(prediction)

from haystack.utils import print_answers
print_answers(prediction, details="medium")  ## Choose from `minimum`, `medium`, and `all`


