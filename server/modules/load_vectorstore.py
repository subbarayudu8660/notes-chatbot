import os
import time
from pathlib import Path
from dotenv import load_dotenv
from tqdm.auto import tqdm
from pinecone import Pinecone,ServerlessSpec
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_experimental.text_splitter import SemanticChunker

load_dotenv()

PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_ENV = "us-east-1"
PINECONE_INDEX_NAME = 'Notes_index'

UPLOAD_DIR = "./uploaded_docs"
os.makedirs(UPLOAD_DIR,exist_ok=True)

pc = Pinecone(api_key=PINECONE_API_KEY)
spec= ServerlessSpec(cloud = "aws",region = PINECONE_ENV)
existing_indexes=[i["name"] for i in pc.list_indexes()]

if PINECONE_INDEX_NAME not in existing_indexes:
    pc.create_index(
        name = PINECONE_INDEX_NAME,
        dimension = 384,
        metric = "cosine",
        spec=spec

    )
    while not pc.describe_index(PINECONE_INDEX_NAME).status["ready"]:
        time.sleep(1)

index = pc.Index(PINECONE_INDEX_NAME)

def load_vectorstore(uploaded_file):
    embeddings = HuggingFaceEmbeddings(model_name = "all-MiniLM-L6-v2")
    file_path = []

    for file in uploaded_file:
        save_path = Path(UPLOAD_DIR)/file.filename
        with open(save_path,"wb") as f:
            f.write(file.file.read())
        file_path.append(str(save_path))

    all_semantic_chunks = []
    
    for path in file_path:
        loader = PyPDFLoader(path)
        documents = loader.load()


        semantic_splitter = SemanticChunker(
            embeddings,
            breakpoint_threshold_type = "percentile",
            breakpoint_threshold_amount=95
        )
        chunks = semantic_splitter.split_documents(documents)
        all_semantic_chunks.extend(chunks)

        texts = [chunk.page_content for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]
        ids = [f"{Path(path).stem}_{i}" for i in range(len(chunks))]

        print(f"Adding {len(chunks)} chunks from {Path(path).stem} to Pinecone index...")
        vectors = embeddings.embed_documents(texts)


        with tqdm(total = len(vectors),desc = "Upserting to Pinecone") as progress:
            index.upsert(vectors=zip(ids,vectors,metadatas))
            progress.update(len(vectors))

            print(f"Upload complete for {path}")

