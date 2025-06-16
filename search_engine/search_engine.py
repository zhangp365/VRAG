import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, List, Mapping, Any, Dict
import json
from tqdm import tqdm
import torch
import numpy as np
import time
import logging
from pathlib import Path
import io
import pickle
from llama_index.core.schema import TextNode, NodeRelationship, RelatedNodeInfo, ImageNode
from vl_embedding import VL_Embedding

logger = logging.getLogger(__name__)

def nodefile2node(input_file):
    nodes = []
    if input_file.endswith(".pkl"):
        nodes = pickle.load(open(input_file, 'rb'))
        return nodes

    if input_file.endswith('.npz'):
        start_time = time.time()
        # Use memory mapping for large npz files
        with np.load(input_file, mmap_mode='r', allow_pickle=True) as data:
            np_nodes = data['nodes']
        loading_time = time.time() - start_time
        nodes = np_nodes.tolist()
        tolist_time = time.time() - start_time - loading_time
        logger.info(f"npz file {input_file} loading time: {loading_time:.2f} seconds, tolist time: {tolist_time:.2f} seconds")
        return nodes

    for doc in json.load(open(input_file, 'r')):
        if doc['class_name'] == 'TextNode' and doc['text'] != '':
            nodes.append(TextNode.from_dict(doc))
        elif doc['class_name'] == 'ImageNode':
            nodes.append(ImageNode.from_dict(doc))
        else:
            continue
    return nodes

class SearchEngine:
    def __init__(self, dataset_dir='search_engine/corpus', node_dir_prefix='colqwen_ingestion',embed_model_name='vidore/colqwen2-v1.0'): # "vidore/colqwen2-v0.1"

        self.workers = 1

        self.dataset_dir = dataset_dir

        self.node_dir = os.path.join(self.dataset_dir, node_dir_prefix)
        self.vector_embed_model = VL_Embedding(model=embed_model_name, mode='image')
        self.query_engine = self.load_query_engine()

    def load_nodes(self):
        start_time = time.time()
        
        def parse_file(file,node_dir):
            parse_start = time.time()
            input_file = os.path.join(node_dir, file)
            suffix = input_file.split('.')[-1]
            if suffix not in ['node', 'npz']:
                return []
            nodes = nodefile2node(input_file)
            logger.info(f"File {file} parsing time: {time.time() - parse_start:.2f} seconds")
            return nodes

        def process_file(file):
            process_start = time.time()
            
            # Parse nodes
            nodes = parse_file(file, self.node_dir)
            parse_time = time.time() - process_start
            
            # Extract embeddings
            embed_start = time.time()
            raw_embeddings = [node.embedding for node in nodes]
            # Clean raw embeddings
            for node in nodes:
                node.embedding = None
            embed_extract_time = time.time() - embed_start
                
            # Process embeddings
            tensor_start = time.time()
            if raw_embeddings:
                embedding_lengths = [len(emb) for emb in raw_embeddings]
                if len(set(embedding_lengths)) == 1:
                    # If lengths are consistent, use batch processing
                    batch_size = len(raw_embeddings)
                    embeddings_tensor = torch.tensor(raw_embeddings, dtype=torch.float32)
                    embeddings_tensor = embeddings_tensor.view(batch_size, -1, 128).bfloat16()
                    batch_embeddings = [emb for emb in embeddings_tensor]
                else:
                    # If lengths are inconsistent, process one by one
                    batch_embeddings = []
                    for emb in raw_embeddings:
                        emb_tensor = torch.tensor(emb, dtype=torch.float32)
                        emb_tensor = emb_tensor.view(-1, 128).bfloat16()
                        batch_embeddings.append(emb_tensor)
                tensor_time = time.time() - tensor_start
                logger.info(f"File {file} processing times - Parse: {parse_time:.2f}s, Embed extract: {embed_extract_time:.2f}s, Tensor convert: {tensor_time:.2f}s")
                return nodes, batch_embeddings
            return nodes, []

        files = os.listdir(self.node_dir)
        parsed_files = []
        parsed_embeddings = []
        if len(files) > 0 and files[0].endswith('.npz'):
            # when loading big npz file, using single worker is the fastest.
            max_workers = 1 
        else:
            max_workers = min(8, os.cpu_count() or 1)  # Adjust workers based on CPU cores

        # Process files using multiprocessing
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_file, file) for file in files]
            for future in tqdm(as_completed(futures), total=len(files)):
                nodes, embeddings = future.result()
                parsed_files.extend(nodes)
                parsed_embeddings.extend(embeddings)
        
        return parsed_files, parsed_embeddings
        
    def load_query_engine(self):
        print('Loading nodes...')
        self.nodes, self.embedding_img = self.load_nodes()
        self.embedding_img = [tensor.to(self.vector_embed_model.embed_model.device) for tensor in tqdm(self.embedding_img, desc="Moving to Device")]
        self.image_nums = len(self.embedding_img)

    def load_node_postprocessors(self):
        return []
    def batch_search(self, queries: List[str]):
        start_time = time.time()
        batch_queries = self.vector_embed_model.processor.process_queries(queries).to(self.vector_embed_model.embed_model.device)
        with torch.no_grad():
            query_embeddings = self.vector_embed_model.embed_model(**batch_queries)
        scores = self.vector_embed_model.processor.score_multi_vector(query_embeddings, self.embedding_img, batch_size=256, device=self.vector_embed_model.embed_model.device)
        values, indices = torch.topk(scores, k=min(self.image_nums,10), dim=1)
        recall_results = [[self.nodes[idx].metadata['file_name'] for idx in row] for row in indices]
        logger.info(f"batch_search time taken: {(time.time() - start_time):.2f} seconds")
        return recall_results


if __name__ == '__main__':
    search_engine = SearchEngine(dataset_dir='search_engine/corpus',embed_model_name='vidore/colqwen2-v1.0')
    print(search_engine.batch_search(['o','a']))
    

    