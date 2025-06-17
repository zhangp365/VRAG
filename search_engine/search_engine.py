import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List
import json
from tqdm import tqdm
import torch
import numpy as np
import time
import logging
import pickle
from llama_index.core.schema import TextNode,  ImageNode
from vl_embedding import VL_Embedding

logger = logging.getLogger(__name__)


def nodefile2node(input_file):
    nodes, embeddings = [], []
    if input_file.endswith(".pkl"):
        with open(input_file, 'rb') as f:
            nodes = pickle.load(f)
    elif input_file.endswith('.npz'):
        with np.load(input_file, allow_pickle=True) as data:
            np_nodes = data['nodes']
        nodes = np_nodes.tolist()
    else:
        nodes = json.load(open(input_file, 'r'))

    result_nodes = []
    for node in nodes:
        node_embedding = node.embedding if isinstance(node, ImageNode) else node['embedding']
        if isinstance(node_embedding, torch.Tensor):
            embeddings.append(node_embedding.view(-1, 128).to(torch.bfloat16))
        else:
            embeddings.append(torch.tensor(node_embedding, dtype=torch.float32).view(-1, 128).to(torch.bfloat16))
        if isinstance(node, dict):
            node['embedding'] = None
            if node['class_name'] == 'TextNode' and node['text'] != '':
                result_nodes.append(TextNode.from_dict(node))
            elif node['class_name'] == 'ImageNode':
                result_nodes.append(ImageNode.from_dict(node))
        else:
            node.embedding = None
            result_nodes.append(node)
    return result_nodes, embeddings

class SearchEngine:
    def __init__(self, dataset_dir='search_engine/corpus', node_dir_prefix='colqwen_ingestion',embed_model_name='vidore/colqwen2-v1.0'): # "vidore/colqwen2-v0.1"

        self.workers = 1

        self.dataset_dir = dataset_dir

        self.node_dir = os.path.join(self.dataset_dir, node_dir_prefix)
        self.vector_embed_model = VL_Embedding(model=embed_model_name, mode='image')
        self.query_engine = self.load_query_engine()

    def load_nodes(self):
        
        def parse_file(file,node_dir):
            parse_start = time.time()
            input_file = os.path.join(node_dir, file)
            suffix = input_file.split('.')[-1]
            if suffix not in ['node', 'npz', 'pkl']:
                return [],[]
            nodes, embeddings = nodefile2node(input_file)
            logger.info(f"File {file} parsing time: {time.time() - parse_start:.2f} seconds")
            return nodes, embeddings        

        files = os.listdir(self.node_dir)
        parsed_files = []
        parsed_embeddings = []
        if len(files) > 0 and files[0].endswith(('.npz', '.pkl')):
            # when loading big npz file, using single worker is the fastest.
            max_workers = 1 
        else:
            max_workers = min(8, os.cpu_count() or 1)  # Adjust workers based on CPU cores

        # Process files using multiprocessing
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(parse_file, file, self.node_dir) for file in files]
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
        logger.info(f"search results: queries {queries}, top 10 scores: {values}")
        recall_results = [[self.nodes[idx].metadata['file_name'] for idx in row] for row in indices]
        logger.info(f"batch_search time taken: {(time.time() - start_time):.2f} seconds")
        return recall_results


if __name__ == '__main__':
    search_engine = SearchEngine(dataset_dir='search_engine/corpus',embed_model_name='vidore/colqwen2-v1.0')
    print(search_engine.batch_search(['o','a']))
    

    