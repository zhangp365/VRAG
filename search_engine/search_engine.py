import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, List, Mapping, Any, Dict
import json
from tqdm import tqdm
import torch

from llama_index.core.schema import TextNode, NodeRelationship, RelatedNodeInfo, ImageNode
from vl_embedding import VL_Embedding
import numpy as np
import time
import logging
logger = logging.getLogger(__name__)


def nodefile2node(input_file):
    nodes = []
    if input_file.endswith('.npz'):
        np_nodes = np.load(input_file, allow_pickle=True)['nodes']
        nodes = [node for node in np_nodes]
        del np_nodes
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
        def parse_file(file,node_dir):
            input_file = os.path.join(node_dir, file)
            suffix = input_file.split('.')[-1]
            if suffix not in ['node', 'npz']:
                return []
            nodes = nodefile2node(input_file)
            return nodes

        def process_batch(file_batch):
            batch_nodes = []
            raw_embeddings = []
            
            # Collect all nodes and raw embeddings
            for file in file_batch:
                nodes = parse_file(file, self.node_dir)
                batch_nodes.extend(nodes)
                raw_embeddings.extend([node.embedding for node in nodes])
                # Clean raw embeddings
                for node in nodes:
                    node.embedding = None
            
            # Process all embeddings in batch
            if raw_embeddings:
                # Check if all embedding lengths are consistent
                embedding_lengths = [len(emb) for emb in raw_embeddings]
                if len(set(embedding_lengths)) == 1:
                    # If lengths are consistent, use batch processing
                    batch_size = len(raw_embeddings)
                    # Convert to tensor at once
                    embeddings_tensor = torch.tensor(raw_embeddings, dtype=torch.float32)
                    # Convert to bfloat16 in batch
                    embeddings_tensor = embeddings_tensor.view(batch_size, -1, 128).bfloat16()
                    # Convert to list
                    batch_embeddings = [emb for emb in embeddings_tensor]
                else:
                    # If lengths are inconsistent, process one by one
                    batch_embeddings = []
                    for emb in raw_embeddings:
                        emb_tensor = torch.tensor(emb, dtype=torch.float32)
                        emb_tensor = emb_tensor.view(-1, 128).bfloat16()
                        batch_embeddings.append(emb_tensor)
            else:
                batch_embeddings = []
                
            return batch_nodes, batch_embeddings

        files = os.listdir(self.node_dir)
        parsed_files = []
        parsed_embeddings = []
        max_workers = 8
        batch_size = 100  # Number of files to process at a time

        # Split file list into multiple batches
        file_batches = [files[i:i + batch_size] for i in range(0, len(files), batch_size)]

        if max_workers == 1:
            for batch in tqdm(file_batches):
                nodes, embeddings = process_batch(batch)
                parsed_files.extend(nodes)
                parsed_embeddings.extend(embeddings)
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(process_batch, batch) for batch in file_batches]
                for future in tqdm(as_completed(futures), total=len(file_batches)):
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
    

    