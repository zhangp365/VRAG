import os
import sys
import json
import argparse

from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from pathlib import Path
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SimpleFileNodeParser
from llama_index.core import SimpleDirectoryReader

from vl_embedding import VL_Embedding
import logging
import time
import numpy as np
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description='Data Ingestion Processing')
    parser.add_argument('--compress', action='store_true', default=True,
                      help='Whether to compress node data (default: True)')
    parser.add_argument('--dataset_dir', type=str, default='./search_engine/corpus',
                      help='Path to dataset directory')
    parser.add_argument('--input_prefix', type=str, default='img',
                      help='Input directory prefix')
    parser.add_argument('--output_prefix', type=str, default='colqwen_ingestion',
                      help='Output directory prefix')
    parser.add_argument('--embed_model_name', type=str, default='vidore/colqwen2-v1.0',
                      help='Name of the embedding model')
    return parser.parse_args()

class Ingestion:
    def __init__(self, dataset_dir, input_prefix='img', output_prefix='colqwen_ingestion', 
                 embed_model_name='vidore/colqwen2-v1.0', compress=True):
        self.dataset_dir = dataset_dir
        self.input_dir  = os.path.join(dataset_dir, input_prefix)
        self.output_dir = os.path.join(dataset_dir, output_prefix)
        self.workers = 5
        self.embed_model_name = embed_model_name
        self.compress = compress
        self.reader = SimpleDirectoryReader(input_dir = self.input_dir)
        self.pipeline = IngestionPipeline(transformations=[
            SimpleFileNodeParser(),
            VL_Embedding(model=embed_model_name,mode='image')
        ])

    def ingestion_example(self, input_file, output_file):  
        documents = self.reader.load_file(Path(input_file),self.reader.file_metadata,self.reader.file_extractor)

        start_time = time.time()
        nodes = self.pipeline.run(documents=documents,num_workers=1, show_progress=False)
        logger.info(f"document {input_file} ingestion time taken: {time.time() - start_time} seconds")
        if self.compress:
            output_file = output_file.replace('.node', '.npz')
            np.savez_compressed(output_file, nodes=nodes)
        else:
            nodes_json = [node.to_dict() for node in nodes]
            with open(output_file, 'w') as json_file:
                json.dump(nodes_json, json_file, indent=2, ensure_ascii=False)        
        return True
    
    def ingestion_multi_session(self):
        os.makedirs(self.output_dir, exist_ok=True)
        file_to_process = []
        for file in os.listdir(self.input_dir):
            file_prefix,_ = os.path.splitext(file)
            input_file = os.path.join(self.input_dir, file)
            if self.compress:
                output_file = os.path.join(self.output_dir, file_prefix) + '.npz'
            else:
                output_file = os.path.join(self.output_dir, file_prefix) + '.node'
            if not os.path.exists(output_file):
                file_to_process.append((input_file, output_file))
        if self.workers == 1:
            for input_file, output_file in tqdm(file_to_process):
                self.ingestion_example(input_file, output_file)
        else:
            with ThreadPoolExecutor(max_workers=self.workers) as executor:
                future_to_file = {executor.submit(self.ingestion_example, input_file, output_file): (input_file, output_file) for input_file, output_file in file_to_process}
                for future in tqdm(as_completed(future_to_file), total=len(file_to_process), desc='Processing files'):
                    result_type = future.result()
    


if __name__ == '__main__':
    args = parse_args()
    ingestion = Ingestion(
        dataset_dir=args.dataset_dir,
        input_prefix=args.input_prefix,
        output_prefix=args.output_prefix,
        embed_model_name=args.embed_model_name,
        compress=args.compress
    )
    ingestion.ingestion_multi_session()

