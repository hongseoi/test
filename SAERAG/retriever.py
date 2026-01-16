from typing import List, Dict, Tuple
import os
import time
import tqdm
import uuid
import numpy as np
import torch
import logging
import pandas as pd
from transformers import AutoTokenizer, AutoModel

# BEIR 관련 임포트
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.lexical import BM25Search as OriginalBM25Search # 이름 변경
from beir.retrieval.search.lexical.elastic_search import ElasticSearch

logging.basicConfig(level=logging.INFO) 
logger = logging.getLogger(__name__)

# ==========================================
# [중요 수정] 추상 메서드 오류 해결을 위한 클래스 재정의
# OriginalBM25Search를 상속받아 빈 메서드를 구현한 새 클래스를 만듭니다.
# ==========================================
class FixedBM25Search(OriginalBM25Search):
    def encode(self, query):
        """추상 메서드 오류 방지용 빈 함수"""
        pass
    
    def search_from_files(self, queries, num_docs=10):
        """추상 메서드 오류 방지용 빈 함수"""
        pass

# BM25Search라는 이름이 FixedBM25Search를 가리키도록 설정
BM25Search = FixedBM25Search 
# ==========================================

def get_random_doc_id():
    return f'_{uuid.uuid4()}'

class BM25:
    def __init__(
        self,
        tokenizer: AutoTokenizer = None,
        index_name: str = None,
        engine: str = 'elasticsearch',
        **search_engine_kwargs,
    ):
        self.tokenizer = tokenizer
        # load index
        assert engine in {'elasticsearch', 'bing'}
        if engine == 'elasticsearch':
            self.max_ret_topk = 1000
            # 여기서 FixedBM25Search(위에서 BM25Search로 이름 바꿈)가 사용됨
            self.retriever = EvaluateRetrieval(
                BM25Search(index_name=index_name, hostname='localhost', initialize=False, number_of_shards=1),
                k_values=[self.max_ret_topk])

    def retrieve(
        self,
        queries: List[str],  # (bs,)
        topk: int = 1,
        max_query_length: int = None,
    ):
        assert topk <= self.max_ret_topk
        bs = len(queries)

        # truncate queries
        if max_query_length:
            ori_ps = self.tokenizer.padding_side
            ori_ts = self.tokenizer.truncation_side
            # truncate/pad on the left side
            self.tokenizer.padding_side = 'left'
            self.tokenizer.truncation_side = 'left'
            tokenized = self.tokenizer(
                queries,
                truncation=True,
                padding=True,
                max_length=max_query_length,
                add_special_tokens=False,
                return_tensors='pt')['input_ids']
            self.tokenizer.padding_side = ori_ps
            self.tokenizer.truncation_side = ori_ts
            queries = self.tokenizer.batch_decode(tokenized, skip_special_tokens=True)

        # retrieve
        results = self.retriever.retrieve(
            None, dict(zip(range(len(queries)), queries)), disable_tqdm=True)
        
        if isinstance(results, tuple):
            results = results[0]

        # prepare outputs
        docids: List[str] = []
        docs: List[str] = []
        for qid, query in enumerate(queries):
            qid = str(qid)
            _docids: List[str] = []
            _docs: List[str] = []
            if qid in results:
                sorted_hits = sorted(results[qid].items(), key=lambda item: item[1], reverse=True)
                for did, score in sorted_hits:
                    if isinstance(score, tuple):
                        _docids.append(did)
                        _docs.append(score[1]) 
                    else:
                        _docids.append(did)
                        _docs.append("") 
                    
                    if len(_docids) >= topk:
                        break
            
            if len(_docids) < topk:  # add dummy docs
                _docids += [get_random_doc_id() for _ in range(topk - len(_docids))]
                _docs += [''] * (topk - len(_docs))
            docids.extend(_docids)
            docs.extend(_docs)

        docids = np.array(docids).reshape(bs, topk)  # (bs, topk)
        docs = np.array(docs).reshape(bs, topk)  # (bs, topk)
        return docids, docs


# ==========================================
# 메서드 오버라이딩 (FixedBM25Search에 적용)
# ==========================================
def bm25search_search(self, corpus: Dict[str, Dict[str, str]], queries: Dict[str, str], top_k: int, *args, **kwargs) -> Dict[str, Dict[str, float]]:
    if self.initialize:
        self.index(corpus)
        time.sleep(self.sleep_for)

    query_ids = list(queries.keys())
    queries_list = [queries[qid] for qid in query_ids]

    final_results: Dict[str, Dict[str, Tuple[float, str]]] = {}
    
    for start_idx in tqdm.trange(0, len(queries_list), self.batch_size, desc='que', disable=kwargs.get('disable_tqdm', False)):
        end_idx = start_idx + self.batch_size
        query_ids_batch = query_ids[start_idx:end_idx]
        batch_queries = queries_list[start_idx:end_idx]
        
        results = self.es.lexical_multisearch(
            texts=batch_queries,
            top_hits=top_k)
        
        for (query_id, hit) in zip(query_ids_batch, results):
            scores = {}
            for corpus_id, score, text in hit['hits']:
                scores[corpus_id] = (score, text)
            final_results[query_id] = scores

    return final_results

# 클래스 자체를 바꿨으므로, 해당 클래스의 메서드를 교체
FixedBM25Search.search = bm25search_search


def elasticsearch_lexical_multisearch(self, texts: List[str], top_hits: int, skip: int = 0) -> Dict[str, object]:
    request = []
    assert skip + top_hits <= 10000, "Elastic-Search Window too large, Max-Size = 10000"

    for text in texts:
        req_head = {"index" : self.index_name, "search_type": "dfs_query_then_fetch"}
        req_body = {
            "_source": True, 
            "query": {
                "multi_match": {
                    "query": text,
                    "type": "best_fields",
                    "fields": [self.title_key, self.text_key],
                    "tie_breaker": 0.5
                    }
                },
            "size": skip + top_hits,
            }
        request.extend([req_head, req_body])

    res = self.es.msearch(body = request)

    result = []
    for resp in res["responses"]:
        responses = resp["hits"]["hits"][skip:] if 'hits' in resp else []
        hits = []
        for hit in responses:
            text_content = hit['_source'].get('txt', hit['_source'].get('text', ''))
            hits.append((hit["_id"], hit['_score'], text_content))
        result.append(self.hit_template(es_res=resp, hits=hits))
    return result

ElasticSearch.lexical_multisearch = elasticsearch_lexical_multisearch


def elasticsearch_hit_template(self, es_res: Dict[str, object], hits: List[Tuple[str, float]]) -> Dict[str, object]:
    result = {
        'meta': {
            'total': es_res['hits']['total']['value'] if 'hits' in es_res else None,
            'took': es_res['took'] if 'took' in es_res else None,
            'num_hits': len(hits)
        },
        'hits': hits,
    }
    return result

ElasticSearch.hit_template = elasticsearch_hit_template


class SGPT:
    cannot_encode_id = [6799132, 6799133, 6799134, 6799135, 6799136, 6799137, 6799138, 6799139, 8374206, 8374223, 9411956, 
        9885952, 11795988, 11893344, 12988125, 14919659, 16890347, 16898508]

    def __init__(
        self, 
        model_name_or_path,
        sgpt_encode_file_path,
        passage_file
    ):
        logger.info(f"Loading SGPT model from {model_name_or_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModel.from_pretrained(model_name_or_path, device_map="auto")
        self.model.eval()
        
        self.SPECB_QUE_BOS = self.tokenizer.encode("[", add_special_tokens=False)[0]
        self.SPECB_QUE_EOS = self.tokenizer.encode("]", add_special_tokens=False)[0]
        self.SPECB_DOC_BOS = self.tokenizer.encode("{", add_special_tokens=False)[0]
        self.SPECB_DOC_EOS = self.tokenizer.encode("}", add_special_tokens=False)[0]

        logger.info(f"Building SGPT indexes")

        self.p_reps = []
        encode_file_path = sgpt_encode_file_path
        dir_names = sorted(os.listdir(encode_file_path))
        
        split_parts = 0
        while True:
            split_parts += 1
            if not any(d.startswith(f'{split_parts}_') for d in dir_names):
                break
        
        num_gpus = torch.cuda.device_count()
        if num_gpus == 0:
            logger.warning("No GPU found. Loading indices to CPU.")
            device_map = lambda i: 'cpu'
        else:
            device_map = lambda i: f'cuda:{i % num_gpus}'

        dir_point = 0
        pbar = tqdm.tqdm(total=len(dir_names))

        for i in range(split_parts):
            target_device = device_map(i)
            start_point = dir_point
            
            current_files = []
            while dir_point < len(dir_names) and dir_names[dir_point].startswith(f'{i}_'):
                current_files.append(dir_names[dir_point])
                dir_point += 1
            
            for filename in current_files:
                pbar.update(1)
                full_path = os.path.join(encode_file_path, filename)
                tp = torch.load(full_path, map_location='cpu')

                def get_norm(matrix):
                    norm = matrix.norm(dim=1)
                    norm = torch.where(norm == 0, torch.tensor(1.0, device=norm.device), norm)
                    return norm.view(-1, 1)

                sz = tp.shape[0] // 2
                tp1 = tp[:sz, :]
                tp2 = tp[sz:, :]
                
                self.p_reps.append((tp1.to(target_device), get_norm(tp1).to(target_device)))
                self.p_reps.append((tp2.to(target_device), get_norm(tp2).to(target_device)))
            
        docs_file = passage_file
        df = pd.read_csv(docs_file, delimiter='\t')
        self.docs = list(df['text'])


    def tokenize_with_specb(self, texts, is_query):
        batch_tokens = self.tokenizer(texts, padding=False, truncation=True)   
        
        for seq, att in zip(batch_tokens["input_ids"], batch_tokens["attention_mask"]):
            if is_query:
                seq.insert(0, self.SPECB_QUE_BOS)
                seq.append(self.SPECB_QUE_EOS)
            else:
                seq.insert(0, self.SPECB_DOC_BOS)
                seq.append(self.SPECB_DOC_EOS)
            att.insert(0, 1)
            att.append(1)
        
        batch_tokens = self.tokenizer.pad(batch_tokens, padding=True, return_tensors="pt")
        return batch_tokens

    def get_weightedmean_embedding(self, batch_tokens):
        device = self.model.device
        batch_tokens = {k: v.to(device) for k, v in batch_tokens.items()}

        with torch.no_grad():
            output = self.model(**batch_tokens, output_hidden_states=True, return_dict=True)
            last_hidden_state = output.last_hidden_state

        weights = (
            torch.arange(start=1, end=last_hidden_state.shape[1] + 1)
            .unsqueeze(0)
            .unsqueeze(-1)
            .expand(last_hidden_state.size())
            .float().to(device)
        )

        input_mask_expanded = (
            batch_tokens["attention_mask"]
            .unsqueeze(-1)
            .expand(last_hidden_state.size())
            .float()
        )

        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded * weights, dim=1)
        sum_mask = torch.sum(input_mask_expanded * weights, dim=1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        embeddings = sum_embeddings / sum_mask

        return embeddings

    def retrieve(
        self, 
        queries: List[str], 
        topk: int = 1,
    ):
        q_reps = self.get_weightedmean_embedding(
            self.tokenize_with_specb(queries, is_query=True)
        )
        q_reps.requires_grad_(False)
        q_reps_trans = torch.transpose(q_reps, 0, 1)

        topk_values_list = []
        topk_indices_list = []
        prev_count = 0
        
        for p_rep, p_rep_norm in self.p_reps:
            curr_device = p_rep.device
            curr_q_reps = q_reps_trans.to(curr_device)
            
            sim = p_rep @ curr_q_reps
            sim = sim / p_rep_norm
            
            topk_values, topk_indices = torch.topk(sim, k=topk, dim=0)
            
            topk_values_list.append(topk_values.to('cpu'))
            topk_indices_list.append(topk_indices.to('cpu') + prev_count)
            prev_count += p_rep.shape[0]

        all_topk_values = torch.cat(topk_values_list, dim=0)
        global_topk_values, global_topk_indices = torch.topk(all_topk_values, k=topk, dim=0)

        psgs = []
        for qid in range(q_reps.shape[0]):
            ret = []
            for j in range(topk):
                idx = global_topk_indices[j][qid].item()
                shard_idx = idx // topk
                rank_in_shard = idx % topk
                real_doc_idx = topk_indices_list[shard_idx][rank_in_shard][qid].item()
                
                psg = self.docs[real_doc_idx]
                ret.append(psg)
            psgs.append(ret)
        return psgs