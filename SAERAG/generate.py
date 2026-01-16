import numpy as np
import logging
import spacy
import torch
from math import exp
from scipy.special import softmax
from retriever import BM25, SGPT
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import joblib
import pandas as pd
import torch.nn as nn
from transformers import StoppingCriteria, StoppingCriteriaList
import torch.nn.functional as F
# 기존 import 아래에 추가
from sae_lens import SAE

logging.basicConfig(level=logging.INFO) 
logger = logging.getLogger(__name__)

nlp = spacy.load("en_core_web_sm")

class BasicGenerator:
    def __init__(self, model_name_or_path):
        logger.info(f"Loading model from {model_name_or_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        
        # [수정 1] 패딩 설정
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = 'left'

        self.model_config = AutoConfig.from_pretrained(model_name_or_path,
                                    trust_remote_code = "falcon" in model_name_or_path)
        
        # [수정 2] 단일 GPU 강제 사용 (A6000 메모리 충분함 -> 분산 버그 방지)
        # device_map="auto" 대신 cuda:0으로 고정
        logger.info("Forcing model to load on GPU 0 (Single Device Mode)...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, 
            device_map={'': 0}, # 모든 모듈을 GPU 0에 할당
            torch_dtype=torch.float32, 
            attn_implementation="eager",
            trust_remote_code = "falcon" in model_name_or_path
        )
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        
        # [수정 3] 임베딩 크기 강제 리사이징 (불일치 방지)
        current_embedding_size = self.model.get_input_embeddings().weight.shape[0]
        tokenizer_len = len(self.tokenizer)
        logger.info(f"Original Embedding Size: {current_embedding_size}, Tokenizer Len: {tokenizer_len}")
        
        # 무조건 리사이징하여 안전공간 확보
        new_size = max(current_embedding_size, tokenizer_len)
        self.model.resize_token_embeddings(new_size)
        self.vocab_limit = new_size
        logger.info(f"Final Vocab Limit: {self.vocab_limit}")
        
        if hasattr(self.model.config, "max_position_embeddings"):
            self.max_pos = self.model.config.max_position_embeddings
        else:
            self.max_pos = 2048

        self.model.config.use_cache = False
        if self.model_config.model_type == "llama":
            self.space_token = "▁"
        else:
            self.space_token = self.tokenizer.tokenize(' ')[0]

    def _sanitize_inputs(self, input_ids):
        """입력 ID 범위 강제 조정 및 디버깅"""
        # 디버깅: 입력값 범위 확인
        max_val = input_ids.max().item()
        if max_val >= self.vocab_limit:
            logger.warning(f"Found input ID {max_val} >= vocab limit {self.vocab_limit}. Clamping...")
            input_ids = torch.clamp(input_ids, min=0, max=self.vocab_limit - 1)
        
        if (input_ids < 0).any():
             input_ids = torch.clamp(input_ids, min=0)
             
        return input_ids

    def _sanitize_outputs(self, output_ids):
        """출력 ID 범위 강제 조정 및 리스트 변환"""
        if (output_ids >= self.vocab_limit).any() or (output_ids < 0).any():
            output_ids = torch.clamp(output_ids, min=0, max=self.vocab_limit - 1)
        return output_ids.tolist()

    def generate(self, input_text, max_length, return_logprobs=False):
        safe_input_len = self.max_pos - max_length - 50
        inputs = self.tokenizer(
            input_text, 
            return_tensors="pt", 
            truncation=True, 
            max_length=safe_input_len,
            padding=True
        )
        
        # 모델 디바이스로 이동 (cuda:0)
        input_ids = inputs['input_ids'].to(self.model.device)
        attention_mask = inputs['attention_mask'].to(self.model.device)
        input_length = input_ids.shape[1]

        # [수정 4] 안전장치
        input_ids = self._sanitize_inputs(input_ids)

        if return_logprobs:
            outputs = self.model.generate(
                input_ids = input_ids, 
                attention_mask = attention_mask,
                max_new_tokens = max_length, 
                return_dict_in_generate = True, 
                output_scores = True,
                pad_token_id = self.tokenizer.pad_token_id,
                do_sample = False
            )
            transition_scores = self.model.compute_transition_scores(
                outputs.sequences, outputs.scores, normalize_logits=True
            )

            generated_tokens = outputs.sequences[:, input_length:]
            gen_list = self._sanitize_outputs(generated_tokens[0])
            
            text = self.tokenizer.decode(gen_list, skip_special_tokens=True)
            tokens = self.tokenizer.convert_ids_to_tokens(gen_list)
            logprobs = transition_scores[0]
            logprobs = [p.cpu().numpy() for p in logprobs]
            return text, tokens, logprobs
        
        else:
            outputs = self.model.generate(
                input_ids = input_ids, 
                max_new_tokens = max_length, 
                attention_mask = attention_mask,
                pad_token_id = self.tokenizer.pad_token_id,
                do_sample = False
            )
            generated_tokens = outputs[:, input_length:]
            gen_list = self._sanitize_outputs(generated_tokens[0])
            
            text = self.tokenizer.decode(gen_list, skip_special_tokens=True)
            return text, None, None
    
    def generate_attn(self, input_text, max_length, solver="max", use_entropy = False, use_logprob = False):
        safe_input_len = self.max_pos - max_length - 50
        inputs = self.tokenizer(
            input_text, 
            return_tensors="pt", 
            truncation=True, 
            max_length=safe_input_len,
            padding=True
        )
        
        input_ids = inputs['input_ids'].to(self.model.device)
        attention_mask = inputs['attention_mask'].to(self.model.device)
        input_length = input_ids.shape[1]

        input_ids = self._sanitize_inputs(input_ids)

        outputs = self.model.generate(
            input_ids = input_ids, 
            attention_mask = attention_mask,
            max_new_tokens = max_length, 
            return_dict_in_generate = True, 
            output_scores = True,
            pad_token_id = self.tokenizer.pad_token_id,
            do_sample = False 
        )
        generated_tokens = outputs.sequences[:, input_length:]
        gen_list = self._sanitize_outputs(generated_tokens[0])
        
        tokens = self.tokenizer.convert_ids_to_tokens(gen_list)
        text = self.tokenizer.decode(gen_list, skip_special_tokens=True)

        # merge tokens
        range_ = []
        for i, t in enumerate(tokens):
            # gen_list[i]가 range 안에 있는지 확인 (IndexError 방지)
            token_id = gen_list[i] if i < len(gen_list) else -1
            if i == 0 or t.startswith(self.space_token) or token_id == 13 or t == '</s>':
                range_.append([i, i])
            else:
                range_[-1][-1] += 1

        with torch.no_grad():
            final_input_ids = self._sanitize_inputs(outputs.sequences)
            
            model_outputs = self.model(
                final_input_ids,
                attention_mask=torch.ones_like(final_input_ids),
                output_attentions=True
            )
        
        if model_outputs.attentions is None:
             atten = torch.zeros((1, final_input_ids.shape[1], final_input_ids.shape[1])).to(self.model.device)
        else:
             atten = model_outputs.attentions[-1][0]

        if solver == "max": 
            mean_atten, _ = torch.max(atten, dim=1)
            mean_atten = torch.mean(mean_atten, dim=0)
        elif solver == "avg":
            mean_atten = torch.sum(atten, dim=1)
            mean_atten = torch.mean(mean_atten, dim=0)
            for i in range(mean_atten.shape[0]):
                mean_atten[i] /= (mean_atten.shape[0] - i)
        elif solver == "last_token":
            mean_atten = torch.mean(atten[:, -1], dim=0)
        else:
            raise NotImplementedError
        
        if mean_atten.shape[0] > 1 and len(tokens) > 0 and tokens[0] == '</s>':
             if len(mean_atten) > 1:
                mean_atten = mean_atten / (sum(mean_atten[1:]).item() + 1e-9)
            
        # regular tokens
        seqlist = []
        attns = []
        for r in range_:
            tokenseq = "".join(tokens[r[0]: r[1]+1]).replace(self.space_token, "")
            start, end = r[0], r[1]+1
            if end > len(mean_atten): end = len(mean_atten)
            if start >= end: value = 0.0
            else: value = sum(mean_atten[start: end]).item()
            
            seqlist.append(tokenseq)
            attns.append(value)

        # -log prob
        if use_logprob:
            transition_scores = self.model.compute_transition_scores(
                outputs.sequences, outputs.scores, normalize_logits=True
            )
            logprobs = transition_scores[0]
            logprobs = [p.cpu().numpy() for p in logprobs]
            seqlogprobs = []
            for r in range_:
                start, end = r[0], r[1]+1
                if end > len(logprobs): end = len(logprobs)
                if start >= end: logprobseq = -100.0 
                else: logprobseq = sum(logprobs[start:end]) / (end - start)
                seqlogprobs.append(logprobseq)
        else:
            seqlogprobs = None

        # entropy
        if use_entropy:
            tmp = []
            for v in outputs.scores:
                tmp.append(v.cpu())
            softmax_probs = softmax(tmp, axis=-1)
            entropies = -np.sum(softmax_probs * np.log(softmax_probs + 1e-10), axis=-1)
            entropies = [v[0] for v in entropies]
            seqentropies = []
            for r in range_:
                start, end = r[0], r[1]+1
                if end > len(entropies): end = len(entropies)
                if start >= end: entropyseq = 0.0
                else: entropyseq = sum(entropies[start:end]) / (end - start)
                seqentropies.append(entropyseq) 
        else:
            seqentropies = None 

        return text, seqlist, attns, seqlogprobs, seqentropies
    
    # BasicGenerator 클래스 내부에 추가
    def generate_with_hidden(self, input_text, max_length, output_layer_index=-1):
        """생성된 텍스트와 해당 텍스트의 Hidden States를 반환"""
        safe_input_len = self.max_pos - max_length - 50
        inputs = self.tokenizer(
            input_text, 
            return_tensors="pt", 
            truncation=True, 
            max_length=safe_input_len,
            padding=True
        )
        
        input_ids = inputs['input_ids'].to(self.model.device)
        attention_mask = inputs['attention_mask'].to(self.model.device)
        input_length = input_ids.shape[1]
        input_ids = self._sanitize_inputs(input_ids)

        # 1. 텍스트 생성
        outputs = self.model.generate(
            input_ids = input_ids, 
            attention_mask = attention_mask,
            max_new_tokens = max_length, 
            pad_token_id = self.tokenizer.pad_token_id,
            do_sample = False
        )
        
        generated_sequences = outputs
        gen_ids = generated_sequences[:, input_length:]
        gen_list = self._sanitize_outputs(gen_ids[0])
        text = self.tokenizer.decode(gen_list, skip_special_tokens=True)

        # 2. Hidden State 추출을 위한 Forward Pass (생성된 시퀀스 전체 입력)
        # (생성 시 캐싱 문제로 인해, 정확한 Hidden State를 얻기 위해 완성된 문장으로 다시 forward)
        with torch.no_grad():
            final_inputs = self._sanitize_inputs(generated_sequences)
            model_out = self.model(
                final_inputs, 
                output_hidden_states=True
            )
        
        # 지정된 레이어(output_layer_index)의 Hidden States 중 생성된 토큰 부분만 추출
        all_hidden = model_out.hidden_states[output_layer_index] # [Batch, Total_Seq, Dim]
        gen_hidden = all_hidden[:, input_length:, :] # [Batch, New_Tokens, Dim]

        return text, gen_hidden


class Counter:
    def __init__(self):
        self.retrieve = 0
        self.generate = 0
        self.hallucinated = 0
        self.token = 0
        self.sentence = 0

    def add_generate(self, text, tokenizer):
        self.generate += 1
        ids = tokenizer(text, return_tensors="pt")['input_ids'][0].tolist()
        self.token += len(ids)
        sentences = [sent.text for sent in nlp(text).sents]
        self.sentence += len(sentences)

    def calc(self, other_counter):
        return {
            "retrieve_count": self.retrieve - other_counter.retrieve, 
            "generate_count": self.generate - other_counter.generate,
            "hallucinated_count": self.hallucinated - other_counter.hallucinated, 
            "token_count": self.token - other_counter.token, 
            "sentence_count": self.sentence - other_counter.sentence 
        }
          

class BasicRAG:
    def __init__(self, args):
        args = args.__dict__ 
        for k, v in args.items():
            setattr(self, k, v)
        self.generator = BasicGenerator(self.model_name_or_path)
        if "retriever" in self.__dict__:
            self.retriever_type = self.retriever
            if self.retriever_type == "BM25":
                self.retriever = BM25(
                    tokenizer = self.generator.tokenizer, 
                    index_name = "wiki" if "es_index_name" not in args else self.es_index_name, 
                    engine = "elasticsearch",
                )
            elif self.retriever_type == "SGPT":
                self.retriever = SGPT(
                    model_name_or_path = self.sgpt_model_name_or_path, 
                    sgpt_encode_file_path = self.sgpt_encode_file_path,
                    passage_file = self.passage_file
                )
            else:
                raise NotImplementedError
        
        self.counter = Counter()

    def retrieve(self, query, topk=1, max_query_length=64):
        self.counter.retrieve += 1
        if self.retriever_type == "BM25":
            _docs_ids, docs = self.retriever.retrieve(
                queries = [query], 
                topk = topk, 
                max_query_length = max_query_length,
            )
            return docs[0]
        elif self.retriever_type == "SGPT":
            docs = self.retriever.retrieve(
                queries = [query], 
                topk = topk,
            )
            return docs[0] 
        else:
            raise NotImplementedError
    
    def get_top_sentence(self, text):
        sentences = [sent.text.strip() for sent in nlp(text).sents]
        sentences = [sent for sent in sentences if len(sent) > 0]
        return sentences[0] if len(sentences) > 0 else ""

    def get_last_sentence(self, text):
        sentences = [sent.text.strip() for sent in nlp(text).sents]
        sentences = [sent for sent in sentences if len(sent) > 0]
        return sentences[-1] if len(sentences) > 0 else "" 
    
    def inference(self, question, demo, case):
        # non-retrieval
        assert self.query_formulation == "direct"
        prompt = "".join([d["case"]+"\n" for d in demo])
        prompt += case
        text, _, _ = self.generator.generate(prompt, self.generate_max_length)
        if self.use_counter == True:
            self.counter.add_generate(text, self.generator.tokenizer)
        return text
    

class SingleRAG(BasicRAG):
    def __init__(self, args):
        super().__init__(args)
    
    def inference(self, question, demo, case):
        assert self.query_formulation == "direct"
        docs = self.retrieve(question, topk=self.retrieve_topk)
        # 对 topk 个 passage 生成 prompt
        prompt = "".join([d["case"]+"\n" for d in demo])
        prompt += "Context:\n"
        for i, doc in enumerate(docs):
            prompt += f"[{i+1}] {doc}\n"
        prompt += "Answer in the same format as before.\n"
        prompt += case
        text, _, _ = self.generator.generate(prompt, self.generate_max_length)
        if self.use_counter == True:
            self.counter.add_generate(text, self.generator.tokenizer)
        return text


class FixLengthRAG(BasicRAG):
    def __init__(self, args):
        super().__init__(args)
    
    def inference(self, question, demo, case):
        assert self.query_formulation == "direct"
        text = ""
        retrieve_question = question
        while True:
            old_len = len(text)
            docs = self.retrieve(retrieve_question, topk=self.retrieve_topk)
            prompt = "".join([d["case"]+"\n" for d in demo])
            prompt += "Context:\n"
            for i, doc in enumerate(docs):
                prompt += f"[{i+1}] {doc}\n"
            prompt += "Answer in t he same format as before.\n"
            prompt += case + " " + text
            if self.method == "fix-length-retrieval":
                new_text, _, _ = self.generator.generate(prompt, self.fix_length)
                if self.use_counter == True:
                    self.counter.add_generate(new_text, self.generator.tokenizer)
                text = text.strip() + " " + new_text.strip()
                retrieve_question = new_text.strip()
            else:
                # fix sentence
                new_text, _, _ = self.generator.generate(prompt, self.generate_max_length)
                if self.use_counter == True:
                    self.counter.add_generate(new_text, self.generator.tokenizer)
                new_text = new_text.strip()
                sentences = list(nlp(new_text).sents)
                sentences = [str(sent).strip() for sent in sentences]
                if len(sentences) == 0:
                    break
                text = text.strip() + " " + str(sentences[0])
                retrieve_question = sentences[0]
            
            # 判断 token 的个数要少于 generate_max_length 
            tokens_count = len(self.generator.tokenizer.encode(text))
            if tokens_count > self.generate_max_length or len(text) <= old_len or "the answer is" in text:
                break
        return text


class TokenRAG(BasicRAG):
    def __init__(self, args):
        super().__init__(args)

    def modifier(self, text, tokens, logprobs):
        sentences = [sent.text.strip() for sent in nlp(text).sents]
        sentences = [sent for sent in sentences if len(sent) > 0]

        tid = 0
        for sid, sent in enumerate(sentences):
            pos = 0
            tr = tid
            while tr < len(tokens):
                apr = sent[pos:].find(tokens[tr])
                if apr == -1:
                    break
                pos = apr + len(tokens[tr])
                tr += 1
            probs = [1 - exp(v) for v in logprobs[tid:tr+1]]
            probs = np.array(probs)
            p = {
                "avg": np.mean,
                "max": np.max,
                "min": np.min,
            }.get(self.sentence_solver, lambda x: 0)(probs)
            if p > self.hallucination_threshold: # hallucination
                # keep sentences before hallucination 
                prev = "" if sid == 0 else " ".join(sentences[:sid])
                # replace all hallucinated tokens in current sentence with [xxx]
                curr = sentences[sid]
                pos = 0
                for prob, tok in zip(probs, tokens[tid:tr+1]):
                    apr = curr[pos:].find(tok) + pos
                    if prob > self.hallucination_threshold:
                        curr = curr[:apr] + "[xxx]" + curr[apr+len(tok):]
                        pos = apr + len("[xxx]")
                    else:
                        pos = apr + len(tok)
                return prev, curr, True
            tid = tr + 1
        
        # No hallucination
        return text, None, False
    
    def inference(self, question, demo, case):
        # assert self.query_formulation == "direct"
        text = ""
        while True:
            old_len = len(text)
            prompt = "".join([d["case"]+"\n" for d in demo])
            prompt += case + " " + text
            new_text, tokens, logprobs = self.generator.generate(
                prompt, 
                self.generate_max_length, 
                return_logprobs=True
            )
            if self.use_counter == True:
                self.counter.add_generate(new_text, self.generator.tokenizer)
            ptext, curr, hallucination = self.modifier(new_text, tokens, logprobs)
            if not hallucination:
                text = text.strip() + " " + new_text.strip()
            else:
                if self.query_formulation == "direct":
                    retrieve_question = curr.replace("[xxx]", "")
                elif self.query_formulation == "forward_all":
                    tmp_all = [question, text, ptext]
                    retrieve_question = " ".join(s for s in tmp_all if len(s) > 0)
                else:
                    raise NotImplemented

                docs = self.retrieve(retrieve_question, topk=self.retrieve_topk)
                prompt = "".join([d["case"]+"\n" for d in demo])
                prompt += "Context:\n"
                for i, doc in enumerate(docs):
                    prompt += f"[{i+1}] {doc}\n"
                prompt += "Answer in the same format as before.\n"
                prompt += case + " " + text + " " + ptext.strip()
                new_text, _, _ = self.generator.generate(prompt, self.generate_max_length)
                if self.use_counter == True:
                    self.counter.add_generate(new_text, self.generator.tokenizer)
                    self.counter.hallucinated += 1
                text = text.strip() + " " + ptext.strip() + " " + new_text.strip()
            
            # 判断 token 的个数要少于 generate_max_length 
            tokens_count = len(self.generator.tokenizer.encode(text))
            if tokens_count > self.generate_max_length or len(text) <= old_len or "the answer is" in text:
                break
        return text
    

class EntityRAG(TokenRAG):
    def __init__(self, args):
        super().__init__(args)
    
    def modifier(self, text, tokens, logprobs):
        sentences = [sent.text.strip() for sent in nlp(text).sents]
        sentences = [sent for sent in sentences if len(sent) > 0]

        entity = []
        for sent in sentences:
            doc = nlp(sent)
            li = [ent.text for ent in doc.ents]
            entity.append(li)
        
        belonging = [-1] * len(text)
        pos = 0
        for tid, tok in enumerate(tokens):
            apr = text[pos:].find(tok) + pos
            assert apr != -1
            for j in range(pos, apr+len(tok)):
                belonging[j] = tid
            pos = apr + len(tok)
        
        entity_intv = []
        for sid, sent in enumerate(sentences):
            tmp = []
            pos = text.find(sent)
            for ent in entity[sid]:
                apr = text[pos:].find(ent) + pos
                el = belonging[apr]
                er = belonging[apr + len(ent) - 1]
                tmp.append((el, er))
                pos = apr + len(ent)
            entity_intv.append(tmp)

        entity_prob = []
        for ent_itv_per_sent in entity_intv:
            tmp = []
            for itv in ent_itv_per_sent:
                probs = np.array(logprobs[itv[0]:itv[1]+1])
                p = {
                    "avg": np.mean,
                    "max": np.max,
                    "min": np.min,
                    "first": lambda x: x[0] if len(x) > 0 else 0
                }.get(self.entity_solver, lambda x: 0)(probs)
                tmp.append(p)
            entity_prob.append(tmp)

        for sid in range(len(sentences)):
            if len(entity_prob[sid]) == 0:
                continue
            probs = [1 - exp(v) for v in entity_prob[sid]]
            probs = np.array(probs)
            p = {
                "avg": np.mean,
                "max": np.max,
                "min": np.min,
            }.get(self.sentence_solver, lambda x: 0)(probs)
            if p > self.hallucination_threshold: # hallucination
                # keep sentences before hallucination 
                prev = "" if sid == 0 else " ".join(sentences[:sid])
                # replace all hallucinated entities in current sentence with [xxx]
                curr = sentences[sid]
                pos = 0
                for prob, ent in zip(probs, entity[sid]):
                    apr = curr[pos:].find(ent) + pos
                    if prob > self.hallucination_threshold:
                        curr = curr[:apr] + "[xxx]" + curr[apr+len(ent):]
                        pos = apr + len("[xxx]")
                    else:
                        pos = apr + len(ent)
                return prev, curr, True
        # No hallucination
        return text, None, False

    def inference(self, question, demo, case):
        return super().inference(question, demo, case)


class AttnWeightRAG(BasicRAG):
    def __init__(self, args):
        super().__init__(args)
    
    def modifier(self, text, tokens, attentions, weight):
        sentences = [sent.text.strip() for sent in nlp(text).sents]
        sentences = [sent for sent in sentences if len(sent) > 0]
        tid = 0
        for sid, sent in enumerate(sentences):
            tl, tr = tid, tid
            if sid == len(sentences) - 1:
                tl, tr = tid, len(tokens)
            else:
                for i in range(tid + 1, len(tokens)):
                    seq = " ".join(tokens[tl:i])
                    if sent in seq:
                        tr = i
                        break
                tid = tr
            # value = attenion * (-log prob)
            attns = attentions[tl:tr]
            if len(attns) == 0: continue
            attns = np.array(attns) / (sum(attns) + 1e-9)
            value = [attns[i-tl] * weight[i] * (tr-tl) for i in range(tl, tr)] 
            thres = [1 if v > self.hallucination_threshold else 0 for v in value]
            if 1 in thres:
                # hallucinated
                if "check_real_words" in self.__dict__ and self.check_real_words:
                    doc = nlp(sent)
                    real_words = set(token.text for token in doc if token.pos_ in 
                        ['NOUN', 'ADJ', 'VERB', 'PROPN', 'NUM'])
                    def match(tok):
                        for word in real_words:
                            if word in tok:
                                return True
                        return False
                    for i in range(len(thres)):
                        if not match(tokens[tl+i]):
                            thres[i] = 0                
                
                prev = "" if sid == 0 else " ".join(sentences[:sid])
                return True, prev, tokens[tl:tr], thres
        return False, text, None, None

    def keep_real_words(self, prev_text, curr_tokens, curr_hit):
        curr_text = " ".join(curr_tokens)
        all_text = prev_text + " " + curr_text
        
        # Truncation
        inputs = self.generator.tokenizer(
            all_text, 
            return_tensors="pt", 
            truncation=True, 
            max_length=self.generator.max_pos-50,
            padding=True
        )
        
        # [수정] 모델 디바이스로 이동
        input_ids = inputs['input_ids'].to(self.generator.model.device)
        input_length = input_ids.shape[1]
        
        # [수정] 안전장치 적용
        input_ids = self.generator._sanitize_inputs(input_ids)

        tokens_tmp = self.generator.tokenizer.convert_ids_to_tokens(input_ids[0].tolist())

        with torch.no_grad():
             out = self.generator.model(input_ids, output_attentions=True)
             atten_tmp = out.attentions[-1][0]

        # merge tokens
        range_ = []
        for i, t in enumerate(tokens_tmp):
            if i == 0 or t.startswith(self.generator.space_token) or input_ids[0][i] == 13:
                range_.append([i, i])
            else:
                range_[-1][-1] += 1
        tokens = []
        for r in range_:
            tokenseq = "".join(tokens_tmp[r[0]: r[1]+1]).replace(self.generator.space_token, "")
            tokens.append(tokenseq)

        # 获取幻觉词对应的 attention
        curr_st = len(tokens) - len(curr_tokens)
        if curr_st < 0: curr_st = 0

        atten_tmp = torch.mean(atten_tmp, dim=0)
        attns = []
        for r in range_:
            att = torch.zeros(input_length).to(atten_tmp.device)
            for i in range(r[0], r[1] + 1):
                if i == 0:
                    continue
                v = atten_tmp[i-1][:r[0]]
                v = v / (v.sum() + 1e-9)
                t = torch.zeros(input_length).to(atten_tmp.device)
                t[:r[0]] = v
                att += t
            att /= (r[1] - r[0] + 1)
            # merge token for att
            att = torch.tensor([att[rr[0]:rr[1]+1].sum() for rr in range_])
            attns.append(att)
            
        # 计算每个超过阈值的 token 在前文的 attentions
        forward_attns = torch.zeros(len(tokens))
        hit_cnt = 0
        for i in range(len(curr_hit)):
            if i + curr_st >= len(attns): break
            if curr_hit[i] == 1:
                forward_attns += attns[curr_st + i]
                hit_cnt += 1
        
        if hit_cnt > 0:
            forward_attns /= hit_cnt
        forward_attns = forward_attns.tolist()

        # 分析词性
        doc = nlp(all_text)
        real_words = set(token.text for token in doc if token.pos_ in 
                      ['NOUN', 'ADJ', 'VERB', 'PROPN', 'NUM'])
        
        def match(token):
            for word in real_words:
                if word in token:
                    return True
            return False
        
        real_pairs = []
        for i in range(len(tokens)):
            tok, att = tokens[i], forward_attns[i]
            if i >= curr_st and i-curr_st < len(curr_hit) and curr_hit[i - curr_st]:
                continue
            if match(tok):
                real_pairs.append((att, tok, i))
        
        if "retrieve_keep_top_k" in self.__dict__:
            top_k = min(self.retrieve_keep_top_k, len(real_pairs))
        elif "retrieve_keep_ratio" in self.__dict__:
            top_k = int(len(real_pairs) * self.retrieve_keep_ratio)
        else:
            top_k = len(real_pairs)
        
        real_pairs = sorted(real_pairs, key = lambda x:x[0], reverse=True)
        real_pairs = real_pairs[:top_k]
        real_pairs = sorted(real_pairs, key = lambda x:x[2])
        return " ".join([x[1] for x in real_pairs])
        
    def inference(self, question, demo, case):
        text = ""
        while True:
            old_len = len(text)
            prompt = "".join([d["case"]+"\n" for d in demo])
            tmp_li = [case, text]
            prompt += " ".join(s for s in tmp_li if len(s) > 0)

            new_text, tokens, attns, logprobs, entropies = self.generator.generate_attn(
                prompt, 
                self.generate_max_length, 
                use_entropy = self.method == "dragin", 
                use_logprob = self.method == "attn_prob"
            )
            weight = entropies if self.method == "dragin" else [-v for v in logprobs]

            if self.use_counter == True:
                self.counter.add_generate(new_text, self.generator.tokenizer)
            hallucination, ptext, curr_tokens, curr_hit =  self.modifier(new_text, tokens, attns, weight)
            
            if not hallucination:
                text = text.strip() + " " + new_text.strip()
            else:
                forward_all = [question, text, ptext]
                forward_all = " ".join(s for s in forward_all if len(s) > 0)

                if self.query_formulation == "current":
                    retrieve_question = " ".join(curr_tokens)

                elif self.query_formulation == "current_wo_wrong":
                    retrieve_question = " ".join(
                        list(curr_tokens[i] if curr_hit[i] == 0 else "" for i in range(len(curr_tokens)))
                    )

                elif self.query_formulation == "forward_all":
                    retrieve_question = forward_all
                
                elif self.query_formulation == "last_sentence":
                    retrieve_question = self.get_last_sentence(forward_all)
                
                elif self.query_formulation == "last_n_tokens":
                    assert "retrieve_keep_top_k" in self.__dict__
                    
                    def fetch_last_n_tokens(text, num, tokenizer = self.generator.tokenizer):
                        tokens = tokenizer.tokenize(text)
                        if num >= len(tokens):
                            return text
                        last_n_tokens = tokens[-num:]
                        return ' '.join(last_n_tokens)

                    retrieve_question = fetch_last_n_tokens(
                        forward_all, self.retrieve_keep_top_k)
                
                elif self.query_formulation == "real_words": 
                    retrieve_question = self.keep_real_words(
                        prev_text = question + " " + text + " " + ptext, 
                        curr_tokens = curr_tokens, 
                        curr_hit = curr_hit,
                    ) 
                else:
                    raise NotImplemented

                docs = self.retrieve(retrieve_question, topk=self.retrieve_topk)
                prompt = "".join([d["case"]+"\n" for d in demo])
                prompt += "Context:\n"
                for i, doc in enumerate(docs):
                    prompt += f"[{i+1}] {doc}\n"
                prompt += "Answer in the same format as before.\n"
                tmp_li = [case, text, ptext.strip()]
                prompt += " ".join(s for s in tmp_li if len(s) > 0)
                
                new_text, _, _ = self.generator.generate(prompt, self.generate_max_length)
                if self.use_counter == True:
                    self.counter.add_generate(new_text, self.generator.tokenizer)
                    self.counter.hallucinated += 1
                new_text = self.get_top_sentence(new_text)
                tmp_li = [text.strip(), ptext.strip(), new_text.strip()]
                text = " ".join(s for s in tmp_li if len(s) > 0)
            
            tokens_count = len(self.generator.tokenizer.encode(text))
            if tokens_count > self.generate_max_length or len(text) <= old_len or "the answer is" in text:
                break
        return text


# 1. Projector (sae_lens 활용)
class OptimizedSAEProjector(nn.Module):
    def __init__(self, layer_index, selected_feature_indices, device):
        super().__init__()
        self.device = device
        
        release = "llama_scope_lxr_8x"
        sae_id = f"l{layer_index}r_8x"
        logger.info(f"Loading SAE via sae_lens: {sae_id}")
        
        try:
            try:
                sae = SAE.from_pretrained(release=release, sae_id=sae_id, device=str(device))[0]
            except:
                sae = SAE.from_pretrained(release=release, sae_id=sae_id, device=str(device))
            
            # [최적화] 데이터 타입을 모델과 맞추기 (bfloat16 권장)
            full_W = sae.W_enc.data.to(dtype=torch.bfloat16) 
            full_b = sae.b_enc.data.to(dtype=torch.bfloat16)
            
            selected_indices_tensor = torch.tensor(selected_feature_indices, dtype=torch.long).to(full_W.device)
            optimized_W = full_W[:, selected_indices_tensor] 
            optimized_b = full_b[selected_indices_tensor]

            self.proj = nn.Linear(optimized_W.shape[0], optimized_W.shape[1], bias=True)
            with torch.no_grad():
                self.proj.weight.copy_(optimized_W.t())
                self.proj.bias.copy_(optimized_b)
            
            self.proj.to(device, dtype=torch.bfloat16) # Projector도 bf16으로 변환
            self.proj.eval()
            del sae
            torch.cuda.empty_cache()
            
        except Exception as e:
            logger.error(f"Failed to load SAE: {e}")
            raise e

    def forward(self, hidden_states):
        # hidden_states가 들어오면 바로 투영
        return F.relu(self.proj(hidden_states.to(self.proj.weight.dtype)))

# 2. [신규] 실시간 개입 핸들러 (StoppingCriteria)
class SAEInterventionHandler(StoppingCriteria):
    def __init__(self, sae_projector, lr_model, threshold):
        self.sae_projector = sae_projector
        self.lr_model = lr_model
        self.threshold = threshold
        self.triggered = False
        self.trigger_prob = 0.0

    def reset(self):
        self.triggered = False
        self.trigger_prob = 0.0

    # Hook 함수: 토큰이 생성될 때마다 실행됨
    def hook_fn(self, module, input, output):
        if self.triggered: return # 이미 트리거되었으면 연산 생략

        # output: (batch, seq, dim)
        if isinstance(output, tuple): hidden = output[0]
        else: hidden = output
            
        # 마지막 토큰만 검사
        last_token_hidden = hidden[:, -1:, :]
        
        with torch.no_grad():
            # SAE 계산
            sae_acts = self.sae_projector(last_token_hidden)
            # Pooling & CPU 이동
            feats_pooled = sae_acts.squeeze(1).float().detach().cpu().numpy() # LR은 float32 필요할 수 있음
            
            try:
                probs = self.lr_model.predict_proba(feats_pooled)
                prob_retrieve = probs[0][1]
                
                if prob_retrieve > self.threshold:
                    self.triggered = True
                    self.trigger_prob = prob_retrieve
            except:
                pass

    def __call__(self, input_ids, scores, **kwargs):
        return self.triggered

# 3. [최적화 적용] SAETriggerRAG
class SAETriggerRAG(BasicRAG):
    def __init__(self, args):
        # [중요 최적화] BasicGenerator 초기화 전 dtype 변경을 위해 args 수정 권장
        # 하지만 여기선 generator 생성 후 모델을 casting 하는 방식 사용
        super().__init__(args)

        # [최적화 1] 모델을 bfloat16으로 변환 (A6000 최적화)
        logger.info("Converting model to bfloat16 for speedup...")
        self.generator.model = self.generator.model.to(torch.bfloat16)
        
        # 1. Feature CSV 로드
        logger.info(f"Loading SAE features from {self.feature_csv_path}")
        df = pd.read_csv(self.feature_csv_path)
        
        if 'feature_idx' in df.columns: self.target_indices = df['feature_idx'].tolist()
        elif 'feature_index' in df.columns: self.target_indices = df['feature_index'].tolist()
        elif 'feature_name' in df.columns: self.target_indices = [int(str(x).split('_')[-1]) for x in df['feature_name']]
        else: raise ValueError("CSV error.")
            
        # 2. Projector 초기화
        self.sae_projector = OptimizedSAEProjector(
            layer_index=self.sae_layer_index, 
            selected_feature_indices=self.target_indices, 
            device=self.generator.model.device
        )

        # 3. LR 모델 로드
        logger.info(f"Loading LR model from {self.lr_model_path}")
        self.lr_model = joblib.load(self.lr_model_path)
        self.threshold = getattr(self, 'search_threshold', 0.5)
        
        # 4. 핸들러 생성
        self.handler = SAEInterventionHandler(self.sae_projector, self.lr_model, self.threshold)

    def inference(self, question, demo, case):
        text = ""
        current_context = ""
        retrieve_count = 0
        max_retries = 3 
        
        base_prompt = "".join([d["case"]+"\n" for d in demo]) + case 
        
        # Hook 등록
        target_layer = self.generator.model.model.layers[self.sae_layer_index]
        hook_handle = target_layer.register_forward_hook(self.handler.hook_fn)
        
        try:
            while True:
                prompt = base_prompt + "\n"
                if current_context: prompt += f"Context: {current_context}\n"
                prompt += "Answer: " + text
                
                self.handler.reset()
                
                inputs = self.generator.tokenizer(prompt, return_tensors="pt").to(self.generator.model.device)
                input_len = inputs.input_ids.shape[1]
                
                # [최적화 2] generate_with_hidden 대신 hook + stopping_criteria 사용
                # 두 번 연산하지 않음!
                outputs = self.generator.model.generate(
                    **inputs,
                    max_new_tokens=self.generate_max_length,
                    pad_token_id=self.generator.tokenizer.pad_token_id,
                    do_sample=False,
                    stopping_criteria=StoppingCriteriaList([self.handler])
                )
                
                generated_ids = outputs[0][input_len:]
                new_text = self.generator.tokenizer.decode(generated_ids, skip_special_tokens=True)
                
                if self.use_counter:
                    self.counter.add_generate(new_text, self.generator.tokenizer)

                # 트리거 발생 시 (검색 필요)
                if self.handler.triggered and retrieve_count < max_retries:
                    first_sentence = self.get_top_sentence(new_text)
                    if not first_sentence: first_sentence = new_text
                    
                    logger.info(f"[HOOK TRIGGER] Prob: {self.handler.trigger_prob:.4f} | Text: {first_sentence[:30]}...")
                    
                    # 검색 수행
                    search_query = f"{question} {first_sentence}"
                    docs = self.retrieve(search_query, topk=self.retrieve_topk)
                    current_context += "\n".join([f"[{i+1}] {doc}" for i, doc in enumerate(docs)])
                    retrieve_count += 1
                    continue # 다시 생성

                # 정상 종료 시
                else:
                    first_sentence = self.get_top_sentence(new_text)
                    if not first_sentence: break
                    text = (text + " " + first_sentence).strip()
                    
                    if len(self.generator.tokenizer.encode(text)) > 200 or "the answer is" in first_sentence.lower(): break
                    if not self.handler.triggered and len(new_text) < len(first_sentence) + 5: break
                        
        finally:
            hook_handle.remove() # Hook 제거
        
        return text