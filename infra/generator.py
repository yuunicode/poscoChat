# infra/generator.py
from abc import ABC, abstractmethod
from typing import List
from fastembed import TextEmbedding # TODO: 의존성 해결
import requests, logging
from scipy.spatial.distance import cosine

class BaseLLMGenerator(ABC):
    def __init__(self, model_name: str):
        self.model_name = model_name
        self._default_embedder = TextEmbedding("nomic-ai/nomic-embed-text-v1.5-Q")

    @abstractmethod
    def generate_answers(self, results, user_query: str, prompt_template: str) -> List[dict]: pass

    def select_prompt(self, prompt_type:str) -> str:

        if prompt_type == "configuration":
            prompt = """
            You are an AI assistant for ECMiner software.
            The user wants to know how to configure a specific node or setting.
            The question is about how to set up or adjust specific options in a node or function.  

            Do NOT use prior knowledge or invent answers.
            If information is not available, say: \"Information not found in context.\"
            
            Question: {query}
            Reference context: {context}
            Answer:
            """

        elif prompt_type == "workflow":
            prompt = """
            You are an AI assistant for ECMiner software.
            The user is asking how to perform a specific procedure using ECMINER nodes.   
            The question is about how to perform a sequence of steps to accomplish a task. These are detailed process questions about how to do something in ECMiner.
            Do NOT use prior knowledge or invent answers.
            If information is not available, say: \"Information not found in context.\"
            
            Question: {query}
            Reference context: {context}
            Answer:
            """
        elif prompt_type == "availability":
            prompt = """
            You are an AI assistant for ECMiner software.
            The user wants to know whether a specific feature or function is supported.  
            The question is about whether a certain feature or capability exists. Also if the questioned function is available in ECMiner software. Questions about the possibility of questioned content.
            Do NOT use prior knowledge or invent answers.
            If information is not available, say: \"Information not found in context.\"

            Question: {query}
            Reference context: {context}
            Answer:
            """
        elif prompt_type == "function":
            prompt = """
            You are an AI assistant for ECMiner software.
            The user wants to know what a specific node does.  
            The question is specifically asking about the purpose or function of a feature in the ECMiner software.
            Do NOT use prior knowledge or invent answers.
            If information is not available, say: \"Information not found in context.\"
            
            Question: {query}
            Reference context: {context}
            Answer:
            """
        elif prompt_type == "definition":
            prompt = """
            You are an AI assistant for ECMiner software.
            Define the term in the user's question using only the context. 
            The question asks about what a technical term or concept means. General theory questions.
            Do NOT use prior knowledge or invent answers.
            If information is not available, say: \"Information not found in context.\"
                    
            Question: {query}
            Reference context: {context}
            Answer:
            """
        elif prompt_type == "error":
            prompt = """
            You are an AI assistant for ECMiner software.
            The user is experiencing an issue and wants to know what’s wrong or how to fix it.
            Analyze the situation using only the context. 
            The question is about how to fix or understand an error or unexpected result. Questions about solving problems that occurred in ECMiner software. Errors that appeared in the software that needs instructions to be fixed. 
            Do NOT use prior knowledge or invent answers.
            If information is not available, say: \"Information not found in context.\"
            
            Question: {query}
            Reference context: {context}
            Answer:
            """

        elif prompt_type == "comparison":
            prompt = """
            You are an AI assistant for ECMiner software.
            Compare two or more nodes as requested by the user. 
            Use only information from the context to list differences or similarities
            The question is about the differences or similarities between two or more nodes. Questions that compares between features in ECMiner software.
            Do NOT use prior knowledge or invent answers.
            If information is not available, say: \"Information not found in context.\"
            
            Question: {query}
            Reference context: {context}
            Answer:
            """
        
        elif prompt_type == "usecase":
            prompt = """
            You are an AI assistant for ECMiner software.
            The user is asking which nodes or process should be used to achieve a specific goal.  
            Only mention use cases, nodes, or flows that appear in the context. 
            The question is asking how ECMiner can be applied to solve a real-world business or analytical problem (e.g. customer segmentation, fraud detection, forecasting). These questions focus on **application scenarios**, not step-by-step instructions. They may request examples or typical usage flows for a goal.
            Do NOT use prior knowledge or invent answers.
            If information is not available, say: \"Information not found in context.\"
            
            Question: {query}
            Reference context: {context}
            Answer:
            """
        elif prompt_type == "interpretation":
            prompt = """
            You are an AI assistant for ECMiner software.
            Explain the meaning of the output based strictly on the given context.
            The question is about the understanding the meaning of outputs or analytical results. Questions about analyzing results. 
            Do NOT use prior knowledge or invent answers.
            If information is not available, say: \"Information not found in context.\"
            
            Question: {query}
            Reference context: {context}
            Answer:
            """
        elif prompt_type == "support":
            prompt = """
            You are an AI assistant for ECMiner software.
            Answer the user's question about licensing, supported versions, or installation using only the context.  
            The question is about installation, compatibility, licensing, or support. Asking about the use of the software itself, whether than the usage of the features or functions.
            Do NOT use prior knowledge or invent answers.
            If information is not available, say: \"Information not found in context.\"
            
            Question: {query}
            Reference context: {context}
            Answer:
            """
        else:
            prompt = """
            Question: {query}
            Reference context: {context}
            Answer:
            """

        return prompt

    def filter_references_by_similarity(self, answer: str, results, threshold: float = 0.6) -> List[str]:
        """
        answer와 results의 각 content 간 유사도를 비교해
        threshold 이상인 section_path만 반환
        """
        try:
            answer_vec = list(self._default_embedder.embed([answer]))[0]

            filtered_paths = []
            for sp in results:
                content = sp.payload.get('content', '')
                section_path = (
                    sp.payload.get('metadata', {})
                    .get('general_info', {})
                    .get('section_path')
                )
                if not content or not section_path:
                    continue

                content_vec = list(self._default_embedder.embed([content]))[0]
                similarity = 1 - cosine(answer_vec, content_vec)

                # 유사도 기준 or 텍스트 포함
                if similarity >= threshold or content[:40] in answer:
                    filtered_paths.append(section_path)

            return list(dict.fromkeys(filtered_paths))

        except Exception as e:
            logging.warning(f"[LLMGenerator] 유사도 기반 reference 필터링 실패: {e}")
            return []


class OllamaGenerator(BaseLLMGenerator):
    def __init__(self, model_name: str):

        super().__init__(model_name)
        self.api_url = "http://127.0.0.1:11434/api/chat"

    def generate_answers(self, results, user_query: str, prompt_template: str):
        context_texts = []
        section_paths = []

        for sp in results[:5]:
            content = sp.payload.get('content', '')
            context_texts.append(content)
            section_path = (
                sp.payload.get('metadata', {})
                .get('general_info', {})
                .get('section_path')
            )
            if section_path:
                section_paths.append(section_path)

        joined_context = "\n\n".join(context_texts)

        prompt = prompt_template.format(context=joined_context, query=user_query)
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
            "stream": False
        }

        resp = requests.post(self.api_url, json=payload)
        resp.raise_for_status()
        data = resp.json()
        
        answer = data.get('message', {}).get('content', '').strip() or data.get('response', '').strip()
        section_paths = self.filter_references_by_similarity(answer, results)
        section_info = ""
        if section_paths:
            section_info = "\n\nReference:\n\n" + "\n".join(
                f"{i+1}. {p}" for i, p in enumerate(section_paths)
            ) + "\n"

        answer_with_src = answer + section_info
        return [{"prompt": prompt, "answer": answer_with_src}]


class OpenRouterGenerator(BaseLLMGenerator):
    def __init__(self, model_name: str):
        from openai import OpenAI
        super().__init__(model_name)
        self.client = OpenAI(base_url   = "https://openrouter.ai/api/v1", 
                             api_key    = "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")

    def generate_answers(self, results, user_query: str, prompt_template: str):
        context_texts = []
        section_paths = []
        for sp in results[:5]:
            content = sp.payload.get('content', '')
            context_texts.append(content)
            section_path = (
                sp.payload.get('metadata', {})
                .get('general_info', {})
                .get('section_path')
            )
            if section_path:
                section_paths.append(section_path)

        joined_context = "\n\n".join(context_texts)

        answers = []
        prompt = prompt_template.format(context=joined_context, query=user_query)
        response = self.client.chat.completions.create(
            model       = self.model_name,
            messages    = [{"role": "user", "content": prompt}],
            temperature = 0.5,
            top_p = 0.7,
        )
        answer = response.choices[0].message.content or "[No response received]"
        # 답변 마지막에 section_path(출처) 추가
        section_paths = self.filter_references_by_similarity(answer, results)
        section_info = ""
        if section_paths:
            section_info = "\n\nReference:\n\n" + "\n".join(
                f"{i+1}. {p}" for i, p in enumerate(section_paths)
            ) + "\n"

        answer_with_src = answer + section_info
        answers.append({"prompt": prompt, "answer": answer_with_src})

        return answers

if __name__ == "__main__":
    # 생성기 선택 예시
    generator = OllamaGenerator(model_name="qwen3:4b")  
    # or
    generator = OpenRouterGenerator(model_name="qwen/qwen3-8b:free")
    