"""基于 LLM 的 Reranker 实现。

本模块使用 Large Language Models 评估候选段落与给定查询的相关性，从而完成重排。
它从 config/prompts/rerank.txt 读取提示词，并将 LLM 输出结构化以供下游处理。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.core.settings import resolve_path
from src.libs.llm.base_llm import BaseLLM, Message
from src.libs.llm.llm_factory import LLMFactory
from src.libs.reranker.base_reranker import BaseReranker


class LLMRerankError(RuntimeError):
    """LLM 重排失败时抛出。"""


class LLMReranker(BaseReranker):
    """基于 LLM 和结构化提示词的 Reranker。

    此实现借助 LLM 对候选段落与查询的相关性进行打分和重排。
    它从可配置的文件中读取重排提示词，并期望 LLM 输出结构化的 JSON。

    应用的设计原则：
    - 可插拔：可通过工厂替换为其他 Reranker 实现。
    - 配置驱动：提示词文件路径与 LLM 配置来自配置。
    - 可观测：支持 TraceContext 以进行监控（阶段 F 集成）。
    - 可降级：失败时返回信号，供上游降级逻辑处理。
    - 结构化输出：按预期 schema 校验 LLM 输出。
    """
    
    def __init__(
        self,
        settings: Any,
        prompt_path: Optional[str] = None,
        llm: Optional[BaseLLM] = None,
        **kwargs: Any
    ) -> None:
        """初始化 LLM Reranker。

        参数:
            settings: 包含 LLM 与 rerank 配置的应用设置。
            prompt_path: 可选的重排提示词文件路径。若为 None，则使用默认的
                'config/prompts/rerank.txt'。用于测试时注入自定义提示词。
            llm: 可选的 LLM 实例。若为 None，则通过 LLMFactory 从 settings 创建。
                用于测试时注入 mock LLM。
            **kwargs: 额外的 provider 特定参数。
        """
        self.settings = settings
        self.prompt_path = prompt_path or str(resolve_path("config/prompts/rerank.txt"))
        self.llm = llm or LLMFactory.create(settings)
        self.kwargs = kwargs
        
        # 加载提示词模板
        try:
            self.prompt_template = self._load_prompt_template(self.prompt_path)
        except Exception as e:
            raise LLMRerankError(f"Failed to load rerank prompt from {self.prompt_path}: {e}") from e
    
    def _load_prompt_template(self, path: str) -> str:
        """从文件加载重排提示词模板。

        参数:
            path: 提示词模板文件路径。

        返回:
            以字符串形式返回的提示词模板。

        异常:
            FileNotFoundError: 当提示词文件不存在时。
            IOError: 当文件无法读取时。
        """
        prompt_file = Path(path)
        if not prompt_file.exists():
            raise FileNotFoundError(f"Rerank prompt file not found: {path}")
        
        return prompt_file.read_text(encoding="utf-8")
    
    def _build_rerank_prompt(self, query: str, candidates: List[Dict[str, Any]]) -> str:
        """使用查询与候选项构建重排提示词。

        参数:
            query: 用户查询字符串。
            candidates: 待重排的候选记录列表。

        返回:
            可直接用于 LLM 的格式化提示词字符串。
        """
        # 为提示词格式化候选
        candidates_text = []
        for i, candidate in enumerate(candidates):
            passage_id = candidate.get("id", f"passage_{i}")
            text = candidate.get("text", candidate.get("content", ""))
            candidates_text.append(f"Passage ID: {passage_id}\nText: {text}\n")

        candidates_str = "\n".join(candidates_text)

        # 构建完整提示词
        full_prompt = f"{self.prompt_template}\n\nQuery: {query}\n\nPassages:\n{candidates_str}\n\nOutput your response as a JSON array of objects, one per passage."
        
        return full_prompt
    
    def _parse_llm_response(self, response_text: str) -> List[Dict[str, Any]]:
        """解析并校验 LLM 响应。

        参数:
            response_text: LLM 返回的原始文本响应。

        返回:
            解析后的排序记录列表，包含 passage_id 与 score。

        异常:
            LLMRerankError: 当响应不符合预期 schema 时。
        """
        # 尝试从响应中提取 JSON（LLM 可能用 markdown 包裹）
        text = response_text.strip()

        # 若存在 markdown 代码块则移除
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
        
        # 解析 JSON
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as e:
            raise LLMRerankError(
                f"LLM response is not valid JSON: {e}\nResponse: {response_text[:200]}"
            ) from e
        
        # 校验结构
        if not isinstance(parsed, list):
            raise LLMRerankError(
                f"Expected JSON array, got {type(parsed).__name__}\nResponse: {response_text[:200]}"
            )
        
        # 校验每一项
        for i, item in enumerate(parsed):
            if not isinstance(item, dict):
                raise LLMRerankError(
                    f"Item {i} is not a dict (type: {type(item).__name__})"
                )
            if "passage_id" not in item:
                raise LLMRerankError(f"Item {i} missing required field 'passage_id'")
            if "score" not in item:
                raise LLMRerankError(f"Item {i} missing required field 'score'")
            
            # 校验 score 为数值
            score = item["score"]
            if not isinstance(score, (int, float)):
                raise LLMRerankError(
                    f"Item {i} score must be numeric, got {type(score).__name__}: {score}"
                )
        
        return parsed
    
    def _map_results_to_candidates(
        self,
        parsed_results: List[Dict[str, Any]],
        candidates: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """将 LLM 排序结果映射回原始候选对象。

        参数:
            parsed_results: 解析后的 LLM 输出，包含 passage_id 与 score。
            candidates: 原始候选列表。

        返回:
            按分数降序排序后的重排候选对象列表。
        """
        # 创建 id 到候选的映射
        id_to_candidate = {}
        for i, candidate in enumerate(candidates):
            candidate_id = candidate.get("id", f"passage_{i}")
            id_to_candidate[candidate_id] = candidate

        # 构建重排后的列表
        reranked = []
        for result in parsed_results:
            passage_id = result["passage_id"]
            score = result["score"]

            if passage_id in id_to_candidate:
                candidate = id_to_candidate[passage_id].copy()
                # 用 LLM 重排分数更新 score
                candidate["rerank_score"] = float(score)
                reranked.append(candidate)

        # 按重排分数降序排序
        reranked.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)
        
        return reranked
    
    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """使用基于 LLM 的相关性打分对候选进行重排。

        参数:
            query: 用户查询字符串。
            candidates: 待重排的候选记录列表。
            trace: 可选的 TraceContext，用于可观测性。
            **kwargs: 额外参数（timeout、temperature 等）。

        返回:
            按 LLM 给出的相关性分数排序后的重排候选列表。
            成功时，每个候选的元数据中都会包含 'rerank_score'。

        异常:
            ValueError: 当 query 或 candidates 无效时。
            LLMRerankError: 当 LLM 调用失败或响应格式错误时。
        """
        # 校验输入
        self.validate_query(query)
        self.validate_candidates(candidates)

        # 若只有一个候选，无需重排
        if len(candidates) == 1:
            return candidates

        # 构建提示词
        try:
            prompt = self._build_rerank_prompt(query, candidates)
        except Exception as e:
            raise LLMRerankError(f"Failed to build rerank prompt: {e}") from e

        # 调用 LLM
        try:
            messages = [Message(role="user", content=prompt)]
            response = self.llm.chat(messages, trace=trace, **kwargs)
            response_text = response.content
        except Exception as e:
            # 返回降级信号 - 由上游决定如何处理
            raise LLMRerankError(f"LLM call failed during reranking: {e}") from e

        # 解析响应
        try:
            parsed_results = self._parse_llm_response(response_text)
        except LLMRerankError:
            raise
        except Exception as e:
            raise LLMRerankError(f"Failed to parse LLM rerank response: {e}") from e

        # 映射回候选并排序
        try:
            reranked = self._map_results_to_candidates(parsed_results, candidates)
        except Exception as e:
            raise LLMRerankError(f"Failed to map LLM results to candidates: {e}") from e
        
        return reranked
