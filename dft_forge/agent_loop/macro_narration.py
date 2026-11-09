"""Read-only scientific narration for deterministically routed macros.

Loss-sensitive intents (charge-state lists, adsorption-site lists) are routed to
their macro *before* the free-form LLM planner, so a model cannot silently drop a
requested site or collapse a charged defect into the neutral workflow. That
routing is correct, but it left the product's two most scientifically involved
tasks with a bare one-line tool description while every model-planned task got
several paragraphs of discussion.

This module restores the explanation without reopening the decision: the tool id
and its arguments are already fixed, and only narrative fields are requested.
Anything the model returns that is not a narrative field is discarded, so a
failed, slow or adversarial reply degrades to exactly the previous bare plan.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Narrative fields only. `actions`, `tool`, `args`, `execution_mode` and
# `support_status` are deliberately absent: the plan is already decided.
_NARRATIVE_KEYS = ("reply", "goal", "assumptions", "expected_outputs")

_SYSTEM = """你是 DFT-Forge 的科研方案说明员。计算方案已由确定性规则选定，不可更改。
你的唯一任务是解释这个既定方案的科学内容。

铁律：
- 不得建议改用其他工具、增删步骤，或改动任何参数值。
- 只解释 tool 与 args 中已经写明的内容；不要发明用户没有给出的数值。
- 明确指出该方案的科学边界（例如：吸附位点最低仅代表本次请求且验证通过的初始构型集合，
  不代表全局最稳定构型；带电缺陷只能引用显式提供的介电常数、势对齐与有限尺寸修正，
  不能称为尺寸收敛或带隙修正后的结果）。
- 不要声称计算已经完成或已经收敛。

只返回一个 JSON 对象，不要 markdown 代码围栏：
{"reply": "2-5 句中文科学说明",
 "goal": "一句话目标",
 "assumptions": ["该方案隐含的科学假设"],
 "expected_outputs": ["预期输出项"]}"""


def _clean_str_list(value: Any, *, limit: int, item_max: int) -> List[str]:
    if not isinstance(value, list):
        return []
    out: List[str] = []
    for item in value[:limit]:
        if isinstance(item, (str, int, float)) and not isinstance(item, bool):
            text = str(item).strip()
            if text:
                out.append(text[:item_max])
    return out


def narrate_macro_plan(
    provider: Any,
    message: str,
    steps: List[Dict[str, Any]],
    *,
    timeout_note: str = "",
) -> Optional[Dict[str, Any]]:
    """Return narrative-only fields for an already-decided macro plan.

    Returns None whenever narration is unavailable or unusable, in which case
    the caller keeps the deterministic plan exactly as it was.
    """
    if provider is None or not steps:
        return None
    try:
        payload = {
            "user_request": message[:2000],
            "decided_plan": [
                {
                    "tool": step.get("tool"),
                    "args": step.get("args") or {},
                    "description": str(step.get("description") or "")[:400],
                }
                for step in steps[:8]
            ],
        }
        raw = provider.chat(
            _SYSTEM + (f"\n{timeout_note}" if timeout_note else ""),
            json.dumps(payload, ensure_ascii=False, default=str),
        )
        text = str(raw).strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        data = json.loads(text)
        if not isinstance(data, dict):
            return None
    except Exception as exc:  # noqa: BLE001 - narration is best-effort by design
        logger.info("macro narration unavailable (%s); keeping bare plan", type(exc).__name__)
        return None

    narrative: Dict[str, Any] = {}
    reply = data.get("reply")
    if isinstance(reply, str) and reply.strip():
        narrative["reply"] = reply.strip()[:4000]
    goal = data.get("goal")
    if isinstance(goal, str) and goal.strip():
        narrative["goal"] = goal.strip()[:240]
    assumptions = _clean_str_list(data.get("assumptions"), limit=8, item_max=300)
    if assumptions:
        narrative["assumptions"] = assumptions
    expected = _clean_str_list(data.get("expected_outputs"), limit=10, item_max=200)
    if expected:
        narrative["expected_outputs"] = expected
    # A narration with no usable prose is not worth overwriting anything for.
    return narrative if narrative.get("reply") else None


def apply_macro_narration(plan: Any, narrative: Optional[Dict[str, Any]]) -> bool:
    """Copy narrative-only fields onto a PlanIR. Never touches actions/args."""
    if not narrative:
        return False
    for key in _NARRATIVE_KEYS:
        if key not in narrative:
            continue
        # Only fill what deterministic routing left empty, so a rule-supplied
        # goal or expected output is never overwritten by model prose.
        if key == "goal" and getattr(plan, "goal", ""):
            continue
        if key in ("assumptions", "expected_outputs") and getattr(plan, key, None):
            continue
        setattr(plan, key, narrative[key])
    return True
