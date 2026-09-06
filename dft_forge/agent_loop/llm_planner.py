"""LLM-based planner: natural language → tool-call steps.

Falls back (returns None) on any error so the rule planner can take over.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from dft_forge.agent_loop.helpers import _CHAT_SKIP_TOOLS
from dft_forge.agent_loop.planning import PlanIR, capability_catalog_text, compile_plan
from dft_forge.agent_loop.step_validator import validate_steps

logger = logging.getLogger(__name__)


def _is_confirmation(message: str) -> bool:
    """Whether a terse follow-up confirms the immediately prior offer."""
    import re

    return bool(re.fullmatch(r"\s*(?:需要|可以|好的|好|继续|是的|要|请|提取|拿出来)\s*[。！!,.，]?\s*", message or ""))


class LLMPlanner:
    """LLM-based planner: natural language → tool-call steps.

    Falls back (returns None) on any error so AgentLoop can use rules.
    """

    def __init__(self, registry_obj):
        self.registry = registry_obj
        self._provider = None
        self._checked = False
        self.last_plan_ir: Optional[PlanIR] = None

    def _get_provider(self):
        if self._checked:
            return self._provider
        self._checked = True
        import os

        if os.environ.get("DFT_FORGE_LLM_PROVIDER", "dummy").lower() not in ("openai", "openai_compatible", "openai-compatible"):
            return None
        vendor = os.environ.get("DFT_FORGE_LLM_VENDOR", "custom").lower()
        if not os.environ.get("DFT_FORGE_LLM_API_KEY") and vendor != "ollama":
            return None
        try:
            from dft_forge.llm import get_llm_provider

            provider = get_llm_provider()
            self._provider = provider if hasattr(provider, "chat") else None
        except Exception:
            self._provider = None
        return self._provider

    def _catalog(self) -> str:
        from dft_forge.compiler import MATERIAL_DB

        lines = [line for line in capability_catalog_text(
            self.registry, Path(__file__).resolve().parents[2] / "templates"
        ).splitlines() if not any(f"(tool={skip})" in line for skip in _CHAT_SKIP_TOOLS)]
        lines.append(f"Available bulk materials: {', '.join(sorted(MATERIAL_DB))}")
        return "\n".join(lines)

    def plan(self, message: str, ctx: Dict[str, Any], history: Optional[List[Dict[str, Any]]] = None):
        """Returns (steps, direct_reply) or (None, None) to fall back to rules."""
        provider = self._get_provider()
        if provider is None:
            return None, None
        self.last_plan_ir = None
        # Short confirmations are common after the agent has offered to
        # inspect a completed run. Route them deterministically so a model
        # cannot turn "需要" into another generic one-step calculation.
        if ctx.get("last_run_id") and _is_confirmation(message):
            self.last_plan_ir = PlanIR.from_steps(message, [{
                "tool": "graph.status",
                "args": {"run_id": ctx["last_run_id"]},
                "description": "读取上一次计算的详细输出并提取关键数值",
            }], source="follow-up")
            return ([{
                "tool": "graph.status",
                "args": {"run_id": ctx["last_run_id"]},
                "description": "读取上一次计算的详细输出并提取关键数值",
            }], None)
        system = f"""You are the planner of DFT-Forge, a DFT computation agent (Quantum ESPRESSO).
Convert the user's request into a JSON plan of tool calls.

Rules:
- Reply with ONLY one JSON object, no markdown fences:
  {{"reply": string, "goal": string, "constraints": {{}},
    "assumptions": [string], "expected_outputs": [string],
    "actions": [{{"capability": string, "args": {{}}, "description": string}}],
    "steps": [{{"tool": string, "args": {{}}, "description": string}}]}}
- First express the objective, constraints, assumptions and expected outputs.
  Then express actions using semantic capability ids from the catalog. The
  legacy steps field is accepted for compatibility, but actions are preferred.
- Use ONLY the tool ids listed below; args keys must match exactly.
- Known catalog materials and trusted prototype formulas can be passed as
  inputs.material of graph.run (e.g. CaTiO3, SrTiO3, GaAs, MoS2). A chemical
  formula does NOT uniquely determine a crystal: when a composition is not in
  the catalog, ask the user to upload CIF/POSCAR instead of inventing a phase.
  Bulk MoS2/WS2/MoSe2/WSe2 build the trusted 2H layered crystal.
- For a full verified calculation prefer graph.run with a template_id
  (t1_vc_relax = structure optimization, t2_bands = band structure, t2_dos = density of states).
  "算 X 的能带" → one graph.run t2_bands step; combining several targets emits one step each.
- 2D monolayers (单层/monolayer/二维/2D intent, or 石墨烯/graphene, 氮化硼/h-BN,
  MoS2/WS2/MoSe2/WSe2 单层): FIRST structure.build2d with kind 'graphene'|'bn'|
  'mos2'|'ws2'|'mose2'|'wse2' plus supercell/dopants/adsorb as requested,
  THEN graph.run with template_id and empty inputs — omit inputs.material so the
  built structure chains automatically. Kind matches the material named.
- Bulk doping ("P掺杂Si", "B-doped diamond", X掺杂Y): structure.dope
  with source (host material/formula), element (dopant), supercell '2x2x2'
  (default; use '3x3x3' for lower concentration), then graph.run with
  empty inputs — the doped structure chains automatically.
- 2D doping/adsorption (N掺杂石墨烯, O吸附在MoS2): structure.build2d with
  dopants/adsorb args (supercell '3x3' typical for doping), then graph.run
  with empty inputs.
- Site scan ("different sites/positions"): emit one structure.build2d step per site
  (top/bridge/hollow), each with its own output path.
- Adsorption energy (吸附能/E_ads): a 3-calculation workflow —
  (1) structure.build2d with adsorb + supercell '3x3', (2) graph.run t0_scf,
  (3) structure.build2d same supercell WITHOUT adsorb, (4) graph.run t0_scf,
  (5) structure.molecule (O→o2, H→h2, N→n2, Cl→cl2), (6) graph.run t0_scf,
  (7) thermo.eads with empty args (executor injects the three energies).
- Formation energy (形成能): (1) graph.run t0_scf with material=compound,
  then for EACH element: structure.reference {{element}} + graph.run t0_scf,
  finally thermo.formation {{formula}} with empty refs (auto-injected).
  Reference phases use structure.reference (NOT structure.generate), and the
  graph.run AFTER any builder (build2d/dope/molecule/reference) must have
  EMPTY inputs — the built structure chains automatically; never invent a
  material name for it.
- Crystallography knowledge: a primitive cell of rocksalt NaCl has only 2
  atoms (1 Na + 1 Cl) in a rhombohedral FCC setting — that is CORRECT; the
  conventional cubic cell has 4 formula units. Never call 2-atom NaCl wrong.
  structure.analyze reports space group + conventional cell for verification.
- Comparison requests (对比/比较/区别/差异 X和Y/哪个更): NEVER reply that you
  cannot find historical records — check 已算记录 in the context. If one of the
  systems is missing there, PLAN its calculation first (structure.build2d /
  structure.dope / graph.run with the right template), then add a final
  analysis.compare step with {{"use_last": N, "metric": ...}} where N =
  (systems already in 已算记录 that this comparison needs) + (systems this
  plan will compute). The executor injects the last N ledger entries.
  Never emit analysis.compare as the only step when a needed system is missing.
- Comparisons are only meaningful between LIKE cells: when computing the
  counterpart of an already-calculated system, match its supercell size and
  atom count (shown in 已算记录). Physical reason: bands fold when the cell
  grows — a 3x3 graphene supercell folds the Dirac point to Γ while 1x1 keeps
  it at K, so gaps from different-sized cells are not comparable. Also note
  gaps below ~0.27 eV are within smearing resolution and reported as
  effectively metallic.
- If the context contains 上次失败, the user asking to retry (再试/重跑/继续)
  refers to that failure — plan the same calculation again; do not ask what
  they mean.
- If the user confirms a previous offer to inspect/extract results (例如“需要”
  /“可以”/“继续”), and context has last_run_id, call graph.status with that
  run_id and report its numeric outputs; do not start a new graph.run.
- 能量差距/能量对比/哪个更稳定 → t0_scf total energies compared via
  analysis.compare metric "energy" (per-atom energies matter when atom counts
  differ). 能带隙/带隙对比 → t2_bands + metric "band_gap". These are DIFFERENT
  quantities — do not compute band structure when the user asks for energy.
- Self-doping is meaningless: C掺杂石墨烯 is just pure graphene (graphene IS
  carbon), Si掺杂Si is pure Si. If the user asks for this, do NOT plan a
  build for it — explain in the reply and offer a sensible alternative
  (compare doped vs pure, or a real dopant like B/N/P/S). The executor also
  rejects such builds.
- Chit-chat / questions about you: empty steps, answer in "reply".
- Use the conversation history for context: pronouns like "它/这个/再算一次/继续"
  refer to the material or calculation mentioned earlier.
- descriptions in Chinese. At most 8 steps.

Tools:
{self._catalog()}"""
        ctx_summary = {k: ctx.get(k) for k in ("last_structure_source", "last_formula", "last_input_file", "last_job_id", "last_run_id", "last_execution", "last_plan") if ctx.get(k)}
        last_failures = ctx.get("last_failures") or []
        if last_failures:
            ctx_summary["上次失败"] = "; ".join(str(f)[:160] for f in last_failures[-3:])
        ledger = ctx.get("ledger") or []
        if ledger:
            led_lines = []
            for i, e in enumerate(ledger[-8:], 1):
                bits = [f"[{i}] {e.get('label') or e.get('material') or '计算'}"]
                if e.get("template"):
                    bits.append(f"模板{e['template']}")
                if e.get("band_gap_ev") is not None:
                    bits.append(f"带隙{e['band_gap_ev']}eV")
                if e.get("energy_ry") is not None:
                    bits.append(f"E={e['energy_ry']}Ry")
                if e.get("natoms"):
                    bits.append(f"{e['natoms']}原子")
                led_lines.append(" ".join(bits))
            ctx_summary["已算记录"] = " | ".join(led_lines)
        user = f"Context: {json.dumps(ctx_summary, ensure_ascii=False)}"
        if history:
            lines = []
            for h in history[-8:]:
                who = "用户" if h.get("role") == "user" else "助手"
                text = str(h.get("text", ""))[:200]
                if text:
                    lines.append(f"{who}: {text}")
            if lines:
                user += "\n对话历史（旧→新）:\n" + "\n".join(lines)
        user += f"\nUser: {message}"
        steps, reply, err = None, None, None
        # one retry with the error fed back: a bad first answer is usually
        # fixable (unknown tool id, malformed JSON), and without the retry the
        # fallback to rules was silent and indistinguishable from "no LLM"
        for attempt in (1, 2):
            try:
                raw = provider.chat(system, user)
                data = json.loads(raw[raw.find("{"): raw.rfind("}") + 1])
            except Exception as exc:
                logger.warning("LLM planner attempt %d failed (API/JSON): %s", attempt, exc)
                err = f"invalid response: {exc}"
                if attempt == 1:
                    user = (f"Your previous answer was not valid JSON ({exc}). "
                            "Reply again with ONLY one JSON object.")[:6000] + f"\nUser: {message}"
                continue
            if not isinstance(data, dict):
                err = "response is not a JSON object"
                user = "Your previous answer was not a JSON object. Reply again with ONLY one JSON object."
                continue
            plan_ir = PlanIR.from_payload(data, source="llm")
            if not plan_ir.goal:
                plan_ir.goal = message[:240]
            steps = compile_plan(
                plan_ir, self.registry,
                templates_dir=Path(__file__).resolve().parents[2] / "templates",
            )
            if plan_ir.validation:
                err = "; ".join(plan_ir.validation[:4])
                logger.warning("LLM planner attempt %d used unknown capabilities: %s", attempt, err)
                if attempt == 1:
                    user = (f"Your plan used unknown capabilities ({err}). "
                            "Use ONLY capability ids from the catalog. Reply again with ONLY one JSON object.") + f"\nUser: {message}"
                continue
            if not plan_ir.expected_outputs:
                plan_ir.expected_outputs = PlanIR.from_steps(message, steps, source="llm").expected_outputs
            schema_errors = validate_steps(steps, self.registry)
            if schema_errors and attempt == 1:
                err = "; ".join(schema_errors[:4])
                logger.warning("LLM planner attempt 1 args invalid: %s", err)
                user = (
                    "Your plan has invalid args: " + err + ". "
                    "Fix arg types/fields to match each tool's schema exactly. "
                    "Reply again with ONLY one JSON object."
                ) + f"\nUser: {message}"
                continue
            if schema_errors:
                # pass through loudly: the registry's execution boundary
                # re-validates and refuses truly invalid args with a clean
                # per-tool error, so a warned plan beats a dead one — but
                # nothing illegal can execute anymore
                logger.warning(
                    "LLM planner args still invalid after retry (%s); passing through", schema_errors
                )
            reply = str(data.get("reply") or "").strip()
            self.last_plan_ir = plan_ir
            return steps, (reply if reply and not steps else None)
        if err:
            logger.warning("LLM planner giving up after retry (%s); falling back to rules", err)
        return None, None
