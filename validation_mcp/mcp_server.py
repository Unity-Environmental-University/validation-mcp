#!/usr/bin/env python3
"""
Validation MCP Server (stdio)

Supports multiple validation strategies and can optionally update Kanban MCP
cards with results to replace sprint-manager test gating.

Tools:
- validation_handshake(user_key, name)
- validate_test(user_key, external_id, cmd)
- validate_metric(user_key, external_id, script, threshold)
- validate_rubric(user_key, external_id, rubric_json|rubric_path, evidence?)
- validate_story(user_key, external_id, acceptance_criteria?, evidence_map?)
- evaluate_draft(rubric_json|rubric_path, instructions?, draft_text|draft_path, n_graders?)

Behavior:
- Returns a normalized result: {strategy, passed, score?, details, ts} as MCP text.
- If Kanban MCP is configured (env), updates the matching card (by external_id):
  - Adds a tag (validated/needs-validation)
  - Updates a field 'validation' with the JSON result
  - Optionally moves column on pass/fail via env mapping

Env (optional):
- KANBAN_MCP_HTTP / KANBAN_MCP_CMD / KANBAN_MCP_TOKEN
- VALIDATION_MOVE_ON_PASS (column name; e.g., "Done")
- VALIDATION_MOVE_ON_FAIL (column name; e.g., "Needs Validation")
- VALIDATION_USER_KEY (default: dev)
"""
import json
import os
import sys
import time
import subprocess
import urllib.request
import urllib.error
from pathlib import Path
from typing import Any, Dict, Optional

# Repo paths
REPO_ROOT = Path(__file__).resolve().parents[2]


def _now_iso() -> str:
    import datetime
    return datetime.datetime.utcnow().isoformat() + 'Z'


def mcp_result_text(text: str) -> Dict[str, Any]:
    return {"content": [{"type": "text", "text": text}]}


def _safe_read_json(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def list_tools() -> Dict[str, Any]:
    return {
        "tools": [
            {"name": "validation_handshake", "description": "Register or verify user", "inputSchema": {"type": "object", "properties": {"user_key": {"type": "string"}, "name": {"type": "string"}}, "required": ["user_key", "name"]}},
            {"name": "validate_test", "description": "Run a test command; pass if exit=0", "inputSchema": {"type": "object", "properties": {"user_key": {"type": "string"}, "external_id": {"type": "string"}, "cmd": {"type": "string"}}, "required": ["user_key", "external_id", "cmd"]}},
            {"name": "validate_metric", "description": "Run a script that prints a JSON metric; compare to threshold", "inputSchema": {"type": "object", "properties": {"user_key": {"type": "string"}, "external_id": {"type": "string"}, "script": {"type": "string"}, "threshold": {"type": "number"}}, "required": ["user_key", "external_id", "script", "threshold"]}},
            {"name": "validate_rubric", "description": "Score evidence against a rubric (JSON or file path)", "inputSchema": {"type": "object", "properties": {"user_key": {"type": "string"}, "external_id": {"type": "string"}, "rubric_json": {"type": "object"}, "rubric_path": {"type": "string"}, "evidence": {"type": "string"}}, "required": ["user_key", "external_id"]}},
            {"name": "validate_story", "description": "Check acceptance criteria (all true) with optional evidence_map", "inputSchema": {"type": "object", "properties": {"user_key": {"type": "string"}, "external_id": {"type": "string"}, "criteria": {"type": "array", "items": {"type": "string"}}, "evidence_map": {"type": "object"}}, "required": ["user_key", "external_id"]}},
            {"name": "evaluate_draft", "description": "Evaluate a real student draft against a rubric + instructions. Requires LLM provider or caller-supplied grader_results.", "inputSchema": {"type": "object", "properties": {"rubric_json": {"type": "object"}, "rubric_path": {"type": "string"}, "instructions": {"type": "string"}, "draft_text": {"type": "string"}, "draft_path": {"type": "string"}, "n_graders": {"type": "integer"}, "grader_results": {"type": "object"}, "coaching_mode": {"type": "string"}}, "required": []}},
            {"name": "provider_status", "description": "Report Validation MCP provider configuration status and setup hints", "inputSchema": {"type": "object", "properties": {} , "required": []}},
            {"name": "generate_sample_draft", "description": "Generate a synthetic draft likely to satisfy a rubric (for testing only)", "inputSchema": {"type": "object", "properties": {"rubric_json": {"type": "object"}, "rubric_path": {"type": "string"}, "desired_score": {"type": "number"}, "word_count": {"type": "integer"}, "notes": {"type": "string"}}, "required": []}},
            {"name": "validate_citations", "description": "Parse citations (offline) and flag that DOI verification is pending (stub)", "inputSchema": {"type": "object", "properties": {"draft_text": {"type": "string"}, "draft_path": {"type": "string"}, "style": {"type": "string"}}, "required": []}},
            {"name": "doi_lookup", "description": "Stub: DOI verification tool (will use Crossref/OpenAlex). Currently returns pending.", "inputSchema": {"type": "object", "properties": {"doi": {"type": "string"}}, "required": ["doi"]}},
            {"name": "empathizer_probe", "description": "Assess engagement, frustration, success perception, identity and metaphor spaces (LLM; no generation)", "inputSchema": {"type": "object", "properties": {"utterances": {"type": "string"}, "notes": {"type": "string"}}, "required": ["utterances"]}},
        ]
    }


class ValidationServer:
    def __init__(self) -> None:
        # Kanban integration
        self.kanban_available = bool(os.environ.get("KANBAN_MCP_HTTP") or os.environ.get("KANBAN_MCP_CMD"))
        self.move_on_pass = os.environ.get("VALIDATION_MOVE_ON_PASS", "").strip() or None
        self.move_on_fail = os.environ.get("VALIDATION_MOVE_ON_FAIL", "").strip() or None
        self.user_key_default = os.environ.get("VALIDATION_USER_KEY", "dev")
        # Events
        self.events_path = os.environ.get("VALIDATION_EVENTS_FILE", str(REPO_ROOT / ".local_context" / "logs" / "validation_events.jsonl"))
        try:
            Path(self.events_path).parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

    def _delegate_drafting(self, tool: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Delegate drafting tool to external drafting-mcp via stdio.
        Uses DRAFTING_MCP_CMD if set; returns pending hint otherwise.
        """
        cmd = os.environ.get("DRAFTING_MCP_CMD", "").strip()
        if not cmd:
            return {"strategy": tool, "passed": False, "details": {"status": "pending", "hint": "Set DRAFTING_MCP_CMD to enable drafting-mcp delegation."}, "ts": _now_iso()}
        try:
            import subprocess
            req1 = {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}
            req2 = {"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {"name": tool, "arguments": args}}
            proc = subprocess.Popen(cmd.split(), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            assert proc.stdin and proc.stdout
            proc.stdin.write(json.dumps(req1) + "\n"); proc.stdin.flush()
            proc.stdin.write(json.dumps(req2) + "\n"); proc.stdin.flush()
            _ = proc.stdout.readline()
            line = proc.stdout.readline().strip()
            try:
                proc.kill()
            except Exception:
                pass
            resp = json.loads(line)
            if "error" in resp:
                return {"strategy": tool, "passed": False, "details": {"status": "error", "error": resp.get("error")}, "ts": _now_iso()}
            content = ((resp.get("result") or {}).get("content") or [{}])[0]
            text = content.get("text", "{}")
            try:
                return json.loads(text)
            except Exception:
                return {"strategy": tool, "passed": False, "details": {"status": "error", "raw": text}, "ts": _now_iso()}
        except Exception as e:
            return {"strategy": tool, "passed": False, "details": {"status": "error", "hint": str(e)}, "ts": _now_iso()}

    def _emit_event(self, kind: str, payload: Dict[str, Any]) -> None:
        try:
            rec = {"ts": _now_iso(), "type": kind, **payload}
            with open(self.events_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except Exception:
            pass

    def _post_kanban(self, external_id: str, result: Dict[str, Any]) -> None:
        if not self.kanban_available:
            return
        client = None
        try:
            from scripts.sprint_management.kanban_mcp_client import KanbanMCPClient as _K
            client = _K()
        except Exception:
            try:
                from kanban_mcp_client import KanbanMCPClient as _K
                client = _K()
            except Exception:
                return
        # update validation field
        client.update_field(external_id, "validation", result)
        # tag and optionally move
        if result.get("passed"):
            client.add_tags(external_id, ["validated"])
            if self.move_on_pass:
                client.move_story(external_id, self.move_on_pass)
        else:
            client.add_tags(external_id, ["needs-validation"])
            if self.move_on_fail:
                client.move_story(external_id, self.move_on_fail)

    def _result(self, strategy: str, passed: bool, details: Dict[str, Any]) -> Dict[str, Any]:
        out = {"strategy": strategy, "passed": bool(passed), "details": details, "ts": _now_iso()}
        return out

    def validate_test(self, user_key: str, external_id: str, cmd: str) -> Dict[str, Any]:
        start = time.time()
        try:
            cp = subprocess.run(["bash", "-lc", cmd], cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=120)
            passed = (cp.returncode == 0)
            det = {"cmd": cmd, "rc": cp.returncode, "stdout": cp.stdout[-4000:], "stderr": cp.stderr[-4000:], "duration_ms": int((time.time()-start)*1000)}
        except subprocess.TimeoutExpired as e:
            passed = False
            det = {"cmd": cmd, "error": "timeout", "duration_ms": int((time.time()-start)*1000)}
        res = self._result("test", passed, det)
        self._post_kanban(external_id, res)
        return res

    def validate_metric(self, user_key: str, external_id: str, script: str, threshold: float) -> Dict[str, Any]:
        # Run script which prints a JSON object with a 'value' number
        try:
            cp = subprocess.run(["bash", "-lc", script], cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=60)
            data = json.loads(cp.stdout.strip() or "{}") if cp.returncode == 0 else {}
        except Exception as e:
            data = {"error": str(e)}
        value = float(data.get("value", float("nan"))) if isinstance(data, dict) else float("nan")
        passed = (value == value) and (value >= float(threshold))  # NaN check
        det = {"script": script, "threshold": threshold, "value": value, "raw": data}
        res = self._result("metric", passed, det)
        self._post_kanban(external_id, res)
        return res

    def validate_rubric(self, user_key: str, external_id: str, rubric_json: Optional[dict] = None, rubric_path: Optional[str] = None, evidence: str = "") -> Dict[str, Any]:
        rubric = rubric_json
        if rubric is None and rubric_path:
            rubric = _safe_read_json(Path(rubric_path))
        if not rubric or "criteria" not in rubric:
            res = self._result("rubric", False, {"error": "invalid rubric"})
            self._post_kanban(external_id, res)
            return res
        total_w = 0.0
        score = 0.0
        crit_results = []
        for c in rubric.get("criteria", []):
            w = float(c.get("weight", 1.0))
            total_w += w
            # Very simple: if keyword appears in evidence or checklist met
            ok = False
            if "keyword" in c and evidence:
                ok = c["keyword"].lower() in evidence.lower()
            if "check" in c and isinstance(c["check"], bool):
                ok = bool(c["check"])  # allow pre-evaluated
            if ok:
                score += w
            crit_results.append({"name": c.get("name", ""), "weight": w, "ok": ok})
        pct = (score / total_w) if total_w > 0 else 0.0
        passed = pct >= float(rubric.get("pass_threshold", 0.7))
        det = {"score_weighted": round(pct, 3), "criteria": crit_results}
        res = self._result("rubric", passed, det)
        self._post_kanban(external_id, res)
        return res

    def validate_story(self, user_key: str, external_id: str, criteria: Optional[list] = None, evidence_map: Optional[dict] = None) -> Dict[str, Any]:
        # If criteria not provided, try to fetch from local StoryGoalMCP DB
        crit = criteria or []
        if not crit:
            try:
                # Try sibling suite repo first: ../story-goal-mcp
                import sqlite3
                sg_db = REPO_ROOT.parent / "story-goal-mcp" / "story_goals.db"
                if not sg_db.exists():
                    # Fallback to in-repo DB if running embedded
                    sg_db = REPO_ROOT / "story_goals.db"
                conn = sqlite3.connect(sg_db)
                cur = conn.execute("SELECT acceptance_criteria FROM stories WHERE id = ?", (external_id,))
                row = cur.fetchone()
                if row and row[0]:
                    import json as _j
                    crit = _j.loads(row[0])
                conn.close()
            except Exception:
                crit = []
        ev = evidence_map or {}
        results = []
        all_ok = True
        for idx, c in enumerate(crit):
            ok = bool(ev.get(str(idx)) or ev.get(c) or False)
            results.append({"criterion": c, "ok": ok})
            if not ok:
                all_ok = False
        det = {"checked": len(results), "results": results}
        res = self._result("story", all_ok and len(results) > 0, det)
        self._post_kanban(external_id, res)
        return res

    # --- Student Draft Evaluator (offline heuristic MVP) ---
    def evaluate_draft(self, rubric_json: Optional[dict] = None, rubric_path: Optional[str] = None, instructions: str = "", draft_text: Optional[str] = None, draft_path: Optional[str] = None, n_graders: int = 1, grader_results: Optional[dict] = None, coaching_mode: Optional[str] = None) -> Dict[str, Any]:
        """Evaluate a real student draft.

        Requires either:
          - caller-supplied grader_results (structured scores/feedback), or
          - an LLM provider (OpenAI) configured via OPENAI_API_KEY (simple MVP).
        No heuristic grading path exists.
        """
        if grader_results and isinstance(grader_results, dict):
            out = {
                "strategy": "draft_evaluator",
                "passed": bool(grader_results.get("passed", False)),
                "score": float(grader_results.get("score", 0.0) or 0.0),
                "details": {
                    "per_criterion": grader_results.get("per_criterion", []),
                    "instruction_fit_score": grader_results.get("instruction_fit_score", None),
                    "coaching_suggestions": grader_results.get("coaching_suggestions", []),
                },
                "ts": _now_iso(),
            }
            self._emit_event("draft.evaluated", {"score": out["score"], "passed": out["passed"], "mode": "external"})
            return out

        # LLM provider path (OpenAI or Ollama)
        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        ollama_host = os.environ.get("OLLAMA_HOST", "").strip()
        ollama_model = os.environ.get("OLLAMA_MODEL", "llama3.1:8b").strip()
        use_openai = bool(api_key)
        use_ollama = (not use_openai) and bool(ollama_host)
        if not (use_openai or use_ollama):
            raise RuntimeError("LLM provider not configured. Set OPENAI_API_KEY or OLLAMA_HOST/OLLAMA_MODEL, or provide 'grader_results'.")

        # Load rubric and draft
        rubric = rubric_json
        if rubric is None and rubric_path:
            rubric = _safe_read_json(Path(rubric_path))
        rubric = rubric or {"criteria": [], "pass_threshold": 0.7}
        pass_threshold = float(rubric.get("pass_threshold", 0.7))
        # Default citation style (configurable), unless rubric specifies
        citation_style = rubric.get("citation_style") or os.environ.get("VALIDATION_CITATION_STYLE", "APA7")
        # Avoid leaking example hints to the grader model; keep only essential fields
        rubric_for_model = {
            "pass_threshold": pass_threshold,
            "criteria": rubric.get("criteria", []),
        }
        if "assignment_body" in rubric:
            rubric_for_model["assignment_body"] = rubric.get("assignment_body")
        rubric_for_model["citation_style"] = citation_style

        draft = draft_text
        if not draft and draft_path:
            try:
                draft = Path(draft_path).read_text(encoding="utf-8")
            except Exception:
                draft = ""
        draft = draft or ""

        n = max(1, int(n_graders or 1))

        def _openai_grade_one(idx: int) -> dict:
            sys_prompt = (
                "You are a strict grader. Respond ONLY with JSON, no prose. "
                "Fields: passed (bool), score (0..1), per_criterion (list of {name, weight, score, comment}), "
                "instruction_fit_score (0..1), coaching_suggestions (list of strings). "
                "When relevant, check citation format; default citation_style is " + citation_style + "."
            )
            user_msg = json.dumps({
                "rubric": rubric_for_model,
                "instructions": instructions or "",
                "draft": draft,
                "notes": {"grader_id": idx}
            }, ensure_ascii=False)
            payload = {
                "model": os.environ.get("VALIDATION_OPENAI_MODEL_GRADER", os.environ.get("OPENAI_MODEL", "gpt-4o-mini")),
                "temperature": float(os.environ.get("VALIDATION_OPENAI_TEMP", "0.1")),
                "messages": [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_msg}
                ],
                "max_tokens": int(os.environ.get("VALIDATION_OPENAI_MAX_TOKENS", "600")),
            }
            req = urllib.request.Request(
                os.environ.get("VALIDATION_OPENAI_ENDPOINT", "https://api.openai.com/v1/chat/completions"),
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}",
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=20) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                self._emit_event("provider.selected", {"provider": "openai", "model": payload["model"]})
            except urllib.error.HTTPError as e:
                err = e.read().decode("utf-8", errors="ignore")
                raise RuntimeError(f"OpenAI error: {e.code} {err}")
            except Exception as e:
                raise RuntimeError(f"OpenAI request failed: {e}")

            msg = (((data.get("choices") or [{}])[0]).get("message") or {}).get("content", "")
            # Extract JSON from content (robust to code fences)
            txt = msg.strip()
            def _parse_json(s: str):
                s = s.strip()
                if s.startswith("```"):
                    s = s.strip('`')
                    if s.lower().startswith("json\n"):
                        s = s[5:]
                a = s.find('{'); b = s.rfind('}')
                if a != -1 and b != -1 and b > a:
                    s = s[a:b+1]
                return json.loads(s)
            try:
                obj = _parse_json(txt)
            except Exception:
                raise RuntimeError("Grader returned non-JSON content")
            return obj

        def _ollama_grade_one(idx: int) -> dict:
            sys_prompt = (
                "You are a strict grader. Respond ONLY with JSON, no prose. "
                "Fields: passed (bool), score (0..1), per_criterion (list of {name, weight, score, comment}), "
                "instruction_fit_score (0..1), coaching_suggestions (list of strings). "
                "When relevant, check citation format; default citation_style is " + citation_style + "."
            )
            user_msg = json.dumps({
                "rubric": rubric_for_model,
                "instructions": instructions or "",
                "draft": draft,
                "notes": {"grader_id": idx}
            }, ensure_ascii=False)
            payload = {
                "model": ollama_model,
                "messages": [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_msg}
                ],
                "stream": False
            }
            url = (ollama_host.rstrip('/') if ollama_host else "http://127.0.0.1:11434") + "/api/chat"
            req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json"}, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=25) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                self._emit_event("provider.selected", {"provider": "ollama", "model": payload["model"]})
            except Exception as e:
                raise RuntimeError(f"Ollama request failed: {e}")
            content = ((data.get("message") or {}).get("content") or "").strip()
            def _parse_json2(s: str):
                s = s.strip()
                if s.startswith("```"):
                    s = s.strip('`')
                    if s.lower().startswith("json\n"):
                        s = s[5:]
                a = s.find('{'); b = s.rfind('}')
                if a != -1 and b != -1 and b > a:
                    s = s[a:b+1]
                return json.loads(s)
            try:
                return _parse_json2(content)
            except Exception:
                raise RuntimeError("Grader returned non-JSON content")

        grades = []
        for i in range(n):
            g = _openai_grade_one(i) if use_openai else _ollama_grade_one(i)
            grades.append(g)

        # Aggregate
        scores = [float(g.get("score", 0.0) or 0.0) for g in grades]
        avg = sum(scores) / max(1, len(scores))
        passed = avg >= pass_threshold

        # Merge per_criterion by name (avg score, keep first weight, concat comments truncated)
        per_map = {}
        for g in grades:
            for c in g.get("per_criterion", []) or []:
                name = c.get("name", "")
                if name not in per_map:
                    per_map[name] = {"name": name, "weight": c.get("weight", 1.0), "scores": [], "comments": []}
                per_map[name]["scores"].append(float(c.get("score", 0.0) or 0.0))
                if c.get("comment"):
                    per_map[name]["comments"].append(str(c.get("comment"))[:200])
        per_agg = []
        for name, v in per_map.items():
            s = sum(v["scores"]) / max(1, len(v["scores"]))
            per_agg.append({"name": name, "weight": v.get("weight", 1.0), "score": round(s, 3), "comment": "; ".join(v["comments"])[:300]})

        instr_scores = [float(g.get("instruction_fit_score", 0.0) or 0.0) for g in grades]
        instr_avg = sum(instr_scores) / max(1, len(instr_scores)) if instr_scores else None
        coach = []
        for g in grades:
            coach.extend([str(x) for x in (g.get("coaching_suggestions") or [])])
        coach = coach[:10]

        # Socratic GM coaching: turn suggestions into questions or request an LLM pass (future)
        mode = (coaching_mode or os.environ.get("COACHING_MODE", "socratic_gm")).lower()
        if mode.startswith("socratic"):
            qs = []
            for c in per_agg:
                name = c.get("name", "criterion")
                s = float(c.get("score", 0))
                if s < 0.8:
                    qs.append(f"For {name}, what specific change would raise quality? Give one example from your draft.")
            # Fallback to transforming existing suggestions into questions
            if not qs:
                for tip in coach:
                    tip = str(tip).strip()
                    if not tip:
                        continue
                    qs.append("What would it look like if you " + tip[0].lower() + tip[1:] + "?")
            # Replace coaching suggestions with questions
            coach = qs[:5]
            self._emit_event("coaching.socratic_gm", {"questions": coach, "mode": mode})

        result = {
            "strategy": "draft_evaluator",
            "passed": bool(passed),
            "score": round(avg, 3),
            "details": {
                "per_criterion": per_agg,
                "instruction_fit_score": (round(instr_avg, 3) if instr_avg is not None else None),
                "coaching_suggestions": coach,
                "graders": len(grades),
            },
            "ts": _now_iso(),
        }
        self._emit_event("draft.evaluated", {"score": result["score"], "passed": result["passed"], "mode": "openai", "graders": len(grades)})
        return result

    def generate_sample_draft(self, rubric_json: Optional[dict] = None, rubric_path: Optional[str] = None, desired_score: float = 0.8, word_count: int = 600, notes: str = "") -> Dict[str, Any]:
        """Generate a synthetic draft intended to score above desired_score for testing.

        Uses the same provider path (OpenAI preferred, else Ollama). Does not include rubric examples.
        Returns the draft text and an immediate evaluation result (n_graders=1) for feedback.
        """
        # Provider selection shared with evaluate_draft
        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        ollama_host = os.environ.get("OLLAMA_HOST", "").strip()
        ollama_model = os.environ.get("OLLAMA_MODEL", "llama3.1:8b").strip()
        use_openai = bool(api_key)
        use_ollama = (not use_openai) and bool(ollama_host)
        if not (use_openai or use_ollama):
            raise RuntimeError("LLM provider not configured. Set OPENAI_API_KEY or OLLAMA_HOST/OLLAMA_MODEL.")

        # Load rubric and prepare safe model payload
        rubric = rubric_json
        if rubric is None and rubric_path:
            rubric = _safe_read_json(Path(rubric_path))
        rubric = rubric or {"criteria": [], "pass_threshold": 0.7}
        pass_threshold = float(rubric.get("pass_threshold", 0.7))
        citation_style = rubric.get("citation_style") or os.environ.get("VALIDATION_CITATION_STYLE", "APA7")

        rubric_for_model = {
            "pass_threshold": pass_threshold,
            "criteria": rubric.get("criteria", []),
        }
        if "assignment_body" in rubric:
            rubric_for_model["assignment_body"] = rubric.get("assignment_body")
        rubric_for_model["citation_style"] = citation_style

        sys_prompt = (
            "You generate synthetic student drafts for testing. Produce a cohesive draft that is likely to satisfy the rubric and assignment_body. "
            "Do NOT include explanations or JSON, return ONLY the draft text. When relevant, adhere to the given citation_style (default: " + citation_style + ")."
        )
        user_payload = {
            "rubric": rubric_for_model,
            "target_score": float(desired_score),
            "word_count": int(word_count),
            "notes": notes or ""
        }

        def _openai_gen() -> str:
            payload = {
                "model": os.environ.get("VALIDATION_OPENAI_MODEL_GRADER", os.environ.get("OPENAI_MODEL", "gpt-4o-mini")),
                "temperature": float(os.environ.get("VALIDATION_OPENAI_TEMP", "0.2")),
                "messages": [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)}
                ],
                "max_tokens": int(os.environ.get("VALIDATION_OPENAI_MAX_TOKENS", "1200")),
            }
            req = urllib.request.Request(
                os.environ.get("VALIDATION_OPENAI_ENDPOINT", "https://api.openai.com/v1/chat/completions"),
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            content = (((data.get("choices") or [{}])[0]).get("message") or {}).get("content", "").strip()
            return content

        def _ollama_gen() -> str:
            payload = {
                "model": ollama_model,
                "messages": [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)}
                ],
                "stream": False
            }
            url = (ollama_host.rstrip('/') if ollama_host else "http://127.0.0.1:11434") + "/api/chat"
            req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json"}, method="POST")
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            return ((data.get("message") or {}).get("content") or "").strip()

        draft = _openai_gen() if use_openai else _ollama_gen()

        # Immediate evaluation (n_graders=1) for feedback
        eval_res = self.evaluate_draft(rubric_json=rubric, instructions="", draft_text=draft, draft_path=None, n_graders=1)
        out = {
            "draft": draft,
            "evaluation": eval_res,
        }
        return out

    # ---- Citation validation (stub with offline parsing) ----
    def validate_citations(self, draft_text: Optional[str] = None, draft_path: Optional[str] = None, style: Optional[str] = None) -> Dict[str, Any]:
        """Offline citation parsing stub.

        - Extracts DOIs via regex and returns them.
        - Does NOT perform network verification (annoying-but-not-blocking stub).
        - Always sets needs_verification=true and status='pending'.
        """
        text = draft_text or ""
        if not text and draft_path:
            try:
                text = Path(draft_path).read_text(encoding="utf-8")
            except Exception:
                text = ""
        # DOI regex from Crossref recommendations (simplified)
        import re
        doi_pat = re.compile(r"\b10\.\d{4,9}/\S+\b", re.IGNORECASE)
        dois = sorted(set(doi_pat.findall(text)))
        details = {
            "style": style or os.environ.get("VALIDATION_CITATION_STYLE", "APA7"),
            "in_text_count": len(re.findall(r"\([A-Z][A-Za-z]+,\s*\d{4}\)", text)),
            "references_section": bool(re.search(r"\n\s*references\s*\n", text, flags=re.IGNORECASE)),
            "dois_found": dois,
            "needs_verification": True,
            "status": "pending",
            "issues": ["DOI verification pending (network disabled/not implemented)"] if dois else ["No DOIs found; verification skipped"],
        }
        return {"strategy": "citations", "passed": True, "details": details, "ts": _now_iso()}

    def doi_lookup(self, doi: str) -> Dict[str, Any]:
        """Stub DOI lookup: always returns pending with a hint to enable network or use dedicated doi-verify-mcp later."""
        return {
            "doi": doi,
            "status": "pending",
            "hint": "Enable network verification in Validation MCP or call future doi-verify-mcp."
        }

    # ---- Empathizer Probe (LLM-based; questions-only assessment) ----
    def empathizer_probe(self, utterances: str, notes: str = "") -> Dict[str, Any]:
        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        ollama_host = os.environ.get("OLLAMA_HOST", "").strip()
        ollama_model = os.environ.get("OLLAMA_MODEL", "llama3.1:8b").strip()
        use_openai = bool(api_key)
        use_ollama = (not use_openai) and bool(ollama_host)
        if not (use_openai or use_ollama):
            return {"status": "pending", "hint": "No LLM provider configured. Set OPENAI_API_KEY or OLLAMA_HOST."}
        sys_prompt = (
            "You are an empathic tutor-coach. Analyze the student's utterances and return ONLY JSON with fields: "
            "engagement (0..1), frustration (0..1), success (0..1), identity_tags (list of strings), cares_about (list), metaphor_spaces (list), "
            "and questions (3-5 short Socratic questions to re-center). Do not invent biographical facts."
        )
        user_msg = json.dumps({"utterances": utterances, "notes": notes}, ensure_ascii=False)
        def parse_choice(msg: str) -> dict:
            s = msg.strip()
            if s.startswith("```"):
                s = s.strip('`')
                if s.lower().startswith("json\n"):
                    s = s[5:]
            a = s.find('{'); b = s.rfind('}')
            if a != -1 and b != -1 and b > a:
                s = s[a:b+1]
            return json.loads(s)
        if use_openai:
            payload = {
                "model": os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
                "temperature": 0.2,
                "messages": [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_msg}
                ],
                "max_tokens": 400,
            }
            req = urllib.request.Request(os.environ.get("VALIDATION_OPENAI_ENDPOINT", "https://api.openai.com/v1/chat/completions"), data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}, method="POST")
            with urllib.request.urlopen(req, timeout=20) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            msg = (((data.get("choices") or [{}])[0]).get("message") or {}).get("content", "")
            try:
                obj = parse_choice(msg)
            except Exception:
                obj = {"status": "error", "hint": "Non-JSON response from model"}
            self._emit_event("coach.empathizer", {"status": obj.get("status", "ok")})
            return obj
        else:
            payload = {"model": ollama_model, "messages": [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_msg}], "stream": False}
            url = (ollama_host.rstrip('/') if ollama_host else "http://127.0.0.1:11434") + "/api/chat"
            req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json"}, method="POST")
            with urllib.request.urlopen(req, timeout=25) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            content = ((data.get("message") or {}).get("content") or "").strip()
            try:
                obj = parse_choice(content)
            except Exception:
                obj = {"status": "error", "hint": "Non-JSON response from model"}
            self._emit_event("coach.empathizer", {"status": obj.get("status", "ok")})
            return obj

    def provider_status(self) -> Dict[str, Any]:
        """Return a summary of provider configuration and how to enable LLM-backed grading."""
        status = {
            "openai_key": bool(os.environ.get("OPENAI_API_KEY")),
            "anthropic_key": bool(os.environ.get("ANTHROPIC_API_KEY")),
            "providers_file": bool(os.environ.get("VALIDATION_PROVIDERS_PATH")),
            "ollama": bool(os.environ.get("OLLAMA_HOST")),
            "docs": "docs/VALIDATION-MCP.md",
            "hint": "Set VALIDATION_PROVIDERS_PATH for chains, or export OPENAI_API_KEY / ANTHROPIC_API_KEY, or set OLLAMA_HOST/OLLAMA_MODEL to use a local model."
        }
        return status

    def handle_call(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        uk = args.get("user_key") or self.user_key_default
        if name == "validation_handshake":
            return mcp_result_text(json.dumps({"user_key": uk, "repo": str(REPO_ROOT)}))
        if name == "validate_test":
            res = self.validate_test(uk, args.get("external_id", ""), args.get("cmd", ""))
            return mcp_result_text(json.dumps(res, ensure_ascii=False))
        if name == "validate_metric":
            res = self.validate_metric(uk, args.get("external_id", ""), args.get("script", ""), float(args.get("threshold", 0)))
            return mcp_result_text(json.dumps(res, ensure_ascii=False))
        if name == "validate_rubric":
            rj = args.get("rubric_json")
            rp = args.get("rubric_path")
            ev = args.get("evidence", "")
            res = self.validate_rubric(uk, args.get("external_id", ""), rj, rp, ev)
            return mcp_result_text(json.dumps(res, ensure_ascii=False))
        if name == "validate_story":
            res = self.validate_story(uk, args.get("external_id", ""), args.get("criteria"), args.get("evidence_map"))
            return mcp_result_text(json.dumps(res, ensure_ascii=False))
        if name == "evaluate_draft":
            res = self._delegate_drafting("evaluate_draft", {
                "rubric_json": args.get("rubric_json"),
                "rubric_path": args.get("rubric_path"),
                "instructions": args.get("instructions", ""),
                "draft_text": args.get("draft_text"),
                "draft_path": args.get("draft_path"),
                "n_graders": int(args.get("n_graders", 1) or 1)
            })
            return mcp_result_text(json.dumps(res, ensure_ascii=False))
        if name == "provider_status":
            res = self.provider_status()
            return mcp_result_text(json.dumps(res, ensure_ascii=False))
        if name == "generate_sample_draft":
            res = self._delegate_drafting("generate_sample_draft", {
                "rubric_json": args.get("rubric_json"),
                "rubric_path": args.get("rubric_path"),
                "desired_score": float(args.get("desired_score", 0.8) or 0.8),
                "word_count": int(args.get("word_count", 600) or 600),
                "notes": args.get("notes", "")
            })
            return mcp_result_text(json.dumps(res, ensure_ascii=False))
        if name == "validate_citations":
            res = self._delegate_drafting("validate_citations", {
                "draft_text": args.get("draft_text"),
                "draft_path": args.get("draft_path"),
                "style": args.get("style")
            })
            return mcp_result_text(json.dumps(res, ensure_ascii=False))
        if name == "doi_lookup":
            res = self._delegate_drafting("doi_lookup", {"doi": args.get("doi", "")})
            return mcp_result_text(json.dumps(res, ensure_ascii=False))
        if name == "empathizer_probe":
            res = self.empathizer_probe(args.get("utterances", ""), args.get("notes", ""))
            return mcp_result_text(json.dumps(res, ensure_ascii=False))
        raise ValueError(f"Unknown tool: {name}")


def main():
    server = ValidationServer()
    while True:
        line = sys.stdin.readline()
        if not line:
            break
        try:
            req = json.loads(line)
            method = req.get('method')
            _id = req.get('id')
            if method == 'initialize':
                resp = {"jsonrpc": "2.0", "id": _id, "result": {"serverInfo": {"name": "validation-mcp", "version": "0.1.0"}}}
            elif method == 'tools/list':
                resp = {"jsonrpc": "2.0", "id": _id, "result": list_tools()}
            elif method == 'tools/call':
                params = req.get('params') or {}
                name = params.get('name')
                args = params.get('arguments') or {}
                try:
                    result = server.handle_call(name, args)
                    resp = {"jsonrpc": "2.0", "id": _id, "result": result}
                except Exception as e:
                    resp = {"jsonrpc": "2.0", "id": _id, "error": {"code": -32603, "message": f"Internal error: {e}"}}
            else:
                resp = {"jsonrpc": "2.0", "id": _id, "error": {"code": -32601, "message": "Method not found"}}
        except Exception as e:
            resp = {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": f"Parse error: {e}"}}
        sys.stdout.write(json.dumps(resp) + "\n")
        sys.stdout.flush()


if __name__ == '__main__':
    main()
