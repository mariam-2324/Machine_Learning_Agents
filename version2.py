#!/usr/bin/env python3
"""
ml_agentic_system.py

Single-file multi-agent project (ModelSelection, Training, Evaluation) that:
- Calls Mistral (LLM) via a simple async HTTP wrapper
- Calls ElevenLabs TTS to produce an MP3 of the final answer
- Can run in multiple modes so each agent can be started as its own process (suitable for registering in Coral Server application.yaml)

Usage examples (from your Ubuntu WSL terminal):
# interactive orchestrator (sequential pipeline)
python3 ml_agentic_system.py orchestrator

# run a single agent instance (reads a single line from stdin and prints result)
echo "Task: ...\nDataset: ..." | python3 ml_agentic_system.py model_selection

Requirements:
- Python 3.10+
- Install dependencies: `pip install httpx python-dotenv`

Environment (.env) (example):
MISTRAL_API_KEY=sk_...
MISTRAL_API_URL=https://api.mistral.ai         # adjust if your provider requires an alternate endpoint
MISTRAL_MODEL=mistral-large                   # or the model name you want to use
ELEVEN_LABS_KEY=eleven_xxx...
ELEVEN_VOICE_ID=21m00Tcm4TlvDq8ikWAM          # pick from ElevenLabs voice list
ELEVEN_API_URL=https://api.elevenlabs.io
"""

import os
import sys
import json
import asyncio
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
import httpx


# ---------- Small example dataset for user testing ----------
EXAMPLE_DATASET = [
    {
        "task": "Predict student grades based on study hours and attendance",
        "dataset_summary": "Small CSV with 10 rows, columns: hours_studied, attendance_percentage, final_grade",
        "query_type": "model_selection"
    },
    {
        "task": "Train a simple regression model for student grades",
        "dataset_summary": "Same small CSV as above, 10 rows",
        "query_type": "training"
    },
    {
        "task": "Evaluate regression model for student grades",
        "dataset_summary": "Same small CSV as above, 10 rows",
        "query_type": "evaluation"
    }
]

# ---------- Load environment variables ----------
load_dotenv()

# ---------- Configuration from .env ----------
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
MISTRAL_API_URL = os.getenv("MISTRAL_API_URL", "https://api.mistral.ai")
MISTRAL_MODEL = os.getenv("MISTRAL_MODEL", "mistral-small-latest")

print("Using MISTRAL_API_KEY:", MISTRAL_API_KEY)
ELEVENLABS_API_KEY = os.getenv("SOLANA_HACKATHON")
ELEVEN_VOICE_ID = os.getenv("ELEVEN_VOICE_ID")
ELEVEN_API_URL = os.getenv("ELEVEN_API_URL", "https://api.elevenlabs.io")

# Basic validation
if not MISTRAL_API_KEY:
    raise RuntimeError("MISTRAL_API_KEY is required in your .env file")

if not ELEVENLABS_API_KEY:
    print("[warning] ELEVENLABS_API_KEY not set. TTS will be disabled.")


# ---------- Async Mistral HTTP client (OpenAI-compatible style) ----------
class AsyncMistralClient:
    def __init__(self, api_key: str, base_url: str = "https://api.mistral.ai", model: str = "mistral-large"):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._client: Optional[httpx.AsyncClient] = None

    async def _client_obj(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=60.0)
        return self._client

    async def chat(self, messages: List[Dict[str, str]], temperature: float = 0.2, max_tokens: int = 1024) -> str:
        client = await self._client_obj()
        url = f"{self.base_url}/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        resp = await client.post(url, headers=headers, json=payload)
        if resp.status_code >= 400:
            raise RuntimeError(f"Mistral API error {resp.status_code}: {resp.text}")
        data = resp.json()
        choices = data.get("choices") or []
        if choices:
            first = choices[0]
            if "message" in first and isinstance(first["message"], dict):
                return first["message"].get("content", "")
            if "text" in first:
                return first.get("text", "")
        return data.get("output") or data.get("result") or json.dumps(data)

    async def close(self):
        if self._client is not None:
            await self._client.aclose()
            self._client = None


# ---------- ElevenLabs simple TTS wrapper (async) ----------
class ElevenLabsTTS:
    def __init__(self, api_key: str, base_url: str = "https://api.elevenlabs.io"):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")

    async def text_to_speech(self, text: str, voice_id: str, out_path: str = "ml_agent_response.mp3") -> str:
        if not self.api_key:
            raise RuntimeError("ELEVEN_LABS_KEY not configured")
        url = f"{self.base_url}/v1/text-to-speech/{voice_id}"
        headers = {
            "xi-api-key": self.api_key,
            "Content-Type": "application/json",
        }
        payload = {
            "text": text,
            "model_id": "eleven_multilingual_v2"
        }
        async with httpx.AsyncClient(timeout=120.0) as client:
            r = await client.post(url, headers=headers, json=payload)
            if r.status_code >= 400:
                raise RuntimeError(f"ElevenLabs TTS error {r.status_code}: {r.text}")
            with open(out_path, "wb") as f:
                f.write(r.content)

                # 🔊 Auto-play the audio after saving
            try:
                from playsound import playsound
                playsound(out_path)
            except Exception as e:
                print(f"[warning] Could not auto-play audio: {e}")

            return out_path



# ---------- Agent implementations ----------
class AgentResult:
    def __init__(self, text: str, metadata: Optional[Dict[str, Any]] = None):
        self.text = text
        self.metadata = metadata or {}


class ModelSelectionAgent:
    SYSTEM_PROMPT = (
        "You are an ML model selection assistant. Given a short task description and dataset summary, "
        "return JSON with two keys: 'candidates' (list of {name,reason}) and 'recommended' (string)."
        " Keep answers short and machine-parseable."
    )

    def __init__(self, llm: AsyncMistralClient):
        self.llm = llm

    async def run(self, task_description: str, dataset_summary: str) -> AgentResult:
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": f"Task: {task_description}\nDataset: {dataset_summary}"},
        ]
        out = await self.llm.chat(messages, temperature=0.0, max_tokens=512)
        parsed = None
        try:
            parsed = json.loads(out)
        except Exception:
            parsed = None
        return AgentResult(text=out, metadata=parsed)


class TrainingAgent:
    SYSTEM_PROMPT = (
        "You are an ML engineer. Given the selected model name and a short dataset summary, "
        "produce a short, copy-pasteable starter training script in Python (prefer torch or sklearn), "
        "with minimal dependencies and a short usage note. Return only the code block and a short note."
    )

    def __init__(self, llm: AsyncMistralClient):
        self.llm = llm

    async def run(self, selected_model: str, dataset_summary: str) -> AgentResult:
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": f"Selected model: {selected_model}\nDataset: {dataset_summary}"},
        ]
        out = await self.llm.chat(messages, temperature=0.2, max_tokens=1024)
        return AgentResult(text=out)


class EvaluationAgent:
    SYSTEM_PROMPT = (
        "You are an ML evaluator. Explain the appropriate evaluation metrics for the task, "
        "provide python snippets to compute Accuracy, F1, and RMSE, and advise when to choose each metric."
    )

    def __init__(self, llm: AsyncMistralClient):
        self.llm = llm

    async def run(self, task_description: str, dataset_summary: str) -> AgentResult:
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": f"Task: {task_description}\nDataset: {dataset_summary}"},
        ]
        out = await self.llm.chat(messages, temperature=0.0, max_tokens=800)
        return AgentResult(text=out)


# ---------- Pipeline orchestrator ----------
async def run_pipeline(task_description: str, dataset_summary: str, speak: bool = True) -> Dict[str, Any]:
    llm = AsyncMistralClient(api_key=MISTRAL_API_KEY, base_url=MISTRAL_API_URL, model=MISTRAL_MODEL)
    sel_agent = ModelSelectionAgent(llm)
    train_agent = TrainingAgent(llm)
    eval_agent = EvaluationAgent(llm)

    sel = await sel_agent.run(task_description, dataset_summary)

    recommended = None
    if sel.metadata and isinstance(sel.metadata, dict):
        recommended = sel.metadata.get("recommended")
        if not recommended:
            candidates = sel.metadata.get("candidates")
            if isinstance(candidates, list) and candidates:
                first = candidates[0]
                if isinstance(first, dict):
                    recommended = first.get("name")

    if not recommended:
        recommended = "baseline-model"

    training = await train_agent.run(recommended, dataset_summary)
    evaluation = await eval_agent.run(task_description, dataset_summary)

    final_text = (
        "=== Model Selection ===\n" + sel.text + "\n\n"
        "=== Starter Training Code ===\n" + training.text + "\n\n"
        "=== Evaluation Guidance ===\n" + evaluation.text
    )

    audio_path = None
    if speak and ELEVENLABS_API_KEY and ELEVEN_VOICE_ID:
        tts = ElevenLabsTTS(ELEVENLABS_API_KEY, base_url=ELEVEN_API_URL)
        try:
            audio_path = await tts.text_to_speech(final_text, ELEVEN_VOICE_ID, out_path="ml_agent_response.mp3")
        except Exception as e:
            print(f"[warning] ElevenLabs TTS failed: {e}")
            audio_path = None

    await llm.close()

    return {
        "model_selection": sel.text,
        "training": training.text,
        "evaluation": evaluation.text,
        "audio_path": audio_path,
        "recommended": recommended,
    }


# ---------- CLI entrypoints ----------
async def agent_cli_model_selection():
    payload = sys.stdin.read().strip()
    if not payload:
        print("No input received")
        return
    llm = AsyncMistralClient(api_key=MISTRAL_API_KEY, base_url=MISTRAL_API_URL, model=MISTRAL_MODEL)
    agent = ModelSelectionAgent(llm)
    res = await agent.run(payload, "")
    print(res.text)
    await llm.close()


async def agent_cli_training():
    payload = sys.stdin.read().strip()
    if not payload:
        print("No input received")
        return
    lines = payload.splitlines()
    sel = lines[0].strip() if lines else ""
    dataset = "\n".join(lines[1:]) if len(lines) > 1 else ""
    if sel.lower().startswith("selected:"):
        selected_model = sel.split(":", 1)[1].strip()
    else:
        selected_model = sel
    llm = AsyncMistralClient(api_key=MISTRAL_API_KEY, base_url=MISTRAL_API_URL, model=MISTRAL_MODEL)
    agent = TrainingAgent(llm)
    res = await agent.run(selected_model, dataset)
    print(res.text)
    await llm.close()


async def agent_cli_evaluation():
    payload = sys.stdin.read().strip()
    if not payload:
        print("No input received")
        return
    llm = AsyncMistralClient(api_key=MISTRAL_API_KEY, base_url=MISTRAL_API_URL, model=MISTRAL_MODEL)
    agent = EvaluationAgent(llm)
    res = await agent.run(payload, "")
    print(res.text)
    await llm.close()


async def orchestrator_cli_interactive():
    print("ML Mentor Agents — Orchestrator (sequential). Type 'exit' to quit.")
    while True:
        task = input("Task (one-line): ").strip()
        if not task or task.lower() in ("exit", "quit"):
            print("Goodbye")
            break
        dataset = input("Dataset summary (one-line): ").strip()
        print("Running pipeline — please wait...\n")
        out = await run_pipeline(task, dataset, speak=True)
        print("\n=== Final Text Answer ===\n")
        print(
            "=== Model Selection ===\n" + out["model_selection"] + "\n\n" +
            "=== Starter Training Code ===\n" + out["training"] + "\n\n" +
            "=== Evaluation Guidance ===\n" + out["evaluation"]
        )
        print("\nAudio (if produced):", out.get("audio_path"))

        # print("\n---\n")


# ---------- main ----------
if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "orchestrator"
    if mode == "orchestrator":
        asyncio.run(orchestrator_cli_interactive())
    elif mode == "model_selection":
        asyncio.run(agent_cli_model_selection())
    elif mode == "training":
        asyncio.run(agent_cli_training())
    elif mode == "evaluation":
        asyncio.run(agent_cli_evaluation())
    else:
        print("Unknown mode")


