"""Shared async Ollama client with caching."""
import os
import aiohttp

from cache.response_cache import get_cached, set_cached

OLLAMA_URL = os.environ.get(
    "OLLAMA_URL",
    "http://localhost:11434/api/generate"
)

OLLAMA_CHAT_URL = OLLAMA_URL.replace(
    "/api/generate",
    "/api/chat"
)

OLLAMA_MODEL = os.environ.get(
    "OLLAMA_MODEL",
    "llama3.2"
)

TIMEOUT_S = float(
    os.environ.get("OLLAMA_TIMEOUT_S", "20")
)
OLLAMA_TIMEOUT_S = TIMEOUT_S


async def ollama_generate(
    prompt: str,
    system: str = None,
    temperature: float = 0.0
) -> str:

    cache_key = f"generate:{system or ''}:{prompt}"

    cached = get_cached(
        OLLAMA_MODEL,
        cache_key
    )

    if cached:
        return cached.get("response", "")

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "keep_alive": "30m",
        "options": {
            "temperature": temperature
        }
    }

    if system:
        payload["system"] = system

    try:
        timeout = aiohttp.ClientTimeout(
            total=TIMEOUT_S
        )

        async with aiohttp.ClientSession(
            timeout=timeout
        ) as session:

            async with session.post(
                OLLAMA_URL,
                json=payload
            ) as resp:

                resp.raise_for_status()

                data = await resp.json()

                result = {
                    "response": data.get(
                        "response",
                        ""
                    ).strip()
                }

                set_cached(
                    OLLAMA_MODEL,
                    cache_key,
                    result
                )

                return result["response"]

    except Exception as e:
        print(f"[llm] generate failed: {e}")
        return ""


async def ollama_chat(
    system: str,
    user: str,
    temperature: float = 0.0
) -> str:

    cache_key = f"chat:{system}:{user}"

    cached = get_cached(
        OLLAMA_MODEL,
        cache_key
    )

    if cached:
        return cached.get("response", "")

    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {
                "role": "system",
                "content": system
            },
            {
                "role": "user",
                "content": user
            }
        ],
        "stream": False,
        "keep_alive": "30m",
        "options": {
            "temperature": temperature
        }
    }

    try:
        timeout = aiohttp.ClientTimeout(
            total=TIMEOUT_S
        )

        async with aiohttp.ClientSession(
            timeout=timeout
        ) as session:

            async with session.post(
                OLLAMA_CHAT_URL,
                json=payload
            ) as resp:

                resp.raise_for_status()

                data = await resp.json()

                result = {
                    "response": data.get(
                        "message",
                        {}
                    ).get(
                        "content",
                        ""
                    ).strip()
                }

                set_cached(
                    OLLAMA_MODEL,
                    cache_key,
                    result
                )

                return result["response"]

    except Exception as e:
        print(f"[llm] chat failed: {e}")
        return ""


async def warm_ollama() -> bool:

    try:
        timeout = aiohttp.ClientTimeout(
            total=10
        )

        async with aiohttp.ClientSession(
            timeout=timeout
        ) as session:

            async with session.post(
                OLLAMA_URL,
                json={
                    "model": OLLAMA_MODEL,
                    "prompt": "hi",
                    "stream": False
                }
            ) as response:

                return response.status == 200

    except Exception:
        return False
