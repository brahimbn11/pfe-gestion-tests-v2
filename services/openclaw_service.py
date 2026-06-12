import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)


def ask_openclaw(session_id: str, prompt: str, timeout_seconds: int = 300):
    """
    Compatibility layer:
    Keeps the same function name used by the project,
    but uses OpenAI instead of OpenClaw.
    """

    print("OPENAI START")
    print("SESSION:", session_id)

    try:
        response = client.chat.completions.create(
            model="gpt-5-mini",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            timeout=min(timeout_seconds, 120)
        )

        text = response.choices[0].message.content or ""

        print("OPENAI FINISHED")
        print("SESSION:", session_id)

        return {
            "success": True,
            "response": text.strip()
        }

    except Exception as e:

        print("OPENAI ERROR")
        print(str(e))

        return {
            "success": False,
            "response": f"Erreur OpenAI: {str(e)}",
            "error": str(e)
        }