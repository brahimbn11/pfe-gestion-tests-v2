import subprocess
import json

OPENCLAW_BIN = "/opt/homebrew/bin/openclaw"


def ask_openclaw(session_id: str, prompt: str, timeout_seconds: int = 300):
    print("OPENCLAW START")
    print("SESSION:", session_id)

    try:
        result = subprocess.run(
            [
                OPENCLAW_BIN,
                "agent",
                "--local",
                "--session-id",
                session_id,
                "--message",
                prompt,
                "--json"
            ],
            capture_output=True,
            text=True,
            timeout=timeout_seconds
        )

        print("OPENCLAW FINISHED")
        print("SESSION:", session_id)
        print("RETURNCODE:", result.returncode)


    except subprocess.TimeoutExpired:

        print("OPENCLAW TIMEOUT")
        print("SESSION:", session_id)

        return {
            "success": False,
            "response": "OpenClaw a pris trop de temps pour répondre. Réessayez avec un message plus court ou relancez la demande.",
            "error": "openclaw_timeout"
        }

    output = result.stdout if result.stdout.strip() else result.stderr
    
    print("OPENCLAW OUTPUT:")
    print(output[:1000])
        

    if result.returncode != 0:
        return {
            "success": False,
            "response": "Erreur OpenClaw pendant la génération de la réponse.",
            "returncode": result.returncode,
            "output": output
        }

    try:
        data = json.loads(output)

        text = ""
        for payload in data.get("payloads", []):
            text += payload.get("text", "") + "\n"

        return {
            "success": True,
            "response": text.strip(),
            "raw": data
        }

    except Exception as e:
        print("OPENCLAW RAW OUTPUT:")
        print(output)

        return {
            "success": False,
            "response": "Erreur lors de la lecture de la réponse OpenClaw.",
            "error": str(e),
            "returncode": result.returncode,
            "output": output
        }
