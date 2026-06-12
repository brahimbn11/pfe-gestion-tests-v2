import os
import discord
import aiohttp
from dotenv import load_dotenv

load_dotenv()

TOKEN = os.getenv("DISCORD_TOKEN")

API_URL = "http://127.0.0.1:8000/assistant/chat"

intents = discord.Intents.default()
intents.message_content = True

client = discord.Client(intents=intents)


@client.event
async def on_ready():
    print(f"Bot connecte: {client.user}")


@client.event
async def on_message(message):
    if message.author == client.user:
        return

    query = message.content.strip()

    if not query:
        return

    await message.channel.send("Analyse en cours...")

    payload = {
        "user_id": str(message.author.id),
        "message": query
    }

    try:
        timeout = aiohttp.ClientTimeout(total=240)

        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(API_URL, json=payload) as response:
                if response.status != 200:
                    error_text = await response.text()
                    await message.channel.send(
                        f"Erreur backend {response.status}:\n{error_text[:1000]}"
                    )
                    return

                try:
                    data = await response.json()
                except Exception:
                    raw_text = await response.text()
                    await message.channel.send(
                        f"Erreur backend: réponse non JSON.\n{raw_text[:1000]}"
                    )
                    return

        reply = data.get("response", "")

        if not reply:
            reply = "Erreur assistant: " + str(data)

        chunks = [reply[i:i + 1900] for i in range(0, len(reply), 1900)]

        for chunk in chunks:
            await message.channel.send(chunk)

    except Exception as e:
        await message.channel.send(f"Erreur lors de l'analyse: {str(e)}")

print("Token loaded:", TOKEN is not None)
print("Token length:", len(TOKEN) if TOKEN else 0)

client.run(TOKEN)
