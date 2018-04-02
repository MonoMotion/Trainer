import discord
import os
from pathlib import Path
import subprocess

client = discord.Client()

if 'DEEPL2_DISCORD_CHANNEL' in os.environ:
    target_channel = client.get_channel(os.environ['DEEPL2_DISCORD_CHANNEL'])
else:
    raise RuntimeError('Please supply channel.')

monitor_dir = Path('./monitor')

@client.event
async def on_ready():
    print('Logged in as')
    print(client.user.name)
    print(client.user.id)
    print('------')

@client.event
async def on_message(message):
    async def send_state():
        state = subprocess.check_output(['tail', '-1', monitor_dir.joinpath('log.csv')]).decode().rstrip()
        await client.send_message(message.channel, state)
    if message.content.startswith("!progress_video"):
        if client.user != message.author and message.channel == target_channel:
            videos = list(monitor_dir.glob("*.mp4"))
            videos.sort(key=lambda x: x.stat().st_mtime)
            await client.send_file(message.channel, videos[0])
            await send_state()
    elif message.content.startswith("!progress"):
        if client.user != message.author and message.channel == target_channel:
            await send_state()

async def report_to_discord(message):
    client.send_message(target_channel, message)

if 'DEEPL2_DISCORD_TOKEN' in os.environ:
    client.run(os.environ['DEEPL2_DISCORD_TOKEN'])
else:
    raise RuntimeError('Please supply token')
