import asyncio
from threading import Thread
import discord
import os
from pathlib import Path
import subprocess

class DiscordReporter(object):
    def __init__(self, monitor_dir=Path('./monitor')):
        self.monitor_dir = monitor_dir
        self.client = discord.Client()
        self.ready = False
        self.report_queue = []

    def start(self):
        @self.client.event
        async def on_ready():
            self.ready = True
            print('Logged in as')
            print(self.client.user.name)
            print(self.client.user.id)
            print('------')
            if 'DEEPL2_DISCORD_CHANNEL' in os.environ:
                self.target_channel = self.client.get_channel(os.environ['DEEPL2_DISCORD_CHANNEL'])
                if not self.target_channel:
                    raise RuntimeError('Cannot get channel')
            else:
                raise RuntimeError('Please supply channel.')

        @self.client.event
        async def on_message(message):
            async def send_state():
                state = subprocess.check_output(['tail', '-1', str(self.monitor_dir.joinpath('log.csv'))]).decode().rstrip()
                await self.client.send_message(message.channel, state)
            if message.content.startswith("!progress_video"):
                if self.client.user != message.author and message.channel == self.target_channel:
                    videos = list(self.monitor_dir.glob("*.mp4"))
                    videos.sort(key=lambda x: x.stat().st_mtime)
                    await self.client.send_file(message.channel, str(videos[0]))
                    await send_state()
            elif message.content.startswith("!progress"):
                if self.client.user != message.author and message.channel == self.target_channel:
                    await send_state()

        if 'DEEPL2_DISCORD_TOKEN' in os.environ:
            token = os.environ['DEEPL2_DISCORD_TOKEN']
        else:
            raise RuntimeError('Please supply token')

        def t():
            asyncio.set_event_loop(asyncio.new_event_loop())
            self.client.run(token)
        self.thread = Thread(target=t)
        self.thread.start()

    def report(self, message):
        def send(m):
            asyncio.run_coroutine_threadsafe(self.client.send_message(self.target_channel, message), self.client.loop)

        if not self.ready:
            self.report_queue.append(message)
        else:
            send(message)
            for m in self.report_queue:
                send(m)
