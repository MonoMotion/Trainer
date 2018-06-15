import asyncio
from threading import Thread
import discord
import os
import sys
from pathlib import Path
import subprocess

class DiscordReporter(object):
    def __init__(self):
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
            for message in self.report_queue:
                await self.client.send_message(self.target_channel, message)

        if 'DEEPL2_DISCORD_TOKEN' in os.environ:
            token = os.environ['DEEPL2_DISCORD_TOKEN']
        else:
            raise RuntimeError('Please supply token')

        self.prefix = os.environ["DEEPL2_BRANCH_NAME"] + " " + os.environ["DEEPL2_COMMIT_ID"][:6] + ": "
        def t():
            asyncio.set_event_loop(asyncio.new_event_loop())
            self.client.run(token)
        self.thread = Thread(target=t)
        self.thread.start()

    def exit(self):
        asyncio.run_coroutine_threadsafe(self.client.logout(), self.client.loop)

    def report(self, message):
        if not self.ready:
            self.report_queue.append(message)
        else:
            asyncio.run_coroutine_threadsafe(self.client.send_message(self.target_channel, self.prefix+message), self.client.loop)

class DiscordProgressResponder(object):
    def __init__(self, monitor_dir=Path('./monitor')):
        self.monitor_dir = monitor_dir
        self.client = discord.Client()

    def start(self, pidstr):
        @self.client.event
        async def on_ready():
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
                prefix = os.environ["DEEPL2_BRANCH_NAME"] + " " + os.environ["DEEPL2_COMMIT_ID"][:6] + ": "
                state = subprocess.check_output(['tail', '-1', str(self.monitor_dir.joinpath('log.csv'))]).decode().rstrip()
                await self.client.send_message(message.channel, prefix+state)
            if message.content.startswith("!progress_video"):
                if self.client.user != message.author and message.channel == self.target_channel:
                    videos = list(self.monitor_dir.glob("*.mp4"))
                    videos.sort(key=lambda x: x.stat().st_mtime)
                    await self.client.send_file(message.channel, str(videos[-1]))
                    await send_state()
            elif message.content.startswith("!progress"):
                if self.client.user != message.author and message.channel == self.target_channel:
                    await send_state()
            elif message.content.startswith("!terminate"):
                subprocess.call(['kill', '-9', pidstr])
                sys.exit()

        if 'DEEPL2_DISCORD_TOKEN' in os.environ:
            token = os.environ['DEEPL2_DISCORD_TOKEN']
        else:
            raise RuntimeError('Please supply token')

        self.client.run(token)

if __name__ == '__main__':
    p = DiscordProgressResponder()
    p.start(sys.argv[1])
