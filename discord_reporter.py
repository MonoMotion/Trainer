import asyncio
from threading import Thread
import discord
import os
import sys
from pathlib import Path
import subprocess
from plot import plot

class DiscordReporter(object):
    def __init__(self):
        self.client = discord.Client()
        self.ready = False
        self.report_queue = []
        self.prefix = os.environ["DEEPL2_BRANCH_NAME"] + " " + os.environ["DEEPL2_COMMIT_ID"][:6] + ": "

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
                await self.client.send_message(self.target_channel, self.prefix+message)

        if 'DEEPL2_DISCORD_TOKEN' in os.environ:
            token = os.environ['DEEPL2_DISCORD_TOKEN']
        else:
            raise RuntimeError('Please supply token')

        def t():
            asyncio.set_event_loop(asyncio.new_event_loop())
            self.client.run(token)
        self.thread = Thread(target=t)
        self.thread.start()

    def exit(self):
        asyncio.run_coroutine_threadsafe(self.client.logout(), self.client.loop)

    def send_file(self, path):
        if not self.ready:
            raise RuntimeError("Attempt to send file before reporter become ready")
        else:
            asyncio.run_coroutine_threadsafe(self.client.send_file(self.target_channel, path), self.client.loop)

    def report(self, message):
        if not self.ready:
            self.report_queue.append(message)
        else:
            asyncio.run_coroutine_threadsafe(self.client.send_message(self.target_channel, self.prefix+message), self.client.loop)

class DiscordProgressResponder(object):
    def __init__(self, monitor_dir=Path('./monitor')):
        self.monitor_dir = monitor_dir
        self.client = discord.Client()
        self.prefix = os.environ["DEEPL2_BRANCH_NAME"] + " " + os.environ["DEEPL2_COMMIT_ID"][:6] + ": "

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
            if self.client.user == message.author or message.channel != self.target_channel:
                return
            if len(message.content.split()) > 1 and not os.environ["DEEPL2_COMMIT_ID"].startswith(message.content.split()[1]):
                return

            async def send_state():
                state = subprocess.check_output(['tail', '-1', str(self.monitor_dir.joinpath('log.csv'))]).decode().rstrip()
                await self.client.send_message(message.channel, self.prefix+state)
            if message.content.startswith("!progress_video"):
                videos = list(self.monitor_dir.glob("*.mp4"))
                videos.sort(key=lambda x: x.stat().st_mtime)
                await self.client.send_file(message.channel, str(videos[-1]))
                await send_state()
            elif message.content.startswith("!progress"):
                await send_state()
            elif message.content.startswith("!plot"):
                plot(str(self.monitor_dir.glob("*.csv").__next__()), "reward_sum", "final_distance", 10, 100, title=self.prefix+"Plot", filename="plot.png")
                await self.client.send_file(message.channel, "./plot.png")
            elif message.content.startswith("!terminate"):
                await self.client.send_message(message.channel, self.prefix + "Terminating...")
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
