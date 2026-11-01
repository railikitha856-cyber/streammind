"""Background TwitchIO EventSub connection for StreamMind."""
import asyncio
import os
import queue
import re
import threading
from datetime import datetime, timezone

from dotenv import load_dotenv

load_dotenv(encoding="utf-8-sig")
CHANNEL_PATTERN = re.compile(r"^[a-zA-Z0-9_]{4,25}$")


class TwitchChatManager:
    """Runs TwitchIO on its own asyncio loop so Streamlit stays responsive."""

    def __init__(self):
        self.messages = queue.Queue(maxsize=500)
        self._lock = threading.Lock()
        self._loop = None
        self._thread = None
        self._client = None
        self._start_task = None
        self._target_channel = ""
        self._target_broadcaster_id = ""
        self._subscribed_user_ids = set()
        self._status = "Disconnected"
        self._running = False
        self._configured = False

    @property
    def status(self):
        with self._lock:
            return self._status

    @property
    def running(self):
        with self._lock:
            return self._running

    def _set_status(self, status, running=None):
        with self._lock:
            self._status = status
            if running is not None:
                self._running = running

    def start(self, channel_name):
        """Start TwitchIO and prepare a subscription for a valid channel."""
        channel_name = (channel_name or "").strip().lstrip("#")
        if not CHANNEL_PATTERN.fullmatch(channel_name):
            return "Enter a valid Twitch channel name (4–25 letters, numbers, or underscores)."
        client_id = os.getenv("TWITCH_CLIENT_ID", "").strip()
        client_secret = os.getenv("TWITCH_CLIENT_SECRET", "").strip()
        if not client_id or not client_secret:
            return "Add TWITCH_CLIENT_ID and TWITCH_CLIENT_SECRET to your local .env file first."
        if self.running:
            if self._target_channel.lower() == channel_name.lower():
                return ""
            self.stop()
        elif self._thread and self._thread.is_alive():
            self.stop()

        self._target_channel = channel_name.lower()
        self._target_broadcaster_id = ""
        self._subscribed_user_ids.clear()
        self._set_status("Starting Twitch connection…", running=True)
        self._loop_ready.clear()
        self._thread = threading.Thread(
            target=self._run_event_loop,
            args=(client_id, client_secret),
            name="streammind-twitchio",
            daemon=True,
        )
        self._thread.start()
        if not self._loop_ready.wait(timeout=5):
            self._set_status("Could not start the Twitch background connection.", running=False)
            return self.status

        future = asyncio.run_coroutine_threadsafe(self._prepare_channel(), self._loop)
        try:
            future.result(timeout=40)
        except Exception as error:
            message = self._friendly_error(error)
            self._set_status(message, running=False)
            self.stop()
            return message
        return ""

    @property
    def _loop_ready(self):
        if not hasattr(self, "_loop_ready_event"):
            self._loop_ready_event = threading.Event()
        return self._loop_ready_event

    def _run_event_loop(self, client_id, client_secret):
        try:
            import twitchio
            from twitchio import eventsub

            class StreamMindTwitchClient(twitchio.Client):
                def __init__(self, manager):
                    super().__init__(client_id=client_id, client_secret=client_secret)
                    self.manager = manager

                async def event_oauth_authorized(self, payload):
                    # TwitchIO keeps and refreshes this token pair locally.
                    await self.add_token(payload.access_token, payload.refresh_token)
                    await self.save_tokens()
                    await self.manager._subscribe_for_user(payload.user_id)

                async def event_message(self, payload):
                    channel = (getattr(payload.broadcaster, "name", "") or "").lower()
                    if channel != self.manager._target_channel:
                        return
                    sent_at = payload.timestamp or datetime.now(timezone.utc)
                    if sent_at.tzinfo is None:
                        sent_at = sent_at.replace(tzinfo=timezone.utc)
                    row = {
                        "time": sent_at.astimezone().strftime("%H:%M:%S"),
                        "received_at": datetime.now(timezone.utc).isoformat(),
                        "user": getattr(payload.chatter, "name", "viewer"),
                        "avatar": "💬",
                        "message": payload.text,
                    }
                    try:
                        self.manager.messages.put_nowait(row)
                    except queue.Full:
                        # Keep the newest chat by dropping one oldest queued item.
                        try:
                            self.manager.messages.get_nowait()
                            self.manager.messages.put_nowait(row)
                        except queue.Empty:
                            pass

                async def event_error(self, payload):
                    error = getattr(payload, "exception", None)
                    self.manager._set_status(
                        self.manager._friendly_error(error or "Twitch reported a connection error.")
                    )

                async def event_subscription_revoked(self, payload):
                    self.manager._set_status(
                        "Twitch revoked the chat subscription. Disconnect and reconnect to authorize again."
                    )

            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            self._client = StreamMindTwitchClient(self)
            self._start_task = self._loop.create_task(self._client.start())
            self._start_task.add_done_callback(self._on_client_stopped)
            self._loop_ready.set()
            self._loop.run_forever()
        except Exception as error:
            self._set_status(self._friendly_error(error), running=False)
        finally:
            if self._loop and not self._loop.is_closed():
                pending = asyncio.all_tasks(self._loop)
                for task in pending:
                    task.cancel()
                if pending:
                    self._loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
                self._loop.close()
            if self.status in {"Starting Twitch connection…", "Connected to Twitch chat"}:
                self._set_status("Disconnected", running=False)

    def _on_client_stopped(self, task):
        if task.cancelled():
            return
        error = task.exception()
        if error:
            self._set_status(self._friendly_error(error), running=False)
        else:
            self._set_status("Twitch connection closed.", running=False)
        if self._loop and self._loop.is_running():
            self._loop.call_soon(self._loop.stop)

    async def _prepare_channel(self):
        await asyncio.wait_for(self._client.wait_until_ready(), timeout=30)
        channel = await self._client.fetch_user(login=self._target_channel)
        if channel is None:
            raise ValueError("That Twitch channel was not found. Check the channel name and try again.")
        self._target_broadcaster_id = str(channel.id)

        # A previously authorized Twitch user token is saved by TwitchIO.
        user_ids = list(self._client.tokens.keys())
        if user_ids:
            await self._subscribe_for_user(str(user_ids[0]))
        else:
            self._set_status(
                f"Channel @{self._target_channel} found. Authorize Twitch chat access to continue."
            )

    async def _subscribe_for_user(self, user_id):
        if not self._target_broadcaster_id or not user_id:
            return
        if user_id in self._subscribed_user_ids:
            return
        from twitchio import eventsub

        await self._client.subscribe_websocket(
            eventsub.ChatMessageSubscription(
                broadcaster_user_id=self._target_broadcaster_id,
                user_id=str(user_id),
            ),
            token_for=str(user_id),
        )
        self._subscribed_user_ids.add(str(user_id))
        self._set_status(f"Connected to @{self._target_channel} chat", running=True)

    def drain_messages(self):
        """Move newly received messages into Streamlit's session-state buffer."""
        rows = []
        while True:
            try:
                rows.append(self.messages.get_nowait())
            except queue.Empty:
                break
        return rows

    @staticmethod
    def _friendly_error(error):
        message = str(error)
        lowered = message.lower()
        if "invalid oauth" in lowered or "401" in lowered or "unauthorized" in lowered:
            return "Twitch authorization failed. Check your app credentials and authorize the chat reader again."
        if "403" in lowered or "forbidden" in lowered or "scope" in lowered:
            return "Twitch denied chat access. Reauthorize and grant the user:read:chat permission."
        if "port" in lowered or "address already in use" in lowered:
            return "Twitch's local OAuth listener could not start. Close another app using port 4343 and retry."
        return f"Twitch connection error: {message[:220]}"

    def stop(self):
        """Close the TwitchIO client and its background event loop safely."""
        loop = self._loop
        client = self._client
        if loop and loop.is_running() and client:
            try:
                asyncio.run_coroutine_threadsafe(client.close(), loop).result(timeout=10)
            except Exception:
                pass
            loop.call_soon_threadsafe(loop.stop)
        thread = self._thread
        if thread and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=5)
        self._set_status("Disconnected", running=False)
        self._client = None
        self._loop = None
        self._thread = None
        self._start_task = None
        self._target_broadcaster_id = ""
        self._subscribed_user_ids.clear()
