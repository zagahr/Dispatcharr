from datetime import timedelta
from unittest.mock import patch

from django.core.exceptions import ValidationError
from django.test import TestCase
from django.utils import timezone

from core.models import CoreSettings, RECORDING_PROTECTION_KEY

from apps.channels.models import Channel, Recording
from apps.channels.recording_lock_manager import RecordingLockManager
from apps.channels.recording_locks import get_active_lock
from apps.channels.models import LockStatus


class RecordingLockManagerTests(TestCase):
    def setUp(self):
        super().setUp()
        CoreSettings.objects.update_or_create(
            key=RECORDING_PROTECTION_KEY,
            defaults={
                "name": "Recording Protection Settings",
                "value": {
                    "enabled": True,
                    "max_concurrent_streams": 0,
                    "reserved_for_recording": 0,
                    "allow_override": True,
                    "override_requires_confirmation": False,
                    "override_confirmation_text": "Override recording protection?",
                    "notify_on_block": False,
                    "notify_on_override": False,
                    "lock_buffer_minutes": 0,
                },
            },
        )

    @patch("core.utils.RedisClient.get_client", return_value=None)
    def test_engage_and_check_stream_allowed(self, _redis_client):
        channel = Channel.objects.create(channel_number=1, name="Record Channel")
        recording = Recording.objects.create(
            channel=channel,
            start_time=timezone.now(),
            end_time=timezone.now() + timedelta(hours=1),
        )

        manager = RecordingLockManager()
        lock = manager.engage(recording)
        self.assertIsNotNone(lock)
        self.assertTrue(manager.is_locked())

        status = manager.get_status()
        self.assertTrue(status["locked"])
        self.assertEqual(status["status"], LockStatus.ACTIVE)

        allowed_same = manager.check_stream_allowed(channel.id, user_id=1)
        self.assertTrue(allowed_same["allowed"])
        self.assertEqual(allowed_same["reason"], "recording_channel")

        other_channel = Channel.objects.create(channel_number=2, name="Other")
        blocked = manager.check_stream_allowed(other_channel.id, user_id=1)
        self.assertFalse(blocked["allowed"])
        self.assertEqual(blocked["reason"], "recording_lock_active")

    @patch("core.utils.RedisClient.get_client", return_value=None)
    def test_release_without_cooldown(self, _redis_client):
        channel = Channel.objects.create(channel_number=3, name="Release Channel")
        recording = Recording.objects.create(
            channel=channel,
            start_time=timezone.now(),
            end_time=timezone.now() + timedelta(hours=1),
        )

        manager = RecordingLockManager()
        manager.engage(recording)
        released = manager.release(recording.id)
        self.assertIsNotNone(released)
        self.assertEqual(released.status, LockStatus.RELEASED)
        self.assertFalse(manager.is_locked())

    @patch("core.utils.RedisClient.get_client", return_value=None)
    def test_release_with_cooldown(self, _redis_client):
        CoreSettings.objects.update_or_create(
            key=RECORDING_PROTECTION_KEY,
            defaults={
                "name": "Recording Protection Settings",
                "value": {
                    "enabled": True,
                    "max_concurrent_streams": 0,
                    "reserved_for_recording": 0,
                    "allow_override": True,
                    "override_requires_confirmation": False,
                    "override_confirmation_text": "Override recording protection?",
                    "notify_on_block": False,
                    "notify_on_override": False,
                    "lock_buffer_minutes": 5,
                },
            },
        )

        channel = Channel.objects.create(channel_number=4, name="Cooldown Channel")
        recording = Recording.objects.create(
            channel=channel,
            start_time=timezone.now(),
            end_time=timezone.now() + timedelta(hours=1),
        )

        manager = RecordingLockManager()
        manager.engage(recording)
        lock = manager.release(recording.id)
        self.assertIsNotNone(lock)
        self.assertEqual(lock.status, LockStatus.COOLDOWN)
        self.assertTrue(manager.is_locked())

        lock.refresh_from_db()
        lock.cooldown_until = timezone.now() - timedelta(minutes=1)
        lock.save(update_fields=["cooldown_until"])

        self.assertFalse(manager.is_locked())
        active = get_active_lock()
        self.assertIsNone(active)

    @patch("core.utils.RedisClient.get_client", return_value=None)
    def test_override_invalid_confirmation(self, _redis_client):
        CoreSettings.objects.update_or_create(
            key=RECORDING_PROTECTION_KEY,
            defaults={
                "name": "Recording Protection Settings",
                "value": {
                    "enabled": True,
                    "max_concurrent_streams": 0,
                    "reserved_for_recording": 0,
                    "allow_override": True,
                    "override_requires_confirmation": True,
                    "override_confirmation_text": "Override recording protection?",
                    "notify_on_block": False,
                    "notify_on_override": False,
                    "lock_buffer_minutes": 0,
                },
            },
        )
        channel = Channel.objects.create(channel_number=5, name="Override Channel")
        recording = Recording.objects.create(
            channel=channel,
            start_time=timezone.now(),
            end_time=timezone.now() + timedelta(hours=1),
        )

        manager = RecordingLockManager()
        manager.engage(recording)

        with self.assertRaises(ValidationError) as context:
            manager.override(confirmation_text="nope")
        self.assertEqual(context.exception.messages[0], "invalid_confirmation")

    def test_override_token_generation(self):
        channel = Channel.objects.create(channel_number=6, name="Override Token Channel")
        recording = Recording.objects.create(
            channel=channel,
            start_time=timezone.now(),
            end_time=timezone.now() + timedelta(hours=1),
        )

        manager = RecordingLockManager()

        with patch("core.utils.RedisClient.get_client", return_value=None):
            manager.engage(recording)

        class FakeRedis:
            def __init__(self):
                self.calls = []

            def setex(self, key, ttl, value):
                self.calls.append((key, ttl, value))
                return True

        fake_redis = FakeRedis()

        with patch("core.utils.RedisClient.get_client", return_value=fake_redis):
            result = manager.override(confirmation_text="Override recording protection?", valid_for_seconds=120)

        self.assertTrue(result["token"])
        self.assertEqual(len(result["interrupted_recordings"]), 1)
        self.assertEqual(fake_redis.calls[0][1], 120)
