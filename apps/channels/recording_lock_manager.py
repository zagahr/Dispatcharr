from datetime import timedelta

from django.utils import timezone

from django.core.exceptions import ValidationError

from core.models import CoreSettings
from core.recording_protection import confirmation_matches

from .models import LockStatus
from .recording_locks import (
    append_history_event,
    create_lock,
    create_override_token,
    get_active_lock,
    release_lock,
    set_cooldown,
)


class RecordingLockManager:
    def __init__(self, config=None):
        self.config = config or CoreSettings.get_recording_protection_settings()

    def engage(self, recording):
        if not self.config.enabled:
            return None
        return create_lock(recording)

    def is_locked(self):
        if not self.config.enabled:
            return False
        return self._get_active_lock() is not None

    def get_status(self):
        lock = self._get_active_lock()
        if not lock:
            return {
                "locked": False,
                "status": None,
                "lock_id": None,
                "recording_id": None,
            }
        return {
            "locked": True,
            "status": lock.status,
            "lock_id": str(lock.id),
            "recording_id": str(lock.recording_id),
        }

    def check_stream_allowed(self, requested_channel_id, user_id=None):
        if not self.config.enabled:
            return {
                "allowed": True,
                "reason": "recording_protection_disabled",
                "lock_id": None,
                "status": None,
                "recording_id": None,
                "override_allowed": self.config.allow_override,
                "override_requires_confirmation": self.config.override_requires_confirmation,
                "override_confirmation_text": self.config.override_confirmation_text,
            }

        lock = self._get_active_lock()
        if not lock:
            return {
                "allowed": True,
                "reason": "no_active_lock",
                "lock_id": None,
                "status": None,
                "recording_id": None,
                "override_allowed": self.config.allow_override,
                "override_requires_confirmation": self.config.override_requires_confirmation,
                "override_confirmation_text": self.config.override_confirmation_text,
            }

        recording_channel_id = str(lock.recording.channel_id)
        requested_channel_id = str(requested_channel_id)
        if requested_channel_id == recording_channel_id:
            return {
                "allowed": True,
                "reason": "recording_channel",
                "lock_id": str(lock.id),
                "status": lock.status,
                "recording_id": str(lock.recording_id),
                "override_allowed": self.config.allow_override,
                "override_requires_confirmation": self.config.override_requires_confirmation,
                "override_confirmation_text": self.config.override_confirmation_text,
            }

        return {
            "allowed": False,
            "reason": "recording_lock_active",
            "lock_id": str(lock.id),
            "status": lock.status,
            "recording_id": str(lock.recording_id),
            "override_allowed": self.config.allow_override,
            "override_requires_confirmation": self.config.override_requires_confirmation,
            "override_confirmation_text": self.config.override_confirmation_text,
        }

    def release(self, recording_id):
        lock = get_active_lock()
        if not lock:
            return None
        if str(lock.recording_id) != str(recording_id):
            return None
        if self.config.lock_buffer_minutes > 0:
            until = timezone.now() + timedelta(minutes=self.config.lock_buffer_minutes)
            return set_cooldown(lock.id, until)
        return release_lock(lock_id=lock.id)

    def override(self, confirmation_text=None, valid_for_seconds=300):
        if not self.config.allow_override:
            raise ValidationError("override_not_allowed")
        if self.config.override_requires_confirmation:
            if not confirmation_matches(confirmation_text, self.config.override_confirmation_text):
                lock = self._get_active_lock()
                if lock:
                    append_history_event(
                        {
                            "event": "override_denied",
                            "lock_id": str(lock.id),
                            "recording_id": str(lock.recording_id),
                            "reason": "invalid_confirmation",
                        }
                    )
                raise ValidationError("invalid_confirmation")

        lock = self._get_active_lock()
        if not lock:
            return {"token": None, "interrupted_recordings": []}

        append_history_event(
            {
                "event": "override_requested",
                "lock_id": str(lock.id),
                "recording_id": str(lock.recording_id),
            }
        )

        token = create_override_token(lock, valid_for_seconds)
        recording = lock.recording
        interrupted = [
            {
                "recording_id": str(recording.id),
                "channel_id": str(recording.channel_id),
                "channel_name": recording.channel.name,
                "expected_end": recording.end_time.isoformat() if recording.end_time else None,
            }
        ]
        append_history_event(
            {
                "event": "override_granted",
                "lock_id": str(lock.id),
                "recording_id": str(lock.recording_id),
                "token": token,
            }
        )
        return {"token": token, "interrupted_recordings": interrupted}

    def _get_active_lock(self):
        lock = get_active_lock()
        if not lock:
            return None
        if lock.status == LockStatus.COOLDOWN:
            if lock.cooldown_until and lock.cooldown_until <= timezone.now():
                release_lock(lock_id=lock.id)
                return None
        return lock
