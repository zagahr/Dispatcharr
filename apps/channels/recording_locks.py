import json
import secrets

from django.core.exceptions import ValidationError
from django.db import IntegrityError, transaction
from django.utils import timezone

from core.utils import RedisClient

from .models import LockStatus, Recording, RecordingLock

ACTIVE_LOCK_KEY = "recording_lock:active"
LOCK_HISTORY_KEY = "recording_lock:history"
OVERRIDE_TOKEN_PREFIX = "recording_lock:override:"


def _serialize_lock(lock):
    return {
        "id": str(lock.id),
        "recording_id": str(lock.recording_id),
        "status": lock.status,
        "created_at": lock.created_at.isoformat() if lock.created_at else None,
        "updated_at": lock.updated_at.isoformat() if lock.updated_at else None,
        "released_at": lock.released_at.isoformat() if lock.released_at else None,
        "cooldown_until": lock.cooldown_until.isoformat() if lock.cooldown_until else None,
    }


def _write_active_lock(redis_client, lock):
    if not redis_client:
        return
    redis_client.set(ACTIVE_LOCK_KEY, json.dumps(_serialize_lock(lock)))


def _clear_active_lock(redis_client):
    if not redis_client:
        return
    redis_client.delete(ACTIVE_LOCK_KEY)


def _append_history(redis_client, event):
    if not redis_client:
        return
    redis_client.rpush(LOCK_HISTORY_KEY, json.dumps(event))


def create_lock(recording):
    if not isinstance(recording, Recording):
        raise ValidationError("Recording lock requires a Recording instance.")

    existing = get_active_lock()
    if existing:
        raise ValidationError("Another recording lock is already active.")

    try:
        with transaction.atomic():
            lock = RecordingLock.objects.create(
                recording=recording,
                status=LockStatus.ACTIVE,
            )
    except IntegrityError:
        raise ValidationError("Another recording lock is already active.")

    event = append_history_event(
        {
            "event": "lock_created",
            "lock_id": str(lock.id),
            "recording_id": str(recording.id),
        }
    )

    redis_client = RedisClient.get_client()
    _write_active_lock(redis_client, lock)
    _append_history(redis_client, event)
    return lock


def get_active_lock():
    redis_client = RedisClient.get_client()
    if redis_client:
        try:
            raw = redis_client.get(ACTIVE_LOCK_KEY)
            if raw:
                payload = json.loads(raw)
                lock_id = payload.get("id")
                if lock_id:
                    lock = RecordingLock.objects.filter(id=lock_id).first()
                    if lock and _lock_is_active(lock):
                        return lock
                    if lock and lock.status == LockStatus.COOLDOWN:
                        _clear_active_lock(redis_client)
        except Exception:
            pass

    lock = RecordingLock.objects.filter(status=LockStatus.ACTIVE).first()
    if not lock:
        lock = RecordingLock.objects.filter(status=LockStatus.COOLDOWN).order_by("-updated_at").first()
    if lock and _lock_is_active(lock):
        if redis_client:
            _write_active_lock(redis_client, lock)
        return lock
    if lock and redis_client:
        _clear_active_lock(redis_client)
    return None


def release_lock(lock_id=None, recording_id=None):
    if not lock_id and not recording_id:
        raise ValidationError("lock_id or recording_id is required to release a lock.")

    lock_qs = RecordingLock.objects.all()
    if lock_id:
        lock_qs = lock_qs.filter(id=lock_id)
    if recording_id:
        lock_qs = lock_qs.filter(recording_id=recording_id)

    lock = lock_qs.first()
    if not lock:
        return None

    lock.status = LockStatus.RELEASED
    lock.released_at = timezone.now()
    lock.save(update_fields=["status", "released_at", "updated_at"])

    event = append_history_event(
        {
            "event": "lock_released",
            "lock_id": str(lock.id),
            "recording_id": str(lock.recording_id),
        }
    )

    redis_client = RedisClient.get_client()
    _clear_active_lock(redis_client)
    _append_history(redis_client, event)
    return lock


def set_cooldown(lock_id, until):
    if not lock_id:
        raise ValidationError("lock_id is required to set cooldown.")
    if until is None:
        raise ValidationError("cooldown until time is required.")

    lock = RecordingLock.objects.filter(id=lock_id).first()
    if not lock:
        raise ValidationError("Recording lock not found.")

    lock.status = LockStatus.COOLDOWN
    lock.cooldown_until = until
    lock.save(update_fields=["status", "cooldown_until", "updated_at"])

    event = append_history_event(
        {
            "event": "lock_cooldown_set",
            "lock_id": str(lock.id),
            "recording_id": str(lock.recording_id),
            "cooldown_until": until.isoformat() if hasattr(until, "isoformat") else str(until),
        }
    )

    redis_client = RedisClient.get_client()
    _write_active_lock(redis_client, lock)
    _append_history(redis_client, event)
    return lock


def append_history_event(event):
    if not isinstance(event, dict):
        raise ValidationError("History event must be a dictionary.")
    event = {
        "timestamp": timezone.now().isoformat(),
        **event,
    }

    lock_id = event.get("lock_id")
    recording_id = event.get("recording_id")
    lock = None
    if lock_id:
        lock = RecordingLock.objects.filter(id=lock_id).first()
    if not lock and recording_id:
        lock = RecordingLock.objects.filter(recording_id=recording_id).first()

    if lock:
        lock.history = (lock.history or []) + [event]
        lock.save(update_fields=["history", "updated_at"])

    return event


def _lock_is_active(lock):
    if lock.status == LockStatus.ACTIVE:
        return True
    if lock.status != LockStatus.COOLDOWN:
        return False
    if not lock.cooldown_until:
        return True
    if lock.cooldown_until <= timezone.now():
        release_lock(lock_id=lock.id)
        return False
    return True


def create_override_token(lock, valid_for_seconds):
    if not lock:
        raise ValidationError("Recording lock is required for override token.")
    if not isinstance(valid_for_seconds, int) or valid_for_seconds <= 0:
        raise ValidationError("valid_for_seconds must be a positive integer.")

    redis_client = RedisClient.get_client()
    if not redis_client:
        raise ValidationError("Override token storage unavailable.")

    token = secrets.token_urlsafe(32)
    payload = {
        "lock_id": str(lock.id),
        "recording_id": str(lock.recording_id),
        "created_at": timezone.now().isoformat(),
        "valid_for_seconds": valid_for_seconds,
    }
    redis_client.setex(f"{OVERRIDE_TOKEN_PREFIX}{token}", valid_for_seconds, json.dumps(payload))
    return token
