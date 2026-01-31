from dataclasses import asdict, dataclass

from django.core.exceptions import ValidationError

DEFAULT_RECORDING_PROTECTION_SETTINGS = {
    "enabled": False,
    "max_concurrent_streams": 0,
    "reserved_for_recording": 0,
    "allow_override": True,
    "override_requires_confirmation": False,
    "override_confirmation_text": "Override recording protection?",
    "notify_on_block": False,
    "notify_on_override": False,
    "lock_buffer_minutes": 0,
}


def get_recording_protection_defaults():
    return DEFAULT_RECORDING_PROTECTION_SETTINGS.copy()


def _validate_bool(field_name, value, errors):
    if not isinstance(value, bool):
        errors[field_name] = "must be a boolean"


def _validate_int(field_name, value, errors):
    if isinstance(value, bool) or not isinstance(value, int):
        errors[field_name] = "must be an integer"
    elif value < 0:
        errors[field_name] = "must be non-negative"


def normalize_recording_protection_settings(settings):
    if settings is None:
        settings = {}
    if not isinstance(settings, dict):
        raise ValidationError("Recording protection settings must be a dictionary.")

    merged = {**get_recording_protection_defaults(), **settings}
    errors = {}

    _validate_bool("enabled", merged.get("enabled"), errors)
    _validate_int("max_concurrent_streams", merged.get("max_concurrent_streams"), errors)
    _validate_int("reserved_for_recording", merged.get("reserved_for_recording"), errors)
    _validate_bool("allow_override", merged.get("allow_override"), errors)
    _validate_bool("override_requires_confirmation", merged.get("override_requires_confirmation"), errors)
    _validate_bool("notify_on_block", merged.get("notify_on_block"), errors)
    _validate_bool("notify_on_override", merged.get("notify_on_override"), errors)
    _validate_int("lock_buffer_minutes", merged.get("lock_buffer_minutes"), errors)

    if not isinstance(merged.get("override_confirmation_text"), str):
        errors["override_confirmation_text"] = "must be a string"

    max_streams = merged.get("max_concurrent_streams")
    reserved = merged.get("reserved_for_recording")
    if isinstance(max_streams, int) and isinstance(reserved, int) and max_streams != 0:
        if reserved > max_streams:
            errors["reserved_for_recording"] = "must be <= max_concurrent_streams"

    if merged.get("override_requires_confirmation"):
        if not merged.get("allow_override"):
            errors["override_requires_confirmation"] = "requires allow_override to be true"
        if not merged.get("override_confirmation_text", "").strip():
            errors["override_confirmation_text"] = (
                "must be provided when override_requires_confirmation is enabled"
            )

    if errors:
        raise ValidationError({"recording_protection": errors})

    return merged


def confirmation_matches(input_text, expected_text, policy="exact"):
    if policy != "exact":
        raise ValidationError("Unsupported confirmation policy.")
    return (input_text or "") == (expected_text or "")


@dataclass(frozen=True)
class RecordingProtectionConfig:
    enabled: bool
    max_concurrent_streams: int
    reserved_for_recording: int
    allow_override: bool
    override_requires_confirmation: bool
    override_confirmation_text: str
    notify_on_block: bool
    notify_on_override: bool
    lock_buffer_minutes: int

    @classmethod
    def load(cls, settings):
        normalized = normalize_recording_protection_settings(settings)
        return cls(**normalized)

    def to_dict(self):
        return asdict(self)
