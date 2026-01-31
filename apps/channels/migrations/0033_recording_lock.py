from django.db import migrations, models
import django.db.models.deletion
import uuid


class Migration(migrations.Migration):

    dependencies = [
        ("channels", "0032_channel_is_adult_stream_is_adult"),
    ]

    operations = [
        migrations.CreateModel(
            name="RecordingLock",
            fields=[
                ("id", models.UUIDField(default=uuid.uuid4, editable=False, primary_key=True, serialize=False)),
                ("status", models.CharField(choices=[("active", "Active"), ("released", "Released"), ("cooldown", "Cooldown")], default="active", max_length=20)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("updated_at", models.DateTimeField(auto_now=True)),
                ("released_at", models.DateTimeField(blank=True, null=True)),
                ("cooldown_until", models.DateTimeField(blank=True, null=True)),
                ("history", models.JSONField(blank=True, default=list)),
                ("recording", models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name="locks", to="channels.recording")),
            ],
        ),
        migrations.AddConstraint(
            model_name="recordinglock",
            constraint=models.UniqueConstraint(condition=models.Q(("status", "active")), fields=("status",), name="unique_active_recording_lock"),
        ),
    ]
