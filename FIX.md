Here's your copy-paste ready prompt:

---

**Fix the following FastAPI/pydantic-settings application. The app crashes on startup in a restart loop.**

**Root cause:** `pydantic_settings.sources.SettingsError: error parsing value for field "device" from source "EnvSettingsSource"` — pydantic-settings tries to JSON-decode the `DEVICE=cuda` env var because `DeviceSettings` is a nested `BaseSettings` inside `Settings`, which triggers complex-type JSON parsing on a plain string value.

**File to fix: `src/core/config.py`**

**Problems to fix:**

1. `DeviceSettings` has `env_prefix = ""` — this causes pydantic-settings to intercept `DEVICE` as a complex type and attempt `json.loads("cuda")`, which fails. Fix: Remove `DeviceSettings` as a `BaseSettings` subclass or use `model_config` with `env_nested_delimiter` correctly, OR read device/attention_slicing/cpu_offload as flat fields directly on the main `Settings` class and construct `DeviceSettings` manually.

2. The same potential issue exists for `APISettings`, `ModelSettings`, `GenerationSettings` — all nested `BaseSettings` with prefix env vars being double-parsed.

3. Docker restart loop — caused solely by the startup crash above. Once config loads cleanly, the restart loop stops.

**Fix strategy — replace nested `BaseSettings` with plain `BaseModel` for sub-configs, and read all env vars flat on the root `Settings` class:**

```python
# In Settings class, add flat env var fields:
device_device: str = Field(default="cuda", alias="DEVICE")
device_attention_slicing: bool = Field(default=True, alias="ATTENTION_SLICING")  
device_cpu_offload: bool = Field(default=False, alias="CPU_OFFLOAD")
api_host: str = Field(default="0.0.0.0", alias="API_HOST")
api_port: int = Field(default=8141, alias="API_PORT")
# etc.
```

Then construct `DeviceSettings`, `APISettings` etc. as plain `pydantic.BaseModel` (not `BaseSettings`) from those flat values inside `from_yaml()`, merging YAML + env vars manually.

**Also fix `DeviceSettings.validate_device` validator** — it must also accept `"auto"` since `docker-compose.yml` sets `DEVICE=${DEVICE:-cpu}` which could resolve to other values.

**Requirements:**
- Python 3.10
- pydantic==2.5.3
- pydantic-settings==2.1.0
- All existing field names, types, and method signatures must remain unchanged
- `get_settings()`, `reload_settings()`, `Settings.from_yaml()` signatures unchanged
- YAML config `config/config.yml` must still load and be overridable by env vars
- Docker compose env vars (`DEVICE`, `API_PORT`, `ATTENTION_SLICING`, `CPU_OFFLOAD`, `DEFAULT_STEPS`, `DEFAULT_GUIDANCE`, `DEFAULT_WIDTH`, `DEFAULT_HEIGHT`, `MODEL_DIR`, `DEFAULT_MODEL`, `SAFETY_CHECKER`, `HF_TOKEN`, `LOG_LEVEL`) must all work as plain string env vars without JSON parsing
- All existing tests in `tests/test_config.py` must still pass