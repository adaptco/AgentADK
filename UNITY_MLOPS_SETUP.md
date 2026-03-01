# Unity MLOps Setup

This repository includes a lightweight autonomous Unity training orchestration module: `mlops_unity_pipeline.py`.

## Quick start

```bash
pip install pyyaml croniter
python - <<'PY'
import asyncio
from mlops_unity_pipeline import UnityMLOpsOrchestrator, UnityAssetSpec, RLTrainingConfig, TrainingJob

async def main():
    orchestrator = UnityMLOpsOrchestrator()
    asset = UnityAssetSpec(
        asset_id="test-001",
        name="SimpleAgent",
        asset_type="behavior",
        description="Reach target position",
    )
    config = RLTrainingConfig(algorithm="PPO", max_steps=100_000)
    result = await orchestrator.execute_training_job(TrainingJob("test-job", asset, config))
    print(result)

asyncio.run(main())
PY
```

## Components

- `UnityAssetSpec`: Declares what behavior to generate.
- `RLTrainingConfig`: Declares training configuration.
- `UnityMLOpsOrchestrator`: Runs generation/build/train/register stages.
- `TrainingScheduler`: Handles recurring or continuous runs.

## Notes

- The current implementation is a scaffold with deterministic local artifacts.
- Replace `_generate_unity_code`, `_build_unity_environment`, `_train_agent`, and `_register_model` with production integrations for your environment (LLM provider, Unity headless build command, ML-Agents trainer command, and Vertex AI SDK).
