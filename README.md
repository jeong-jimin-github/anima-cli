# Anima CLI

Anima-2.9B text-to-image generation tool. Pure CLI, designed for agentic AI workflows.

## Install

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Basic generation
python anima.py -p "1girl, masterpiece, cherry blossoms, spring"

# Custom settings
python anima.py -p "1boy, fantasy armor" -o hero.png --steps 40 --cfg 4.5 --seed 12345

# JSON output (for agentic AI parsing)
python anima.py -p "test prompt" --json

# Prompt from file
python anima.py --prompt-file prompt.txt -o output.png

# Batch generation
python anima.py --batch prompts.txt --output-dir ./output --json
```

## Arguments

| Flag | Short | Default | Description |
|------|-------|---------|-------------|
| `--prompt` | `-p` | - | Text prompt |
| `--prompt-file` | - | - | Read prompt from file |
| `--negative-prompt` | `-n` | - | Negative prompt |
| `--width` | `-W` | 812 | Image width |
| `--height` | `-H` | 1216 | Image height |
| `--steps` | `-s` | 30 | Inference steps (28-50) |
| `--cfg` | `-c` | 4.0 | CFG scale (3.5-5) |
| `--sampler` | - | euler | euler, euler_a, dpm++_2m, dpm++_sde, res_multistep, er_sde |
| `--scheduler` | - | sgm_uniform | sgm_uniform, beta, beta57, linear, linear_quadratic |
| `--seed` | - | random | Random seed |
| `--output` | `-o` | anima_{seed}.png | Output path |
| `--json` | - | false | JSON output |
| `--batch` | - | - | Batch prompts file |
| `--model-path` | - | Gazingstars123/Anima-2.9B | Local model path |
| `--device` | - | auto | auto, cuda, cpu, mps |
| `--dtype` | - | bf16 | fp16, bf16, fp32 |

## JSON Output Format

```json
{
  "success": true,
  "path": "/absolute/path/to/image.png",
  "prompt": "1girl, masterpiece...",
  "seed": 12345,
  "width": 812,
  "height": 1216,
  "steps": 30,
  "cfg": 4.0,
  "sampler": "euler",
  "scheduler": "sgm_uniform",
  "elapsed_seconds": 8.234
}
```

## Prompting Tips

Use Danbooru/Gelbooru tag format:
```
masterpiece, best quality, 1girl, solo, cherry blossoms, spring, detailed background, soft lighting, by makoto shinkai
```

- Quality tags: `masterpiece, best quality, highly detailed`
- Character count: `1girl, 1boy`
- Character names after series/copyright tags
- More detailed prompts produce better results

## License

Uses Anima-2.9B under CircleStone Labs Non-Commercial License.
