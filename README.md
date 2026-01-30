# Roku

**Privacy-First AI Assistant for AR Glasses**

A locally-running AI assistant designed for Even G2 AR glasses. All processing happens on-device - no cloud, no data leaving your device.

## What It Does

- 🔒 **100% Local**: LLM inference runs entirely on-device using llama.cpp
- 🎭 **Personalized**: Hot-swappable LoRA adapters for different contexts (work, home, health)
- 🎙️ **Voice-First**: Whisper-based speech recognition for hands-free interaction
- 👓 **AR-Ready**: Optimized for glanceable 488px display

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                      Interfaces                          │
│    CLI  │  Voice  │  G2 Emulator  │  (Future: G2 SDK)   │
├─────────────────────────────────────────────────────────┤
│                        Core                              │
│   LLM (llama.cpp)  │  Router  │  Context  │  Voice      │
├─────────────────────────────────────────────────────────┤
│                      Adapters                            │
│  Personality │ Work │ Home │ Health │ Personal          │
├─────────────────────────────────────────────────────────┤
│                    Local Storage                         │
│     ChromaDB  │  Conversations  │  User Preferences     │
└─────────────────────────────────────────────────────────┘
```

## Tech Stack

| Component | Technology |
|-----------|------------|
| Base Model | Llama 3.2 3B Instruct (Q4_K_M quantized, ~2GB) |
| Inference | llama-cpp-python with Metal GPU acceleration |
| Personalization | LoRA adapters (~50MB each, hot-swappable) |
| Voice | OpenAI Whisper (base model, local) |
| Vector Store | ChromaDB (local embeddings) |

## Quick Start

```bash
# 1. Clone and setup
git clone https://github.com/YOUR_USERNAME/roku-ai.git
cd roku-ai

# 2. Create virtual environment
python3 -m venv ~/roku-env
source ~/roku-env/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download base model (~2GB)
huggingface-cli download bartowski/Llama-3.2-3B-Instruct-GGUF \
  Llama-3.2-3B-Instruct-Q4_K_M.gguf \
  --local-dir models/base/

# 5. Run CLI
python interfaces/cli.py

# 6. Run with voice
python interfaces/cli.py --voice

# 7. Run G2 emulator (mockup)
python interfaces/g2_emulator.py --llm
```

## Project Structure

```
roku-ai/
├── core/
│   ├── llm.py          # LLM wrapper with LoRA support
│   ├── voice.py        # Whisper STT integration
│   ├── context.py      # ChromaDB conversation memory
│   └── router.py       # Query routing (local vs tools)
├── adapters/
│   └── manager.py      # LoRA adapter hot-swapping
├── interfaces/
│   ├── cli.py          # Command-line interface
│   └── g2_emulator.py  # Even G2 display mockup (488px)
├── training/
│   ├── train.py        # LoRA fine-tuning script
│   └── training_data.py # Personality training examples
├── models/
│   ├── base/           # Base GGUF model (gitignored)
│   └── adapters/       # Trained LoRA adapters
└── security/
    └── encryption.py   # Local data encryption
```

## LoRA Adapters

The system uses hot-swappable LoRA adapters for context-aware personalization:

| Adapter | Size | Purpose |
|---------|------|---------|
| `personality.gguf` | ~50MB | Core conversational style |
| `work.gguf` | ~50MB | Professional communication (planned) |
| `home.gguf` | ~50MB | Smart home integration (planned) |
| `health.gguf` | ~50MB | Fitness/wellness tracking (planned) |

Benefits for mobile/AR:
- Ship 2GB base model once
- OTA update adapters (~50MB) without re-downloading base
- Switch contexts by loading different adapter (no full model reload)

## Training Your Own Adapter

```bash
# 1. Edit training data
vim training/training_data.py

# 2. Train (requires HuggingFace login for Llama 3.2 access)
huggingface-cli login
python training/train.py

# 3. Convert to GGUF (clone llama.cpp first)
git clone --depth 1 https://github.com/ggerganov/llama.cpp.git tools/llama-cpp
python tools/llama-cpp/convert_lora_to_gguf.py \
  models/adapters/personality_lora \
  --outfile models/adapters/personality.gguf \
  --outtype f16 \
  --base ~/.cache/huggingface/hub/models--meta-llama--Llama-3.2-3B-Instruct/snapshots/*/
```

## Hardware Requirements

| Environment | RAM | Storage | GPU |
|-------------|-----|---------|-----|
| Development (Mac) | 16GB | 20GB | Apple Silicon (M1+) |
| Mobile (planned) | 8GB | 4GB | Neural Engine |

## Roadmap

- [x] Local LLM inference with llama.cpp
- [x] LoRA adapter training and hot-swapping
- [x] Voice input with Whisper
- [x] G2 display emulator
- [ ] Even G2 SDK integration
- [ ] Domain-specific adapters (work, home, health)
- [ ] Proactive notifications
- [ ] Smart home integrations

## Privacy

- **Zero cloud dependency**: All inference runs locally
- **No telemetry**: Your conversations never leave your device
- **Encrypted storage**: Local data encrypted at rest
- **Open source**: Audit the code yourself

## License

MIT

---

*Named after Avatar Roku from Avatar: The Last Airbender - a wise guide who appears when needed.*
