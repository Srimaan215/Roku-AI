# Roku AI - Setup Status

**Date:** February 6, 2026  
**Status:** ✅ Ready for Daily Use

---

## ✅ Completed

### 1. **DeepSeek-R1 14B Downloaded**
- Model: `DeepSeek-R1-Distill-Qwen-14B-Q4_K_M.gguf`
- Size: 8.4GB (4-bit quantized)
- Location: `~/Roku/roku-ai/models/base/`
- **Upgrade from:** Llama 3.2 3B → DeepSeek-R1 14B (better reasoning)

### 2. **GUI Interface Created**
- File: `interfaces/roku_gui.py`
- Features:
  - Clean chat interface with dark theme
  - Real-time LoRA adapter monitoring
  - Performance metrics (latency, query count)
  - Interaction logging for research
  - Auto-save conversations

### 3. **LoRA Adapters Trained**
- ✅ **personality.gguf** (46MB) - Conversational style
- ✅ **personal.gguf** (50MB) - Your facts & preferences

### 4. **Profile System**
- User profile: `data/profiles/Srimaan.json`
- Contains your background, work, schedule, preferences
- Used for context injection (no hallucination)

---

## 🚀 How to Launch

### **Option 1: Using Launcher Script**
```bash
cd ~/Roku/roku-ai
./launch_roku_gui.sh
```

### **Option 2: Direct Python**
```bash
cd ~/Roku/roku-ai
python interfaces/roku_gui.py
```

---

## 📊 What to Track (For Research)

The GUI automatically logs to `data/conversations/`:
1. **Adapter usage** - which LoRAs activate for each query
2. **Latency** - response time per query
3. **Failure modes** - where adapters struggle
4. **Usage patterns** - what you actually ask daily

**Use this data for:** Implementing Thousand Brains voting layer

---

## 🎯 Next Steps

### **Week 1: Stress Test**
- [ ] Use Roku daily for all queries
- [ ] Log everything
- [ ] Identify limitations

### **Week 2: Analyze Data**
- [ ] Review conversation logs
- [ ] Find patterns where single adapters fail
- [ ] Identify need for voting mechanism

### **Week 3: Implement Voting Layer**
- [ ] Add column voting (Thousand Brains Theory)
- [ ] Test multi-adapter consensus
- [ ] Compare vs single adapter performance

### **Future:**
- [ ] Add more domain adapters (work, health, home)
- [ ] Vision model integration (Qwen-VL or similar)
- [ ] Overnight analysis pipeline
- [ ] Even G2 integration (when accepted)

---

## 🧠 Research Framework

### **Hypothesis:**
Multi-LoRA composition with Thousand Brains voting will outperform single large models for personalized tasks.

### **What You're Testing:**
1. **Composition > Scale** - 14B base + 50MB adapters vs 70B monolithic
2. **Specialization > Generalization** - Domain experts vs general knowledge
3. **Continuous Learning** - Nightly adapter updates vs static weights

### **Metrics to Track:**
- Response quality (your subjective rating)
- Adapter activation patterns
- Failure modes per domain
- Latency differences

---

## 🛠️ Troubleshooting

### **If GUI doesn't launch:**
```bash
# Check dependencies
cd ~/Roku/roku-ai
pip install -r requirements.txt

# Test model loading
python test_deepseek_simple.py
```

### **If model is slow:**
- DeepSeek-R1 14B is ~4x larger than Llama 3.2 3B
- First inference will be slow (loading to RAM/VRAM)
- Subsequent queries should be faster
- Consider switching back to Llama 3.2 3B if too slow

### **To switch models:**
Edit `core/multi_lora.py` line 37:
```python
# DeepSeek-R1 14B (better reasoning, slower)
DEFAULT_MODEL_PATH = Path.home() / "Roku/roku-ai/models/base/DeepSeek-R1-Distill-Qwen-14B-Q4_K_M.gguf"

# OR Llama 3.2 3B (faster, less capable)
DEFAULT_MODEL_PATH = Path.home() / "Roku/roku-ai/models/base/Llama-3.2-3B-Instruct-Q4_K_M.gguf"
```

---

## 📁 Project Structure

```
roku-ai/
├── core/                   # Core AI functionality
│   ├── multi_lora.py      # Multi-LoRA system ✨
│   ├── personalized_roku.py
│   └── context_manager.py # Profile & context injection
├── interfaces/
│   ├── roku_gui.py        # Daily driver GUI ✨
│   └── cli.py
├── models/
│   ├── base/              # LLM models
│   │   ├── DeepSeek-R1-Distill-Qwen-14B-Q4_K_M.gguf ✨ (8.4GB)
│   │   └── Llama-3.2-3B-Instruct-Q4_K_M.gguf (1.9GB)
│   └── adapters/          # LoRA adapters
│       ├── personality.gguf ✨ (46MB)
│       └── personal.gguf ✨ (50MB)
├── data/
│   ├── profiles/          # User profiles
│   │   └── Srimaan.json ✨
│   └── conversations/     # Auto-saved logs
└── training/              # Adapter training pipeline
```

---

## 💡 Tips for Daily Use

1. **Ask varied questions** - test different domains
2. **Note failures** - when Roku doesn't know something
3. **Check the logs** - see which adapters activated
4. **Rate responses** - mental note of quality
5. **Save interesting conversations** - for training data

---

**Ready to start!** Launch the GUI and begin your Thousand Brains research! 🧠
