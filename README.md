# Project Vimaan ✈️  
**AI Co-Pilot for X-Plane**  

Project Vimaan is an experimental AI-powered first officer for the **X-Plane flight simulator**.  
It listens to your **voice commands**, parses them into structured actions (like gear up, flaps down, autopilot settings), and executes them inside the simulator.  

The long-term vision is to build a **fully offline, trainable AI co-pilot** that can assist in flight operations without depending on external LLM providers.  

---

## 🚀 Features (Work in Progress)
- 🎤 **Voice Recognition** → Convert spoken commands into text.  
- 🤖 **Intent Parsing** → Understand pilot intent (gear, flaps, autopilot, etc.).  
- 🛫 **X-Plane Integration** → Execute actions directly in the simulator.  
- 🔌 **Plugin Support** → Deployable as a packaged X-Plane plugin.  

---

## 📍 Roadmap
1. ✅ Voice input + basic command recognition (gear, flaps).  
2. ⏳ Expand vocabulary and intent mapping.  
3. ⏳ Train a small custom model for aviation commands.  
4. ⏳ Full X-Plane plugin with toggleable AI Co-Pilot.  
5. ⏳ Testing + packaging for release.  

---

## 🛠️ Getting Started

### Requirements
- Python 3.9+  
- [SpeechRecognition](https://pypi.org/project/SpeechRecognition/)  
- [PyAudio](https://people.csail.mit.edu/hubert/pyaudio/)  
- [Ollama (optional)](https://ollama.ai) for local LLM inference  

### Installation
```bash
git clone https://github.com/yourusername/project-vimaan.git
cd project-vimaan
pip install -r requirements.txt
