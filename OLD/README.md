# AI Copilot for X-Plane

This project implements an intelligent copilot system for the X-Plane flight simulator. The copilot listens to spoken commands, converts them into text, understands the intent using Natural Language Processing, and maps the interpreted actions to the appropriate X-Plane datarefs. Once the action is executed in the simulator, the copilot provides an aural confirmation to close the loop.

The goal is to create a hands-free, natural interaction layer for pilots, allowing them to issue operational commands just by speaking rather than navigating menus or controls. The system brings together speech recognition, intent classification, parameter extraction, and tight integration with the XPPython3 plugin environment.

## Features
- Audio-to-text command processing  
- NLP-based intent and parameter extraction  
- Automatic mapping to X-Plane datarefs  
- Real-time execution of simulator actions  
- Aural confirmation after successful command completion  
