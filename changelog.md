# V7

## 04.08.25
- Added config.yaml to src/helper to unify configuration

## 05.08.25
- Added Modelfile and created "topsim-expert-v1" based on gemma3n:e4b in ollama for automatic system-prompting 
- Added Query-Expander to src/generation to automatically expand queries 
  - inlcudes a new config.yaml setting to enable it
  - includes a new config.yaml setting to control the minimum number of letters for expansion
  - new Modelfile "QueryExpander"
- Switched to Stream instead of Full Output for better user experience

## 06.08.25
- Added Conversational Memory to src/generation to remember the previous query and answer (experimental)
  - includes a new config.yaml setting to enable it
  - new Modelfile "ConversationCondenser"
- Added low-Vram-Mode in config.yaml for dynamic model loading/unloading