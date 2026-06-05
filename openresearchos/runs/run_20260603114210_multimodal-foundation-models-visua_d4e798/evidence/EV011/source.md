# EV011: Automatic Slide Updating with User-Defined Dynamic Templates and Natural Language Instructions

URL: https://www.semanticscholar.org/paper/03e17954ab6cb5a1145adcecb862fd4229fc769e
Year: 2026
Source: semantic_scholar
Arxiv: 2604.17894

## Abstract

Presentation slides are a primary medium for data-driven reporting, yet keeping complex, analytics-style decks up to date remains labor-intensive. Existing automation methods mostly follow fixed template filling and cannot support dynamic updates for diverse, user-authored slide decks. We therefore define"Dynamic Slide Update via Natural Language Instructions on User-provided Templates"and introduce DynaSlide, a large-scale benchmark with 20,036 real-world instruction-execution triples (source slide, user instruction, target slide) grounded in a shared external database and built from business reporting slides under bring-your-own-template (BYO-template) conditions. To tackle this task, we propose SlideAgent, an agent-based framework that combines multimodal slide parsing, natural language instruction grounding, and tool-augmented reasoning for tables, charts, and textual conclusions. SlideAgent updates content while preserving layout and style, providing a strong reference baseline on DynaSlide. We further design end-to-end and component-level evaluation protocols that reveal key challenges and opportunities for future research. The dataset and code are available at https://github.com/XiaoZhou2024/SlideAgent.
