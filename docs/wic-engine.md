# Recursive WIC Engine (What-If Cascade)

The **Recursive WIC Engine** is an interactive, AI-powered systems-thinking tool designed to help you break out of traditional problem-solving constraints. Instead of solving a problem directly, the WIC Engine systematically dismantles the invisible assumptions holding the problem in place, thrusts you into radical alternative realities, and helps you harvest those realities for breakthrough startup ideas or solutions.

It features a full-stack architecture with a **FastAPI/PostgreSQL backend** orchestrating parallel OpenAI calls, and a **React Flow frontend** rendering the recursive cascade of thought as an interactive, glassmorphic DAG (Directed Acyclic Graph).

---

## 🧠 How the Cascade Works

When you submit a "Problem Statement," the engine triggers a systematic cascade of AI tasks:

1. **Assumption Extraction**: The AI acts as an "assumption archaeologist." It looks at your problem and identifies the single most load-bearing, invisible assumption that makes the problem true. If this assumption is broken, the problem ceases to exist in its current form.
2. **Inversion Generation**: The engine generates three radical "What-If" scenarios that intentionally violate that core assumption in three specific ways:
   * **Opposite** (Red): What if the exact reverse of the assumption is true?
   * **Extremity** (Orange): What if the assumption is taken to an absurd, infinite extreme?
   * **Alien Substrate** (Purple): What if the problem is solved using a completely unrelated, bizarre mechanism (biology, physics, economics, etc.)?
3. **World Simulation**: You select one of the three inversions. The engine then streams a real-time "simulation" of a world where this inversion is the new normal. It identifies:
   * *Breaks*: What legacy systems collapse?
   * *Abundances*: What scarce resource suddenly becomes infinite?
   * *New Constraints*: What new rules govern this world?
   * *New Actors*: Who thrives here?
   * *Narrative*: A short story of a day in the life of this new world.
4. **The Choice (Recurse or Synthesize)**: Once the simulation finishes, you have a choice:
   * **Go Deeper**: Take this new simulated world and treat it as the *new* baseline reality. Extract the assumptions of *this* world, invert them again, and simulate an even weirder, more advanced reality (Depth 1, Depth 2, etc.).
   * **Harvest Idea (Synthesize)**: Pull the "alien technology" from the simulated world back into our present reality to create a concrete, actionable product, startup, or solution.

---

## 🚀 User Guide

### Prerequisites
1. **PostgreSQL**: You must have a local PostgreSQL server running with a database created named `wic_engine`. 
   > *Tip for Mac*: You can use [Postgres.app](https://postgresapp.com/) or install via Homebrew (`brew install postgresql@15`). Run `CREATE DATABASE wic_engine;` in `psql`.
2. **OpenAI API Key**: Ensure you have an active `.env` file in the root project directory containing your API key:
   ```env
   OPENAI_API_KEY=sk-...
   DATABASE_URL=postgresql://postgres:postgres@localhost:5432/wic_engine
   ```

### 1. Starting the Servers
You need two terminal windows open simultaneously to run both the frontend and backend.

**Terminal 1: The Backend (FastAPI)**
```bash
cd backend
source venv/bin/activate
uvicorn main:app --reload
```
*Runs on `http://localhost:8000`*

**Terminal 2: The Frontend (React/Vite)**
```bash
cd frontend
npm run dev
```
*Runs on `http://localhost:5173`*

### 2. Running a Cascade

1. **Launch**: Open `http://localhost:5173` in your browser. You will see a dark, starry canvas with a purple **PROBLEM STATEMENT** node.
2. **Input**: Type a stubborn problem or challenge you are facing into the text box. Example: *"Air travel is too expensive and bad for the environment because airplanes require massive amounts of jet fuel to overcome gravity."*
3. **Cascade**: Click the **Start Cascade** button.
   * The status bar at the bottom will say "Extracting..."
   * The graph will automatically generate an orange **Hidden Assumption** node, followed by a blue/cyan **What-If Inversions** node with three distinct cards.
4. **Choose Your Reality**: Read the three generated inversions carefully. Click on the card that feels the most compelling or bizarre. The unselected cards will grey out.
5. **Watch the Simulation**: A pink **World Simulation** node will appear, and the narrative of this new world will stream in real-time. Wait for the simulation to finish.
6. **Take Action**: Look at the bottom Action Bar.
   * Click **Go Deeper** to run the entire process recursively on the new simulated reality. You can repeat this up to the max depth limit (default is Depth 3).
   * Click **Synthesize Now** to convert the simulation into a green **Synthesized Idea** node, containing a real-world startup pitch, target user, mechanism, minimum viable test (MVT), and adjacent analogues.

### 3. Modifying the Engine
- **Prompts**: If you want to change the "personality" of the AI or how it breaks down problems, all system prompts are hardcoded as constants in `backend/llm_service.py`.
- **Styling**: The UI is built using React Flow with custom CSS. Global design tokens are in `frontend/src/index.css` and node-specific glassmorphic styles are in `frontend/src/components/nodes/nodes.css`.