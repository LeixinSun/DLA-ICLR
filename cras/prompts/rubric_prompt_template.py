def build_rubric_prompt(role: str) -> str:
    return f"""
You are an expert evaluator, tasked with creating a **strict and highly discerning** scoring rubric. Your goal is to evaluate how well an AI agent adheres to its assigned role: "{role}".

The rubric must be precise enough to differentiate between exceptional, mediocre, and poor responses. For each of the four dimensions below, define clear, distinct standards for scores from 1 (critically flawed) to 5 (flawless and exceptional).

**Crucially, define the score levels as follows:**
- **Score 5 (Flawless/Exceptional):** The response is perfect. It not only meets all requirements but does so with elegance, depth, or insight. There are no discernible flaws.
- **Score 3 (Acceptable/Adequate):** The response is largely correct and addresses the main points, but may have minor errors, omissions, or stylistic inconsistencies. It gets the job done, but is not impressive.
- **Score 1 (Critically Flawed):** The response has significant errors, fails to address the core task, or fundamentally violates the role's principles. It is unhelpful or misleading.

Use these definitions to flesh out the 1-5 scale for each dimension:

1.  **Goal Alignment (GA):** How well does the agent's response align with its specific subgoal?
    *Think about: Does it just answer the question, or does it provide a complete, actionable, and insightful solution? Does it misunderstand a key part of the goal?*

2.  **Role Consistency (RC):** Is the response stylistically and logically consistent with the agent’s designated role of a "{role}"?
    *Think about: Does the tone, vocabulary, and reasoning style truly reflect the role? Or does it sound like a generic chatbot? Are there logical inconsistencies?*

3.  **Knowledge Boundary Adherence (KBA):** Does the agent stay strictly within its knowledge domain?
    *Think about: Does it invent facts (hallucinate)? Does it claim ignorance when it should know the answer? Does it provide information outside its designated expertise?*

4.  **Constraint Compliance (CC):** Does the response fully comply with all explicit constraints (e.g., "do not use a certain library," "provide the answer in French")?
    *Think about: Does it ignore a constraint? Does it find a sloppy workaround? Or does it respect the constraint perfectly?*

Please provide your highly discerning rubric in a strict JSON format. Do not include any text outside the JSON block.

{{
  "role": "{role}",
  "rubric": {{
    "GA": {{
      "1": "...",
      "2": "...",
      "3": "...",
      "4": "...",
      "5": "..."
    }},
    "RC": {{...}},
    "KBA": {{...}},
    "CC": {{...}}
  }}
}}
""".strip()