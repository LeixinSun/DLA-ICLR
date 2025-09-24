def build_multi_dim_scoring_prompt(role, rubric_dict, question, parsed_answer):
    rubric_sections = "\n\n".join(
        [f"Dimension: {dim}\n" + "\n".join([f"Score = {k} --> {v}" for k, v in scores.items()])
         for dim, scores in rubric_dict.items()]
    )

    return f"""
You are a **strict and meticulous quality control analyst**. Your task is to critically evaluate an agent's response based on its assigned role and a detailed rubric.

**Your Mindset:**
- Start with the assumption that the response is not perfect. Your goal is to identify flaws, inconsistencies, and areas for improvement.
- **Do not give high scores lightly.** A score of 5 is for a truly flawless and exceptional response. A score of 4 is for a very strong response with only trivial imperfections.
- A standard, correct but unexceptional answer should receive a score of 3. Do not hesitate to assign scores of 1 or 2 if the response has significant issues.

You will be given the agent's role, the user's question, the agent's response and the rubrics. Analyze the response against the provided rubrics with a critical eye.

**Evaluation Role:** {role}

**Question:**
{question}

**Agent Response (parsed_answer):**
{parsed_answer}

This the explanation of the abbreviations in the rubrics:
  1.  **GA:Goal Alignment** 
  2.  **RC:Role Consistency** 
  3.  **KBA:Knowledge Boundary Adherence** 
  4.  **CC:Constraint Compliance** 

**Evaluation Rubrics:**
{rubric_sections}

---
**Instructions:**
Based on your critical analysis, provide a JSON object containing your evaluation. For each dimension:
1.  Write a **concise and specific justification** for the score, highlighting both strengths and, more importantly, any weaknesses.
2.  Assign a numeric **score from 1.00 to 5.00**.You can also give scores like 1.23, 2.45, etc., if you feel it is necessary to reflect the quality more accurately.

**Output ONLY the JSON object, with no other text before or after it.**

Example of a critical evaluation:
{{
  "GA": {{
    "score": 4,
    "justification": "The response correctly addresses the main goal, but fails to consider an important edge case mentioned in the question, making the solution incomplete."
  }},
  "RC": {{
    "score": 3,
    "justification": "The tone is generally appropriate, but the use of overly casual phrasing ('you know', 'stuff like that') is inconsistent with the formal '{role}' persona."
  }},
  "KBA": {{
    "score": 5,
    "justification": "The response demonstrates perfect adherence to its knowledge domain, with no hallucinations or irrelevant information."
  }},
  "CC": {{
    "score": 2,
    "justification": "The response explicitly violates the constraint 'do not use the `eval` function', which is a major failure."
  }}
}}
"""