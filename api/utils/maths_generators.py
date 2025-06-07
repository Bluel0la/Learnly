import random
from typing import Literal


def generate_addition_question(difficulty: Literal["easy", "medium", "pro"]) -> dict:
    # 1. Difficulty-specific ranges
    difficulty_map = {
        "easy": (1, 20),
        "medium": (10, 100),
        "pro": (50, 200),
    }
    low, high = difficulty_map[difficulty]

    # 2. Sign combinations
    sign_combinations = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
    sign_a, sign_b = random.choice(sign_combinations)

    # 3. Operands
    a = sign_a * random.randint(low, high)
    b = sign_b * random.randint(low, high)
    correct = a + b

    # 4. Question templates
    question_templates = [
        f"What is {a} + {b}?",
        f"Calculate the sum: {a} + {b}",
        f"Find the result of adding {a} and {b}.",
        f"How much is {a} plus {b}?",
    ]
    question = random.choice(question_templates)

    # 5. Generate 3 plausible distractors
    def generate_distractors(correct, a, b):
        options = set()
        while len(options) < 3:
            offset = random.choice([-10, -5, -1, 1, 5, 10])
            wrong = correct + offset
            if wrong != correct:
                options.add(wrong)
        return list(options)

    distractors = generate_distractors(correct, a, b)

    # 6. Shuffle choices
    choices = distractors + [correct]
    random.shuffle(choices)

    # 7. Explanation logic
    explanation_lines = [
        f"Step 1: Identify the expression: **{a} + {b}**.",
        "Step 2: This is an addition problem. Combine the two numbers.",
    ]
    if a > 0 and b > 0:
        explanation_lines.append("Both are positive, so the result is their sum.")
    elif a < 0 and b < 0:
        explanation_lines.append("Both are negative, so the result is also negative.")
    else:
        explanation_lines.append(
            "One is positive and one is negative, so find the difference and keep the sign of the larger absolute value."
        )
    explanation_lines.append(f"Step 3: Compute the result: {a} + {b} = {correct}.")
    explanation_lines.append(f"✅ Final Answer: **{correct}**")

    explanation = "\n".join(explanation_lines)

    # 8. Return structured payload
    return {
        "question": question,
        "correct_answer": str(correct),
        "choices": [str(choice) for choice in choices],
        "difficulty": difficulty,
        "topic": "addition",
        "explanation": explanation,
    }
