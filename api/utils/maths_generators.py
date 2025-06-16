from typing import Literal
from fractions import Fraction
from math import gcd
import random, re
from functools import reduce
from collections import Counter


OPS = {
    "+": {"func": "calculator.add", "precedence": 1},
    "-": {"func": "calculator.subtract", "precedence": 1},
    "*": {"func": "calculator.multiply", "precedence": 2},
    "/": {"func": "calculator.divide", "precedence": 2},
    "^": {"func": "calculator.power", "precedence": 3},
}

def random_operand():
    value = random.randint(1, 10)
    sign = random.choice([-1, 1])
    return str(sign * value)


def prime_factors(n):
    i = 2
    factors = []
    while i * i <= n:
        if n % i:
            i += 1
        else:
            n //= i
            factors.append(i)
    if n > 1:
        factors.append(n)
    return factors


def random_fraction(sign=True, max_den=12, max_num=20):
    denom = random.randint(2, max_den)
    numer = random.randint(1, max_num)
    if sign and random.choice([True, False]):
        numer = -numer
    return Fraction(numer, denom)


def format_fraction(f: Fraction) -> str:
    return f"{f.numerator}/{f.denominator}"


def generate_expression(level, max_level):
    if level >= max_level:
        return random_operand()
    left = generate_expression(level + 1, max_level)
    right = generate_expression(level + 1, max_level)
    op = random.choice(list(OPS.keys()))
    return f"({left} {op} {right})"


def tokenize(expr):
    # Match signed integers and operators
    token_pattern = re.compile(r"-?\d+|\^|\*|\/|\+|\-|\(|\)")
    tokens = token_pattern.findall(expr)
    return tokens


class Parser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.pos = 0

    def peek(self):
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def consume(self, expected=None):
        token = self.peek()
        if expected and token != expected:
            raise ValueError(f"Expected {expected} but got {token}")
        self.pos += 1
        return token

    def parse_expression(self, min_prec=1):
        left = self.parse_primary()
        while True:
            op = self.peek()
            if op not in OPS or OPS[op]["precedence"] < min_prec:
                break
            prec = OPS[op]["precedence"]
            next_min_prec = prec + (0 if op == "^" else 1)
            self.consume(op)
            right = self.parse_expression(next_min_prec)
            left = {"type": "binop", "op": op, "left": left, "right": right}
        return left

    def parse_primary(self):
        token = self.peek()
        if token == "(":
            self.consume("(")
            node = self.parse_expression()
            self.consume(")")
            return node
        elif re.match(r"-?\d+", token):
            self.consume()
            return {"type": "number", "value": int(token)}
        else:
            raise ValueError(f"Unexpected token {token}")


def explain_ast(node, step_counter=None):
    if step_counter is None:
        step_counter = {"count": 1}
    if node["type"] == "number":
        value = node["value"]
        formatted = f"({value})" if value < 0 else str(value)
        return formatted, []
    left_str, left_steps = explain_ast(node["left"], step_counter)
    right_str, right_steps = explain_ast(node["right"], step_counter)
    op = node["op"]
    func = OPS[op]["func"]
    step_num = step_counter["count"]
    explanation = (
        f"Step {step_num}: Calculate {left_str} {op} {right_str}:\n"
        f"   → {func}({left_str}, {right_str})"
    )
    step_counter["count"] += 1
    steps = left_steps + right_steps + [explanation]
    expr_str = f"{func}({left_str}, {right_str})"
    return expr_str, steps


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
    explanation_lines.append(f" Final Answer: **{correct}**")

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


def generate_subtraction_question(difficulty: Literal["easy", "medium", "pro"]) -> dict:
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
    correct = a - b

    # 4. Question templates
    question_templates = [
        f"What is {a} - {b}?",
        f"Calculate the result of: {a} minus {b}",
        f"Find the difference: {a} - {b}",
        f"How much is {a} subtract {b}?",
    ]
    question = random.choice(question_templates)

    # 5. Generate 3 distractors
    def generate_distractors(correct):
        distractors = set()
        while len(distractors) < 3:
            offset = random.choice([-10, -5, -1, 1, 5, 10])
            wrong = correct + offset
            if wrong != correct:
                distractors.add(wrong)
        return list(distractors)

    distractors = generate_distractors(correct)

    # 6. Shuffle choices
    choices = distractors + [correct]
    random.shuffle(choices)

    # 7. Explanation
    explanation = []
    explanation.append(f"Step 1: This is a subtraction problem: **{a} - {b}**.")
    explanation.append(
        "Step 2: Subtraction means finding the difference between two numbers."
    )
    explanation.append(
        "Another way to think about subtraction is adding the opposite of the number being subtracted."
    )

    if b > 0:
        explanation.append(
            "Step 3: Subtracting a positive typically decreases the value."
        )
    else:
        explanation.append(
            "Step 3: Subtracting a negative is equivalent to adding, so it increases the value."
        )

    if a < 0 and b > 0:
        explanation.append(
            "Step 4: Starting from a negative and subtracting a positive moves further negative."
        )
    elif a < 0 and b < 0:
        explanation.append(
            "Step 4: Subtracting a negative from a negative increases the value."
        )
    elif a > 0 and b < 0:
        explanation.append(
            "Step 4: Subtracting a negative from a positive increases the total."
        )
    else:
        explanation.append(
            "Step 4: Both numbers are positive, so subtraction proceeds normally."
        )

    explanation.append(f"Step 5: Compute the result: {a} - {b} = {correct}.")
    explanation.append(f" Final Answer: **{correct}**")

    return {
        "question": question,
        "correct_answer": str(correct),
        "choices": [str(choice) for choice in choices],
        "difficulty": difficulty,
        "topic": "subtraction",
        "explanation": "\n".join(explanation),
    }


def generate_division_question(difficulty: Literal["easy", "medium", "pro"]) -> dict:
    # 1. Difficulty range for quotient and divisor
    difficulty_map = {
        "easy": (2, 10),
        "medium": (5, 20),
        "pro": (10, 40),
    }
    low, high = difficulty_map[difficulty]

    # 2. Sign combinations
    sign_combinations = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
    sign_q, sign_d = random.choice(sign_combinations)

    # 3. Generate divisor and quotient to ensure divisible result
    divisor = sign_d * random.randint(low, high)
    quotient = sign_q * random.randint(low, high)
    dividend = divisor * quotient
    correct = quotient

    # 4. Question phrasing
    question_templates = [
        f"What is {dividend} ÷ {divisor}?",
        f"Calculate: {dividend} / {divisor}",
        f"Find how many times {divisor} fits into {dividend}.",
        f"Divide {dividend} by {divisor}.",
    ]
    question = random.choice(question_templates)

    # 5. Generate distractors
    def generate_distractors(correct):
        distractors = set()
        while len(distractors) < 3:
            offset = random.choice([-10, -3, -1, 1, 2, 5])
            wrong = correct + offset
            if wrong != correct:
                distractors.add(wrong)
        return list(distractors)

    distractors = generate_distractors(correct)

    # 6. Shuffle choices
    choices = distractors + [correct]
    random.shuffle(choices)

    # 7. Build explanation
    explanation = [
        f"Step 1: This is a division problem: **{dividend} ÷ {divisor}**.",
        "Step 2: Division means finding how many equal groups the divisor fits into the dividend.",
        f"Step 3: {dividend} ÷ {divisor} = {quotient} because {quotient} × {divisor} = {dividend}.",
    ]

    if (dividend > 0 and divisor > 0) or (dividend < 0 and divisor < 0):
        explanation.append(
            "Step 4: Since both numbers have the same sign, the result is positive."
        )
    else:
        explanation.append("Step 4: Since the signs differ, the result is negative.")

    explanation.append(f" Final Answer: **{quotient}**")

    return {
        "question": question,
        "correct_answer": str(correct),
        "choices": [str(c) for c in choices],
        "difficulty": difficulty,
        "topic": "division",
        "explanation": "\n".join(explanation),
    }


def generate_multiplication_question(
    difficulty: Literal["easy", "medium", "pro"],
) -> dict:
    # 1. Difficulty ranges
    difficulty_map = {
        "easy": (2, 10),
        "medium": (5, 20),
        "pro": (10, 40),
    }
    low, high = difficulty_map[difficulty]

    # 2. Choose operand signs
    sign_combinations = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
    sign_a, sign_b = random.choice(sign_combinations)

    # 3. Generate operands and result
    a = sign_a * random.randint(low, high)
    b = sign_b * random.randint(low, high)
    correct = a * b

    # 4. Question templates
    question_templates = [
        f"What is {a} × {b}?",
        f"Calculate the product of {a} and {b}.",
        f"How much is {a} times {b}?",
        f"Multiply: {a} * {b}",
    ]
    question = random.choice(question_templates)

    # 5. Distractors
    def generate_distractors(correct):
        distractors = set()
        while len(distractors) < 3:
            offset = random.choice([-10, -5, -2, 2, 5, 10])
            wrong = correct + offset
            if wrong != correct:
                distractors.add(wrong)
        return list(distractors)

    distractors = generate_distractors(correct)

    # 6. Shuffle options
    choices = distractors + [correct]
    random.shuffle(choices)

    # 7. Explanation
    explanation = [
        f"Step 1: This is a multiplication problem: **{a} × {b}**.",
        "Step 2: Multiplication is repeated addition.",
    ]

    abs_a, abs_b = abs(a), abs(b)
    if abs_b <= 5:
        repeated = " + ".join([str(a)] * abs_b)
        explanation.append(f"Step 3: Add {a}, {abs_b} times → {repeated}")
    elif abs_a <= 5:
        repeated = " + ".join([str(b)] * abs_a)
        explanation.append(f"Step 3: Add {b}, {abs_a} times → {repeated}")
    else:
        explanation.append(
            f"Step 3: This would be adding {a} repeatedly, {abs_b} times."
        )

    if a * b > 0:
        explanation.append("Step 4: Same signs → product is positive.")
    else:
        explanation.append("Step 4: Different signs → product is negative.")

    explanation.append(f" Final Answer: **{a} × {b} = {correct}**")

    return {
        "question": question,
        "correct_answer": str(correct),
        "choices": [str(c) for c in choices],
        "difficulty": difficulty,
        "topic": "multiplication",
        "explanation": "\n".join(explanation),
    }


def generate_decimal_addition_question(
    difficulty: Literal["easy", "medium", "pro"],
) -> dict:
    # Difficulty controls range and decimal precision
    difficulty_map = {
        "easy": (1, 20, 1),  # simpler numbers, 1 decimal place
        "medium": (10, 50, 2),  # standard range, 2 decimal places
        "pro": (25, 100, 2),  # larger numbers, 2 decimal places
    }
    low, high, precision = difficulty_map[difficulty]

    # Generate operands with sign
    sign_combinations = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
    sign_a, sign_b = random.choice(sign_combinations)

    a = round(sign_a * random.uniform(low, high), precision)
    b = round(sign_b * random.uniform(low, high), precision)
    correct = round(a + b, precision)

    # Question prompt
    question_templates = [
        f"What is {a} + {b}?",
        f"Add the decimals: {a} + {b}",
        f"Calculate the sum of {a} and {b}.",
        f"How much is {a} plus {b}?",
    ]
    question = random.choice(question_templates)

    # Distractors
    def generate_distractors(correct):
        distractors = set()
        while len(distractors) < 3:
            offset = round(random.uniform(-5, 5), precision)
            option = round(correct + offset, precision)
            if option != correct:
                distractors.add(option)
        return list(distractors)

    distractors = generate_distractors(correct)
    choices = distractors + [correct]
    random.shuffle(choices)

    # Explanation
    explanation = [f"Step 1: This is a decimal addition: **{a} + {b}**."]
    explanation.append("Step 2: Line up the decimal points and add the numbers.")

    if sign_a == sign_b:
        explanation.append(
            f"Step 3: Both are {'positive' if sign_a > 0 else 'negative'} → just add and keep the sign."
        )
    else:
        explanation.append(
            "Step 3: One is negative → subtract smaller absolute value from larger and keep that sign."
        )

    explanation.append(f"Step 4: The total is `calculator.add({a}, {b})`.")
    explanation.append(f" Final Answer: {correct}")

    return {
        "question": question,
        "correct_answer": str(correct),
        "choices": [str(c) for c in choices],
        "difficulty": difficulty,
        "topic": "decimal addition",
        "explanation": "\n".join(explanation),
    }


def generate_decimal_subtraction_question(
    difficulty: Literal["easy", "medium", "pro"],
) -> dict:
    # Difficulty settings: control number range and precision
    difficulty_map = {
        "easy": (1, 20, 1),
        "medium": (10, 50, 2),
        "pro": (25, 100, 2),
    }
    low, high, precision = difficulty_map[difficulty]

    # Randomize signs
    sign_combinations = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
    sign_a, sign_b = random.choice(sign_combinations)

    a = round(sign_a * random.uniform(low, high), precision)
    b = round(sign_b * random.uniform(low, high), precision)
    correct = round(a - b, precision)

    # Question phrasing
    question_templates = [
        f"What is {a} - {b}?",
        f"Calculate: {a} - {b}",
        f"Find the difference between {a} and {b}.",
        f"How much is {b} subtracted from {a}?",
    ]
    question = random.choice(question_templates)

    # Generate distractors
    def generate_distractors(correct_value):
        options = set()
        while len(options) < 3:
            offset = round(random.uniform(-5, 5), precision)
            distractor = round(correct_value + offset, precision)
            if distractor != correct_value:
                options.add(distractor)
        return list(options)

    distractors = generate_distractors(correct)
    choices = distractors + [correct]
    random.shuffle(choices)

    # Explanation
    explanation = [
        f"Step 1: This is a subtraction problem with decimals: **{a} - {b}**.",
        "Step 2: Subtraction tells us how much remains when one quantity is taken from another.",
    ]

    if sign_b < 0:
        explanation.append(
            f"Step 3: Subtracting a negative number is the same as adding its positive value → becomes **{a} + {abs(b)}**."
        )
    else:
        explanation.append(
            f"Step 3: Direct subtraction of {b} from {a}, considering signs."
        )

    explanation.append(
        f"Step 4: Use calculator logic: `calculator.subtract({a}, {b})`."
    )
    explanation.append(f" Final Answer: {correct}")

    return {
        "question": question,
        "correct_answer": str(correct),
        "choices": [str(choice) for choice in choices],
        "difficulty": difficulty,
        "topic": "decimal subtraction",
        "explanation": "\n".join(explanation),
    }


def generate_decimal_multiplication_question(
    difficulty: Literal["easy", "medium", "pro"],
) -> dict:
    # 1. Difficulty tiers → (min, max, decimal precision)
    difficulty_map = {
        "easy": (1, 10, 1),
        "medium": (5, 25, 2),
        "pro": (10, 50, 2),
    }
    low, high, precision = difficulty_map[difficulty]

    # 2. Randomized signs
    sign_combinations = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
    sign_a, sign_b = random.choice(sign_combinations)

    a = round(sign_a * random.uniform(low, high), precision)
    b = round(sign_b * random.uniform(low, high), precision)
    correct = round(a * b, precision)

    # 3. Question template
    question_templates = [
        f"What is {a} × {b}?",
        f"Calculate: {a} * {b}",
        f"Find the product of {a} and {b}.",
        f"How much is {a} multiplied by {b}?",
    ]
    question = random.choice(question_templates)

    # 4. Generate distractors
    def generate_distractors(correct):
        options = set()
        while len(options) < 3:
            offset = round(random.uniform(-5, 5), precision)
            wrong = round(correct + offset, precision)
            if wrong != correct:
                options.add(wrong)
        return list(options)

    distractors = generate_distractors(correct)
    choices = distractors + [correct]
    random.shuffle(choices)

    # 5. Explanation
    explanation = [
        f"Step 1: Decimal multiplication: **{a} * {b}**.",
        "Step 2: Multiplication with decimals means scaling one value by the other.",
    ]

    if abs(b).is_integer() and abs(b) <= 5:
        explanation.append(
            f"Step 3: Repeated addition example → {a} added {int(abs(b))} times: "
            + " + ".join([str(a)] * int(abs(b)))
        )
    elif abs(a).is_integer() and abs(a) <= 5:
        explanation.append(
            f"Step 3: Alternatively, {b} added {int(abs(a))} times: "
            + " + ".join([str(b)] * int(abs(a)))
        )
    else:
        explanation.append(
            "Step 3: Repeated addition is not practical for large/decimal values."
        )

    if a * b >= 0:
        explanation.append("Step 4: Signs are the same → product is positive.")
    else:
        explanation.append("Step 4: Signs differ → product is negative.")

    explanation.append(
        "Step 5: Multiplying decimals may change decimal precision of the result."
    )
    explanation.append(f" Final Answer: {correct}")

    # 6. Return final structure
    return {
        "question": question,
        "correct_answer": str(correct),
        "choices": [str(c) for c in choices],
        "difficulty": difficulty,
        "topic": "decimal multiplication",
        "explanation": "\n".join(explanation),
    }

def generate_decimal_division_question(
    difficulty: Literal["easy", "medium", "pro"]
) -> dict:
    # 1. Difficulty-based ranges
    difficulty_map = {
        "easy": (2, 10, 1),
        "medium": (5, 25, 2),
        "pro": (10, 50, 2),
    }
    a_min, a_max, precision = difficulty_map[difficulty]

    # 2. Random signs
    sign_combinations = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
    sign_a, sign_b = random.choice(sign_combinations)

    # 3. Generate operands with safe divisor
    a = round(sign_a * random.uniform(a_min, a_max), precision)
    b = round(sign_b * random.uniform(1.0, 10.0), precision)
    b = b if abs(b) >= 0.01 else 1.0 * sign_b  # avoid division near zero

    correct = round(a / b, precision)

    # 4. Question phrasing
    question_templates = [
        f"What is {a} / {b}?",
        f"Calculate: {a} divided by {b}",
        f"Find how many times {b} fits into {a}.",
        f"How much is {a} divided by {b}?",
    ]
    question = random.choice(question_templates)

    # 5. Generate distractors
    def generate_distractors(correct):
        options = set()
        while len(options) < 3:
            offset = round(random.uniform(-5, 5), precision)
            wrong = round(correct + offset, precision)
            if wrong != correct:
                options.add(wrong)
        return list(options)

    distractors = generate_distractors(correct)
    choices = distractors + [correct]
    random.shuffle(choices)

    # 6. Explanation
    explanation = [
        f"Step 1: This is a division problem with decimals: **{a} / {b}**.",
        "Step 2: Division means splitting one number into equal parts or finding how many times one number fits into another.",
        f"Step 3: We compute how many times {b} fits into {a}.",
    ]

    if a * b >= 0:
        explanation.append("Step 4: Signs are the same, so the result is positive.")
    else:
        explanation.append("Step 4: Signs are different, so the result is negative.")

    explanation.append("Step 5: Since decimals are involved, the result may also be a decimal.")
    explanation.append(f" Final Answer: {correct}")

    return {
        "question": question,
        "correct_answer": str(correct),
        "choices": [str(c) for c in choices],
        "difficulty": difficulty,
        "topic": "decimal division",
        "explanation": "\n".join(explanation),
    }


def generate_fraction_addition_question(
    difficulty: Literal["easy", "medium", "pro"],
) -> dict:
    # Difficulty scaling
    max_denom = {"easy": 6, "medium": 10, "pro": 20}[difficulty]
    a = random_fraction(sign=True, max_den=max_denom)
    b = random_fraction(sign=True, max_den=max_denom)

    correct = a + b
    correct_str = format_fraction(correct)

    a_str = format_fraction(a)
    b_str = format_fraction(b)

    question = random.choice(
        [
            f"What is {a_str} + {b_str}?",
            f"Add the fractions: {a_str} and {b_str}",
            f"Calculate: {a_str} plus {b_str}",
            f"Find the sum of {a_str} and {b_str}.",
        ]
    )

    def generate_distractors(correct: Fraction):
        distractors = set()
        while len(distractors) < 3:
            noise = Fraction(random.randint(-3, 3), random.randint(2, max_denom))
            distractor = correct + noise
            if distractor != correct:
                distractors.add(format_fraction(distractor))
        return list(distractors)

    choices = generate_distractors(correct) + [correct_str]
    random.shuffle(choices)

    denom_a, denom_b = a.denominator, b.denominator
    lcm = denom_a * denom_b // gcd(denom_a, denom_b)

    explanation = [
        f"Step 1: This is a fraction addition problem: **{a_str} + {b_str}**.",
        "Step 2: Convert to a common denominator.",
        f"→ LCD of {denom_a} and {denom_b} = {lcm}",
        f"→ Convert {a_str} to: {a.numerator * (lcm // denom_a)}/{lcm}",
        f"→ Convert {b_str} to: {b.numerator * (lcm // denom_b)}/{lcm}",
        f"Step 3: Add numerators: {a.numerator * (lcm // denom_a)} + {b.numerator * (lcm // denom_b)}",
        f"→ Final Result: {correct_str}",
    ]

    return {
        "question": question,
        "correct_answer": correct_str,
        "choices": choices,
        "difficulty": difficulty,
        "topic": "fraction addition",
        "explanation": "\n".join(explanation),
    }


def generate_fraction_subtraction_question(
    difficulty: Literal["easy", "medium", "pro"],
) -> dict:
    # Set max denominator per difficulty
    max_denom = {"easy": 6, "medium": 12, "pro": 20}[difficulty]
    a = random_fraction(sign=True, max_den=max_denom)
    b = random_fraction(sign=True, max_den=max_denom)

    correct = a - b
    correct_str = format_fraction(correct)

    a_str = format_fraction(a)
    b_str = format_fraction(b)

    # Generate question
    question = random.choice(
        [
            f"What is {a_str} - {b_str}?",
            f"Calculate: {a_str} minus {b_str}",
            f"Find the result of {a_str} subtract {b_str}.",
            f"Subtract the fractions: {a_str} and {b_str}",
        ]
    )

    # Generate distractors
    def generate_distractors(correct: Fraction):
        distractors = set()
        while len(distractors) < 3:
            noise = Fraction(random.randint(-2, 2), random.randint(2, max_denom))
            wrong = correct + noise
            if wrong != correct:
                distractors.add(format_fraction(wrong))
        return list(distractors)

    choices = generate_distractors(correct) + [correct_str]
    random.shuffle(choices)

    # Explanation
    denom_a = a.denominator
    denom_b = b.denominator
    lcm = denom_a * denom_b // gcd(denom_a, denom_b)

    explanation = [
        f"Step 1: This is a fraction subtraction problem: **{a_str} - {b_str}**.",
        "Step 2: Convert to a common denominator.",
        f"→ LCD of {denom_a} and {denom_b} = {lcm}",
        f"→ Convert {a_str} to: {a.numerator * (lcm // denom_a)}/{lcm}",
        f"→ Convert {b_str} to: {b.numerator * (lcm // denom_b)}/{lcm}",
        f"Step 3: Subtract numerators: {a.numerator * (lcm // denom_a)} - {b.numerator * (lcm // denom_b)}",
        f"→ Final Result: {correct_str}",
    ]

    return {
        "question": question,
        "correct_answer": correct_str,
        "choices": choices,
        "difficulty": difficulty,
        "topic": "fraction subtraction",
        "explanation": "\n".join(explanation),
    }


def generate_fraction_multiplication_question(
    difficulty: Literal["easy", "medium", "pro"],
) -> dict:
    # Difficulty → max denominator scale
    max_denom = {"easy": 6, "medium": 12, "pro": 20}[difficulty]
    a = random_fraction(sign=True, max_den=max_denom)
    b = random_fraction(sign=True, max_den=max_denom)

    correct = a * b
    correct_str = format_fraction(correct)

    a_str = format_fraction(a)
    b_str = format_fraction(b)

    question = random.choice(
        [
            f"What is {a_str} × {b_str}?",
            f"Multiply the fractions: {a_str} and {b_str}",
            f"Calculate the product of {a_str} and {b_str}",
            f"Find the result when {a_str} is multiplied by {b_str}",
        ]
    )

    def generate_distractors(correct: Fraction):
        distractors = set()
        while len(distractors) < 3:
            perturb = Fraction(random.randint(-2, 2), random.randint(1, max_denom))
            wrong = correct + perturb
            if wrong != correct:
                distractors.add(format_fraction(wrong))
        return list(distractors)

    choices = generate_distractors(correct) + [correct_str]
    random.shuffle(choices)

    explanation = [
        f"Step 1: Multiply the numerators: {a.numerator} × {b.numerator} = {a.numerator * b.numerator}",
        f"Step 2: Multiply the denominators: {a.denominator} × {b.denominator} = {a.denominator * b.denominator}",
        f"Step 3: Result = {a.numerator * b.numerator}/{a.denominator * b.denominator}",
        f"Step 4: Simplify the fraction if possible.",
        f" Final Answer: {correct_str}",
    ]

    return {
        "question": question,
        "correct_answer": correct_str,
        "choices": choices,
        "difficulty": difficulty,
        "topic": "fraction multiplication",
        "explanation": "\n".join(explanation),
    }


def generate_fraction_division_question(
    difficulty: Literal["easy", "medium", "pro"],
) -> dict:
    # Difficulty controls size of denominators
    max_denom = {"easy": 6, "medium": 12, "pro": 20}[difficulty]

    a = random_fraction(sign=True, max_den=max_denom)
    b = random_fraction(sign=True, max_den=max_denom)

    while b == 0:
        b = random_fraction(sign=True, max_den=max_denom)

    correct = a / b
    correct_str = format_fraction(correct)

    a_str = format_fraction(a)
    b_str = format_fraction(b)

    question = random.choice(
        [
            f"What is {a_str} ÷ {b_str}?",
            f"Divide the fractions: {a_str} by {b_str}",
            f"Find the quotient of {a_str} divided by {b_str}.",
            f"Calculate: {a_str} ÷ {b_str}",
        ]
    )

    # --- Distractor Generator ---
    def generate_distractors(correct: Fraction):
        distractors = set()
        while len(distractors) < 3:
            perturb = Fraction(random.randint(-2, 2), random.randint(1, max_denom))
            wrong = correct + perturb
            if wrong != correct:
                distractors.add(format_fraction(wrong))
        return list(distractors)

    distractors = generate_distractors(correct)
    choices = distractors + [correct_str]
    random.shuffle(choices)

    reciprocal_b = Fraction(b.denominator, b.numerator)
    reciprocal_b_str = format_fraction(reciprocal_b)

    explanation = [
        f"Step 1: Division of fractions: **{a_str} ÷ {b_str}**",
        f"Step 2: Multiply {a_str} by the reciprocal of {b_str}: {reciprocal_b_str}",
        f"Step 3: Multiply numerators: {a.numerator} × {b.denominator}",
        f"Step 4: Multiply denominators: {a.denominator} × {b.numerator}",
        f"Step 5: Final expression: {a.numerator * b.denominator}/{a.denominator * b.numerator}",
        f"Step 6: Simplify result if possible.",
        f" Final Answer: {correct_str}",
    ]

    return {
        "question": question,
        "correct_answer": correct_str,
        "choices": choices,
        "difficulty": difficulty,
        "topic": "fraction division",
        "explanation": "\n".join(explanation),
    }


def generate_pemdas_question(difficulty: Literal["easy", "medium", "pro"]) -> dict:
    # Difficulty maps to expression depth
    level_map = {"easy": 1, "medium": 2, "pro": 4}
    max_level = level_map[difficulty]

    # 1. Generate expression string
    expr = generate_expression(0, max_level)

    # 2. Parse & explain
    tokens = tokenize(expr)
    parser = Parser(tokens)
    ast = parser.parse_expression()
    final_expr, steps = explain_ast(ast)

    # 3. Evaluate the AST safely
    try:
        correct_answer = str(eval(expr))
    except Exception:
        correct_answer = final_expr  # fallback if eval fails

    # 4. Generate distractors
    def generate_distractors(correct: str):
        correct_val = float(correct)
        distractors = set()
        while len(distractors) < 3:
            delta = random.choice([-3, -2, -1, 1, 2, 3])
            wrong = str(round(correct_val + delta, 2))
            if wrong != correct:
                distractors.add(wrong)
        return list(distractors)

    distractors = generate_distractors(correct_answer)
    choices = distractors + [correct_answer]
    random.shuffle(choices)

    # 5. Final assembly
    question = random.choice(
        [
            f"What is the value of {expr}?",
            f"Evaluate the expression: {expr}",
            f"Simplify: {expr}",
            f"Calculate: {expr}",
        ]
    )

    explanation = "\n".join(steps) + f"\n\n Final Answer: {correct_answer}"

    return {
        "question": question,
        "correct_answer": correct_answer,
        "choices": choices,
        "topic": "PEMDAS",
        "difficulty": difficulty,
        "explanation": explanation,
    }


def generate_percentage_question(difficulty: Literal["easy", "medium", "pro"]) -> dict:
    # 1. Define difficulty-specific ranges
    if difficulty == "easy":
        percent_choices = [10, 25, 50]
        number_range = (10, 100)
    elif difficulty == "medium":
        percent_choices = [5, 12, 20, 30, 40]
        number_range = (50, 200)
    else:  # pro
        percent_choices = [12.5, 17.5, 22.5, 33.33, 66.67]
        number_range = (100, 500)

    percent = random.choice(percent_choices)
    number = random.randint(*number_range)

    # 2. Construct question
    input_templates = [
        f"What is {percent}% of {number}?",
        f"Find {percent} percent of {number}.",
        f"How much is {percent}% of {number}?",
        f"Calculate: {percent}% of {number}",
    ]
    question = random.choice(input_templates)

    # 3. Compute correct answer
    correct = round((percent / 100) * number, 2)
    correct_str = str(correct)

    # 4. Generate distractors
    def generate_distractors(true_val):
        distractors = set()
        while len(distractors) < 3:
            offset = random.uniform(-15, 15)
            wrong = round(true_val + offset, 2)
            if wrong != true_val:
                distractors.add(str(wrong))
        return list(distractors)

    distractors = generate_distractors(correct)
    choices = distractors + [correct_str]
    random.shuffle(choices)

    # 5. Explanation
    explanation = [
        f"Step 1: You are asked to find {percent}% of {number}.",
        f"Step 2: Convert the percentage to a decimal → calculator.divide({percent}, 100)",
        f"Step 3: Multiply the decimal by {number} → calculator.multiply(calculator.divide({percent}, 100), {number})",
        f"Step 4: This gives the answer → {correct}",
        f" Final Answer: {correct}",
    ]

    return {
        "question": question,
        "correct_answer": correct_str,
        "choices": choices,
        "difficulty": difficulty,
        "topic": "percentage",
        "explanation": "\n".join(explanation),
    }


def generate_percentage_relationship_question(
    difficulty: Literal["easy", "medium", "pro"],
) -> dict:
    # 1. Difficulty scaling
    if difficulty == "easy":
        base = random.randint(20, 100)
        percent = random.choice([10, 20, 25, 50])
    elif difficulty == "medium":
        base = random.randint(50, 200)
        percent = random.choice([12, 15, 30, 40, 60])
    else:  # pro
        base = random.randint(100, 400)
        percent = random.choice([17, 22, 33, 45, 66, 75])

    part = round((percent / 100) * base, 2)
    correct = round((part / base) * 100, 2)
    correct_str = str(correct)

    # 2. Question templates
    question_templates = [
        f"What percent of {base} is {part}?",
        f"{part} is what percentage of {base}?",
        f"Find the percentage that {part} is of {base}.",
        f"Calculate: {part} is what percent of {base}?",
    ]
    question = random.choice(question_templates)

    # 3. Distractor generation
    def generate_distractors(correct):
        distractors = set()
        while len(distractors) < 3:
            offset = random.uniform(-15, 15)
            candidate = round(correct + offset, 2)
            if candidate != correct:
                distractors.add(str(candidate))
        return list(distractors)

    choices = generate_distractors(correct) + [correct_str]
    random.shuffle(choices)

    # 4. Explanation
    explanation = [
        f"Step 1: We are asked to find what percent {part} is of {base}.",
        f"Step 2: Divide the part by the whole → calculator.divide({part}, {base})",
        f"Step 3: Multiply the result by 100 to convert to percentage → calculator.multiply(calculator.divide({part}, {base}), 100)",
        f"Step 4: The final value is the percentage → {correct}%",
        f" Final Answer: {correct}%",
    ]

    return {
        "question": question,
        "correct_answer": correct_str,
        "choices": choices,
        "difficulty": difficulty,
        "topic": "percentage",
        "explanation": "\n".join(explanation),
    }


def generate_percentage_comparison_question(
    difficulty: Literal["easy", "medium", "pro"],
) -> dict:
    # 1. Difficulty scaling for original values and percent changes
    if difficulty == "easy":
        original = random.randint(10, 100)
        percent_change = random.choice([10, 20, 25])
    elif difficulty == "medium":
        original = random.randint(50, 200)
        percent_change = random.choice([15, 30, 40])
    else:  # pro
        original = random.randint(100, 500)
        percent_change = random.choice([17, 33, 45, 60])

    is_more = random.choice([True, False])
    if is_more:
        new_value = original + (original * percent_change) // 100
    else:
        new_value = original - (original * percent_change) // 100

    correct = round(abs((new_value - original) / original) * 100, 2)
    correct_str = str(correct)
    comparison = "more" if is_more else "less"

    # 2. Question templates
    question_templates = [
        f"{new_value} is what percent {comparison} than {original}?",
        f"By what percentage is {new_value} {comparison} than {original}?",
        f"Find the percent by which {new_value} is {comparison} than {original}.",
        f"How much percent {comparison} is {new_value} compared to {original}?",
    ]
    question = random.choice(question_templates)

    # 3. Distractor generation
    def generate_distractors(correct):
        distractors = set()
        while len(distractors) < 3:
            offset = random.uniform(-15, 15)
            val = round(correct + offset, 2)
            if val != correct:
                distractors.add(str(val))
        return list(distractors)

    choices = generate_distractors(correct) + [correct_str]
    random.shuffle(choices)

    # 4. Explanation steps
    explanation = [
        f"Step 1: Identify the original and new values.",
        f"   → Original value = {original}",
        f"   → New value = {new_value}",
        (
            f"Step 2: Find the difference → calculator.subtract({new_value}, {original})"
            if is_more
            else f"Step 2: Find the difference → calculator.subtract({original}, {new_value})"
        ),
        "Step 3: Divide the difference by the original value.",
        f"Step 4: calculator.divide(difference, {original})",
        "Step 5: Multiply the result by 100 → calculator.multiply(calculator.divide(...), 100)",
        f"Step 6: This gives the percentage change between the two values.",
        f" Final Answer: {correct}%",
    ]

    return {
        "question": question,
        "correct_answer": correct_str,
        "choices": choices,
        "difficulty": difficulty,
        "topic": "percentage",
        "explanation": "\n".join(explanation),
    }


def generate_gcf_question(difficulty: Literal["easy", "medium", "pro"]) -> dict:
    # 1. Range by difficulty
    ranges = {
        "easy": (10, 50),
        "medium": (30, 100),
        "pro": (50, 200),
    }
    low, high = ranges[difficulty]
    a = random.randint(low, high)
    b = random.randint(low, high)
    while a == b:
        b = random.randint(low, high)

    correct = gcd(a, b)
    correct_str = str(correct)

    # 2. Question Templates
    instruction_templates = [
        "Find the greatest common factor of the given numbers:",
        "Determine the GCF of the two numbers below using prime factorization:",
        "Solve for the highest common factor (HCF):",
        "What is the GCF of the following pair of numbers?",
    ]
    input_templates = [
        f"What is the GCF of {a} and {b}?",
        f"Find the greatest common divisor of {a} and {b}.",
        f"Determine the highest number that divides both {a} and {b}.",
        f"Calculate the GCF of the numbers: {a}, {b}.",
    ]
    question = random.choice(input_templates)
    instruction = random.choice(instruction_templates)

    # 3. Distractor logic
    def distractors(correct, a, b):
        options = set()
        while len(options) < 3:
            offset = random.choice([-1, 1, 2, 3, 4])
            guess = correct + offset
            if 1 < guess <= min(a, b) and guess != correct:
                options.add(str(guess))
        return list(options)

    options = distractors(correct, a, b) + [correct_str]
    random.shuffle(options)

    # 4. Prime factorization
    pf_a = prime_factors(a)
    pf_b = prime_factors(b)
    common = sorted(set(pf_a).intersection(set(pf_b)))

    explanation = [
        f"Step 1: Given numbers → {a} and {b}",
        f"Step 2: Prime factors of {a} → {pf_a}",
        f"Step 3: Prime factors of {b} → {pf_b}",
    ]
    if common:
        explanation.append(f"Step 4: Common prime factors → {common}")
        explanation.append("Step 5: Multiply them to get the GCF")
        explanation.append(f"        → calculator.gcf({a}, {b}) = {correct}")
    else:
        explanation.append("Step 4: No common prime factors → GCF is 1")
        explanation.append(f"        → calculator.gcf({a}, {b}) = 1")

    explanation.append(f" Final Answer: {correct}")

    return {
        "question": question,
        "correct_answer": correct_str,
        "choices": options,
        "difficulty": difficulty,
        "topic": "number_theory_gcf",
        "explanation": "\n".join(explanation),
    }


def generate_lcm_question(difficulty: Literal["easy", "medium", "pro"]) -> dict:
    ranges = {
        "easy": (10, 30),
        "medium": (20, 60),
        "pro": (40, 100),
    }
    low, high = ranges[difficulty]
    a = random.randint(low, high)
    b = random.randint(low, high)
    while a == b:
        b = random.randint(low, high)

    def lcm(x, y):
        from math import gcd

        return x * y // gcd(x, y)

    correct = lcm(a, b)
    correct_str = str(correct)

    instruction_templates = [
        "Find the least common multiple of the given numbers:",
        "Determine the LCM of the two numbers below using prime factorization:",
        "Solve for the lowest common multiple (LCM):",
        "What is the LCM of the following pair of numbers?",
    ]
    input_templates = [
        f"What is the LCM of {a} and {b}?",
        f"Find the least number that is a multiple of both {a} and {b}.",
        f"Determine the LCM of the numbers {a} and {b}.",
        f"Calculate the lowest common multiple of {a}, {b}.",
    ]

    instruction = random.choice(instruction_templates)
    input_text = random.choice(input_templates)

    # Generate distractors
    def generate_distractors(correct, a, b):
        wrong = set()
        while len(wrong) < 3:
            offset = random.choice([-10, -5, -1, 1, 5, 10])
            fake = correct + offset
            if fake > max(a, b) and fake != correct:
                wrong.add(str(fake))
        return list(wrong)

    options = generate_distractors(correct, a, b) + [correct_str]
    random.shuffle(options)

    # Prime factor breakdown
    pf_a = prime_factors(a)
    pf_b = prime_factors(b)
    count_a = Counter(pf_a)
    count_b = Counter(pf_b)
    all_primes = sorted(set(count_a) | set(count_b))
    combined = {p: max(count_a.get(p, 0), count_b.get(p, 0)) for p in all_primes}

    # Build explanation
    explanation = [
        f"Step 1: We are given two numbers: {a} and {b}.",
        "Step 2: The LCM (least common multiple) is the smallest number divisible by both.",
        f"Step 3: Prime factorization of {a} → {pf_a}",
        f"Step 4: Prime factorization of {b} → {pf_b}",
        "Step 5: Take the highest power of each prime factor involved:",
    ]
    for p in all_primes:
        explanation.append(
            f"   - Prime {p}: max({count_a.get(p, 0)}, {count_b.get(p, 0)}) = {combined[p]}"
        )

    explanation.extend(
        [
            "Step 6: Multiply these prime factors together to get the LCM.",
            f"Step 7: Final result is `calculator.lcm({a}, {b})`.",
            f" Final Answer: {correct}",
        ]
    )

    return {
        "question": input_text,
        "correct_answer": correct_str,
        "choices": options,
        "difficulty": difficulty,
        "topic": "number_theory_lcm",
        "explanation": "\n".join(explanation),
    }


