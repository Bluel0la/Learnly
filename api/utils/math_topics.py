from api.utils.maths_generators import generate_addition_question
from api.utils.maths_generators import generate_subtraction_question
from api.utils.maths_generators import generate_multiplication_question
from api.utils.maths_generators import generate_division_question
from api.utils.maths_generators import generate_decimal_addition_question
from api.utils.maths_generators import generate_decimal_subtraction_question
from api.utils.maths_generators import generate_decimal_multiplication_question
from api.utils.maths_generators import generate_decimal_division_question
from api.utils.maths_generators import generate_fraction_addition_question
from api.utils.maths_generators import generate_fraction_subtraction_question
from api.utils.maths_generators import generate_fraction_multiplication_question
from api.utils.maths_generators import generate_fraction_division_question
from api.utils.maths_generators import generate_pemdas_question
from api.utils.maths_generators import generate_percentage_question
from api.utils.maths_generators import generate_percentage_relationship_question
from api.utils.maths_generators import generate_percentage_comparison_question
from api.utils.maths_generators import generate_gcf_question
from api.utils.maths_generators import generate_lcm_question

TOPIC_GENERATORS = {
    "addition": generate_addition_question,
    "subtraction": generate_subtraction_question,
    "multiplication": generate_multiplication_question,
    "division": generate_division_question,
    "decimal_addition": generate_decimal_addition_question,
    "decimal_subtraction": generate_decimal_subtraction_question,
    "decimal_multiplication": generate_decimal_multiplication_question,
    "decimal_division": generate_decimal_division_question,
    "fraction_addition": generate_fraction_addition_question,
    "fraction_subtraction": generate_fraction_subtraction_question,
    "fraction_multiplication": generate_fraction_multiplication_question,
    "fraction_division": generate_fraction_division_question,
    "pemdas": generate_pemdas_question,
    "percentage_of_number": generate_percentage_question,
    "percentage_relationship": generate_percentage_relationship_question,
    "percentage_comparison": generate_percentage_comparison_question,
    "number_theory_gcf": generate_gcf_question,
    "number_theory_lcm": generate_lcm_question,
}
