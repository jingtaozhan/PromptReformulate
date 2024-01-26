import re
import logging

logger = logging.getLogger(__name__)



class AdvancedString(str):
    def format(self, **kwargs):
        def eval_placeholder(match):
            # Extract the expression and format specification
            inner_text = match.group(1)
            expr, _, format_spec = inner_text.partition(':')

            # Evaluate the expression
            try:
                result = eval(expr, {}, kwargs)
            except Exception as e:
                return f"[Error: {e}]"

            # Apply format specification if present
            if format_spec:
                format_str = "{:" + format_spec + "}"
                return format_str.format(result)
            else:
                return str(result)

        # Regular expression to find placeholders
        pattern = re.compile(r"{(.*?)}")

        # Replace placeholders with evaluated results
        return pattern.sub(eval_placeholder, self)
    
    
ORIGINAL_PROMPT_TEMPLATE = """The original prompt is: "{prompt}"."""

INFO_TEMPLATE = """A text-to-image generation system transforms text prompts into visual images. The effectiveness of this conversion depends on the prompt. The original prompt leads to images with prompt-image similarity of {init_clip}, aesthetic quality of {init_aesthetic}, and overall quality of {init_overall}. To improve these metrics, new images are generated based on a revised prompt. After evaluating the new images for the initial prompt, the updated scores are: prompt-image similarity of {new_clip}, aesthetic quality of {new_aesthetic}, and overall quality of {new_overall}. The revised prompt, {length_compare} the original prompt, is structured into {new_phrase_cnt} phrases, each separated by a comma."""


PREDICTION_TEMPLATE = """Considering the given information, the revised prompt should be:"""


TRAIN_INPUT_TEMPLATE = INFO_TEMPLATE + " " + ORIGINAL_PROMPT_TEMPLATE + " " + PREDICTION_TEMPLATE

TRAIN_OUTPUT_TEMPLATE = "{rewritten_prompt} "

EVAL_INPUT_TEMPLATE = INFO_TEMPLATE + " " + ORIGINAL_PROMPT_TEMPLATE + " " + PREDICTION_TEMPLATE

EVAL_PREFIXED_INPUT_TEMPLATE = INFO_TEMPLATE + " " + ORIGINAL_PROMPT_TEMPLATE + " " + PREDICTION_TEMPLATE + """ {prompt.strip(". ")},"""

EVAL_OUTPUT_TEMPLATE = "{rewritten_prompt} "


def _test():
    # pseudo data
    data = {
        "prompt": 'original prompt',
        "rewritten_prompt": 'new prompt', 
        "init_clip": 2,
        "init_aesthetic": 3,
        "init_overall": 4,
        "new_clip": 1,
        "new_aesthetic": 8,
        "new_overall": 4,
        "length_compare": "longer than", 
        "new_phrase_cnt": 2,
    }
    print(TRAIN_INPUT_TEMPLATE.format(**data))
    print("\n################\n")
    print(TRAIN_OUTPUT_TEMPLATE.format(**data))
    print("\n################\n")
    print(EVAL_INPUT_TEMPLATE.format(**data))
