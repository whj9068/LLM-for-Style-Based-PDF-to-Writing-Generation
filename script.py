from unsloth.chat_templates import get_chat_template

from datasets import load_dataset, Dataset
import random
from typing import Optional, Dict


def create_style_transfer_prompts(
    dataset,
    content_col: str,
    style_col: str,
    style_descriptions: Optional[Dict[str, str]] = None,
    num_samples: int = 5
):
    conversations = []

    for idx in range(len(dataset)):
        # Get random style examples
        all_examples = list(dataset)
        other_examples = [ex for i, ex in enumerate(all_examples) if i != idx]
        sample_examples = random.sample(other_examples, min(num_samples, len(other_examples)))
        style_samples = "\n".join([f"Example {i+1}:\n{ex[style_col]}"
                                 for i, ex in enumerate(sample_examples)])

        # Format style descriptions
        style_desc = ""
        if style_descriptions:
            style_desc = "\nProvided style characteristics (consider as supplementary):\n"
            style_desc += "\n".join([f"- {k}: {v}" for k, v in style_descriptions.items()])

        system_message = {
            "role": "system",
            "content": """You are an expert in analyzing and adapting writing styles.

Your process for style transfer:
1. Analyze style patterns from examples (vocabulary, structure, tone, devices)
2. Consider provided style descriptions as supplementary guidance
3. Combine analyses to identify key style elements
4. Transform the text while preserving core meaning and applying style patterns

Transform the input text to match the style demonstrated in the examples while maintaining the original meaning and ensuring natural flow."""
        }
        system_message1 = {
            "role": "system",
            "content": """You are an expert in analyzing and adapting writing styles.

Your process for style transfer:
1. Analyze style patterns from examples (vocabulary, structure, tone, devices)
2. Consider provided style descriptions as supplementary guidance
3. Combine analyses to identify key style elements
4. Create text with the same style patterns

Create text that matches the style demonstrated in the examples while ensuring natural flow."""
        }

        conversation = {
            "conversations": [
                system_message1,
                {
                    "role": "user",
                    "content": f"""Context text to be transformed:
"{dataset[idx][content_col]}"

Sample writing with desired writing style:
{style_samples}{style_desc}"""
                },
                {
                    "role": "assistant",
                    "content": dataset[idx][style_col]
                }
            ]
        }
        conversations.append(conversation)

    return conversations

def format_for_llama(examples, tokenizer):
    """
    Apply the LLaMA template to the formatted conversations
    """
    texts = []
    for example in examples:
        text = tokenizer.apply_chat_template(
            example["conversations"],
            tokenize=False,
            add_generation_prompt=False
        )
        texts.append(text)

    return Dataset.from_dict({"text": texts})

# Example usage
ds = load_dataset("jassiyu/poetry-modern")
train_ds = ds["train"]

# Example style descriptions (optional)
style_descriptions = {
    "Word_Length": "The text uses predominantly shorter words (one or two syllables) to maintain simplicity and accessibility, with occasional longer words to add depth and variation.",
    "Syllabic_Word": "The text balances monosyllabic words for clarity and simplicity with polysyllabic words to evoke deeper meaning and emotional resonance, enhancing the relatable tone.",
    "Emotion": "The emotional tone of the text is warm and uplifting, focusing on positive emotions like joy, peace, and gratitude, while fostering a sense of connection and empathy.",
    "Rhetoric": "The text employs relatable metaphors and analogies (e.g., rain, sunshine, breathing) to make abstract concepts tangible. The sentence structure is straightforward, prioritizing readability while subtly weaving in figurative language for emotional impact."
}

# Format conversations
formatted_convos = create_style_transfer_prompts(
    dataset=train_ds,
    content_col="shakespearean",
    style_col="modern",
    style_descriptions= None,
    num_samples=5
)

# Apply LLaMA template
tokenizer = get_chat_template(
    tokenizer,
    chat_template="llama-3.1"
)
final_dataset = format_for_llama(formatted_convos, tokenizer)

# Example usage for any style transfer dataset
"""
# For any style transfer dataset:
style_descriptions = {
    "Word_Length": "your_analysis",
    "Syllabic_Word": "your_analysis",
    "Emotion": "your_analysis",
    "Rhetoric": "your_analysis"
}

formatted_convos = create_comprehensive_style_prompts(
    dataset=your_dataset,
    content_col="original_text",
    style_col="styled_text",
    style_descriptions=style_descriptions,  # Optional
    num_samples=5
)
"""