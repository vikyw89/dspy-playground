import os
import random
import dspy
import logging

import dspy.evaluate

logger = logging.getLogger(__name__)


async def main():
    llm = dspy.OpenAI(
        model="meta-llama/llama-3.1-70b-instruct",
        base_url=os.getenv("OPENROUTER_BASE_URL"),
        api_key=os.getenv("OPENROUTER_API_KEY"),
        temperature=0.2,
    )

    dspy.configure(lm=llm)

    from src.components.training import load_training_data_as_example

    trainset = load_training_data_as_example(name="20240827-training.csv")
    from dspy.evaluate import Evaluate
    score_accuracy = dspy.evaluate.metrics.answer_passage_match
    evaluator = Evaluate(devset=trainset, num_threads=1, display_progress=True)
    class Translation(dspy.Signature):
        """Translate a text from Vietnamese to Korean"""
        input = dspy.InputField(desc="Vietnamese text to translate")
        translation = dspy.OutputField(desc="Korean translation")

    class ScoNeCoT(dspy.Module):
        def __init__(self):
            super().__init__()
            self.generate_answer = dspy.ChainOfThought(Translation)

        def forward(self, input):
            return self.generate_answer(input=input.vn)
        
    cot_zeroshot = ScoNeCoT()
    evaluator(cot_zeroshot, metric=score_accuracy)
    
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        filemode="a+",
        filename="debug.log",
    )
    import asyncio

    asyncio.run(main())
