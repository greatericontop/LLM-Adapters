
_instr = 'Below is an instruction that describes a question. Write a response that appropriately completes the question. Your response must end with your final answer. Follow the example given.'
_ex_question = 'Alice buys 3 pizzas for $10 each. How much did she spend in total?'

# gsm50 version
#_ex_answer = ''
# math10k version
_ex_answer = 'Alice buys 3 pizzas. Each one costs $10, so the total cost is 3 * 10 = 30. The answer is 30.'


def get_eval_prompt(instruction: str) -> str:
    return (
        f'{_instr}\n'
        f'\n'
        f'### Example Question:\n'
        f'{_ex_question}\n'
        f'\n'
        f'### Example Answer:\n'
        f'{_ex_answer}\n'
        f'\n'
        f'### Question:\n'
        f'{instruction}\n'
        f'\n'
        f'### Response:\n'
    )


def get_finetune_prompt(instruction: str, output: str) -> str:
    return (
        f'{_instr}\n'
        f'\n'
        f'### Example Question:\n'
        f'{_ex_question}\n'
        f'\n'
        f'### Example Answer:\n'
        f'{_ex_answer}\n'
        f'\n'
        f'### Question:\n'
        f'{instruction}\n'
        f'\n'
        f'### Response:\n'
        f'{output}'
    )
