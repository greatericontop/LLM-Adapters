
#_instr = 'Below is an instruction that describes a question. Write a response that appropriately completes the question. Your response must end with your final answer. Follow the example given.'
_instr = 'Answer the question below. Make sure to explain your reasoning. An example question and answer are provided.'
#_instr = 'Answer the questions given by the user. Make sure to explain your reasoning.'
#_ex_question = 'Alice buys 3 pizzas for $10 each and 2 donuts for $2 each. How much did she spend in total?'

# gsm50 version
#_ex_answer = ''
# math10k version
#_ex_answer = 'Alice buys 3 pizzas. Each one costs $10, so the total cost of pizzas is 3 * 10 = 30. Alice buys 2 donuts. Each one costs $2, so the total cost of donuts is 2 * 2 = 4. The total cost is 30 + 4 = 34. The answer is 34.'

_examples = [
    'Tom has only been getting 6 hours of sleep a day.  He increases that by 1/3.  How many hours of sleep does he get per night?',
    '''To increase his sleep by 1/3, we need to find 1/3 of his current sleep time and add it to his current sleep time.
1/3 of 6 hours is (1/3)*6 = 2 hours.
Adding 2 hours to his current sleep time of 6 hours gives us:
6 + 2 = 8 hours.
Therefore, Tom now gets 8 hours of sleep per night.
The answer in Arabic numerals is 8.''',

    'Jimmy has a collection of 5 action figures. Each figure is worth $15, except for one which is worth $20. He decided to sell his collection. To do it fast he decided to sell each of them for $5 less than their value. How much will Jimmy earn if he sells all the figures?',
    '''Step 1: Calculate the total value of the collection
- 4 action figures are worth $15 each: 4 x $15 = $60
- 1 action figure is worth $20
- Total value = $60 + $20 = $80
Step 2: Calculate the selling price of each action figure
- Each figure will be sold for $5 less than its value
- For the 4 figures worth $15 each: $15 - $5 = $10
- For the figure worth $20: $20 - $5 = $15
Step 3: Calculate the total earning from selling the collection
- Jimmy has 5 action figures to sell
- He will earn $10 for 4 of them: 4 x $10 = $40
- He will earn $15 for 1 of them
- Total earning = $40 + $15 = $55
Therefore, Jimmy will earn $55 if he sells all the figures. The answer is 55.''',

    'There were 28 students inside a bus before stopping at an intersection. After stopping at an intersection, there were 58 students on the bus. What\'s 40% of the number of students who entered the bus at the intermediate stop?',
    '''The number of students who entered the bus at the intermediate stop is:
58 - 28 = 30
To find 40% of 30, we can multiply 30 by 0.4:
30 x 0.4 = 12
Therefore, 40% of the number of students who entered the bus at the intermediate stop is 12.'''
]


def get_eval_prompt(instruction: str) -> str:
    return f'''
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
{_instr}
<|eot_id|>
<|start_header_id|>user<|end_header_id|>
{_examples[0]}
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
{_examples[1]}
<|eot_id|>
<|start_header_id|>user<|end_header_id|>
{_examples[2]}
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
{_examples[3]}
<|eot_id|>
<|start_header_id|>user<|end_header_id|>
{_examples[4]}
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
{_examples[5]}
<|eot_id|>
<|start_header_id|>user<|end_header_id|>
{instruction}
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
'''


def get_finetune_prompt(instruction: str, output: str) -> str:
    return f'''
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
{_instr}
<|eot_id|>
<|start_header_id|>user<|end_header_id|>
{_examples[0]}
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
{_examples[1]}
<|eot_id|>
<|start_header_id|>user<|end_header_id|>
{_examples[2]}
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
{_examples[3]}
<|eot_id|>
<|start_header_id|>user<|end_header_id|>
{_examples[4]}
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
{_examples[5]}
<|eot_id|>
<|start_header_id|>user<|end_header_id|>
{instruction}
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
{output}
<|eot_id|><|end_of_text|>
'''
