
import datetime
import json
import sys

filename = sys.argv[1]
save_file = 'manual_grade_logs.txt'

print('[only doing 8 questions]')
input('...')

with open(filename, 'r') as f:
    responses = json.load(f)

running_score = 0
for i, data in enumerate(responses[:8]):
    print('\033[H\033[2J\033[3J')
    print('----------------------------------------------------------------------------------------------------')
    print('----------------------------------------------------------------------------------------------------')
    print(f'#{i} (running total {running_score}/{i})')
    print('----------------------------------------------------------------------------------------------------')
    print('----------------------------------------------------------------------------------------------------')
    print(data['raw_output'])
    print('----------------------------------------------------------------------------------------------------')
    print('----------------------------------------------------------------------------------------------------')
    score = int(input('enter score (0=bad, 1=good with issues, or 2=good):'))
    running_score += score

with open(save_file, 'a') as f:
    f.write(f'{datetime.datetime.now().isoformat()}:    {filename=}    {running_score}/{len(responses)}\n')
