# This script preprocesses the DuReader dataset to the SQuAD format.

import json
import time
from pprint import pprint
import argparse


def main(fn, skip_questions_with_no_answers, toy_output, num_samples):
    print('Converting DuReader dataset to SQuAD format ...')
    t = time.time()

    # Initialize variables
    output = {}
    output['data'] = []
    num_success = 0
    num_failures = 0
    num_skipped = 0
    total_questions = 0

    # Loop through each question
    for json_obj in open(fn, 'r'):

        total_questions += 1

        # Load DuReader question-set
        duReader = json.loads(json_obj)

        # Initialize output dictionaries
        answer = {}
        qas = {}
        paragraph = {}
        data = {}
        input_para = ''

        if len(duReader['answer_docs']) != 0:  # Case 1: Answer exists

            # Parse DuReader question-set
            try:
                answer_doc_idx = duReader['answer_docs'][0]
                dr_answer_start = duReader['answer_spans'][0][0]
                dr_answer_end = duReader['answer_spans'][0][1]
                answer_doc = duReader['documents'][answer_doc_idx]
                answer_para_idx = answer_doc['most_related_para']

                # Get answer start index and input_para
                answer_start = 0

                for i, para in enumerate(answer_doc['segmented_paragraphs']):
                    concat_para = ''.join(para)

                    if i < answer_para_idx:  # Add length to answer_start
                        answer_start += len(concat_para)

                    if i == answer_para_idx:  # Complete answer_start.
                        generated_answer = ''.join(
                            para[dr_answer_start:dr_answer_end + 1])
                        para_before_answer = ''.join(para[:dr_answer_start])
                        answer_start += len(para_before_answer)

                    input_para += concat_para

                fake_answer = duReader['fake_answers'][0]

                # Check parsed results
                assert generated_answer == fake_answer, f'Answer span not equal to fake answer. Answer span is "{generated_answer}". Fake answer is "{fake_answer}".'
                answer_from_input = input_para[answer_start:answer_start + len(
                    generated_answer)]
                assert answer_from_input == generated_answer, f'Error with indexing. Answer span not matching with input para. Input para answer: {answer_from_input}.'
                assert answer_from_input == fake_answer, 'Input para answer span does not match.'

                # Update output
                answer['answer_start'] = answer_start
                answer['text'] = fake_answer
                qas['answers'] = [answer]
                qas['is_impossible'] = False

            except:
                num_failures += 1
                continue

        else:  # Case 2: Answer does not exist.

            if skip_questions_with_no_answers:
                num_skipped += 1
                continue

            else:
                # Get input para (concat all paragraphs)
                for doc in duReader['documents']:
                    input_para += ''.join(doc['paragraphs'])

                # Update output
                qas['answers'] = []
                qas['is_impossible'] = True

        # Update output
        qas['id'] = duReader['question_id']
        qas['question'] = duReader['question']

        paragraph['context'] = input_para
        paragraph['qas'] = [qas]

        data['paragraphs'] = [paragraph]
        data['title'] = 'N.A.'

        output['data'].append(data)
        num_success += 1

        # Exit early if creating toy set
        if toy_output and (num_success >= num_samples):
            break

    # Save to JSON file
    prefix = '/'.join(fn.split('/')[:-1])
    suffix = fn.split('/')[-1]

    if toy_output:
        tag = f'/squad_toy{num_samples}_'
    elif skip_questions_with_no_answers:
        tag = '/squad_withAns_'
    else:
        tag = '/squad_'

    output_fn = prefix + tag + suffix

    with open(output_fn, 'w') as f:
        json.dump(output, f)

    print(f'Done! Tried to convert {total_questions} questions.'
          f'\n{num_success} questions successfully converted.'
          f'\n{num_skipped} questions skipped due to having no answers.'
          f'\n{num_failures} questions failed to convert.'
          f'\nConversion took {time.time()-t:.2f}s.')


if __name__ == '__main__':
    # # Replace filename here
    # fn = 'duReader_preprocessed/trainset/search.train.json'
    #
    # # Set settings here
    # skip_questions_with_no_answers = True
    # toy_output = True
    # num_samples = 1000  # only valid if toy_output is True

    parser = argparse.ArgumentParser(
        description='Run preprocessing on DuReader Dataset')
    parser.add_argument('dureader_input',
                        help='Path to duReader json')
    parser.add_argument('--keep_no_answer',
                        dest='skip_questions_with_no_answers',
                        action="store_false",
                        default=True,
                        help='Do not skip questions that do not have an answer')
    parser.add_argument('--no_toy_output',
                        dest='toy_output',
                        action="store_false",
                        default=True,
                        help='Limit the number of samples to num_samples or not')
    parser.add_argument('--num_samples',
                        action="store",
                        type=int,
                        default=1000,
                        help='Number of samples to get')

    args = parser.parse_args()

    fn = args.dureader_input
    skip_questions_with_no_answers = args.skip_questions_with_no_answers
    toy_output = args.toy_output
    num_samples = args.num_samples

    main(fn, skip_questions_with_no_answers, toy_output, num_samples)
