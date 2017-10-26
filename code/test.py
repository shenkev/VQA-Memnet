import util
import pdb

captions, questions, answers, vocab = util.load_data('/Users/atef/VQA-Memnet/birds', 10.0)
sentence_size, vocab_size, word_idx = util.calculate_parameter_values(captions=captions, questions=questions, vocab=vocab)
captions_vec, questions_vec, answers_vec = util.vectorize_data(captions=captions, questions=questions, answers=answers,
                                                              sentence_size=sentence_size, word_idx=word_idx)

S, Q, A = util.generate_s_q_a(questions=questions_vec, answers=answers_vec, limit_to_species=False)

data = (S, Q, A)
train_set, train_batches, test_set, test_batches = util.batch_data(data=data, batch_size=16, test_size=0.2)


pdb.set_trace()
x = 1