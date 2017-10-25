import util
import pdb

captions, questions, answers, vocab = util.load_data('/ais/gobi5/atef/VQA-Memnet/birds', 10.0)
sentence_size, vocab_size, word_idx = util.calculate_parameter_values(captions=captions, questions=questions, vocab=vocab)
captions_vec, questions_vec, answers_vec = util.vectorize_data(captions=captions, questions=questions, answers=answers,
                                                              sentence_size=sentence_size, word_idx=word_idx)

S, Q, A = util.generate_s_q_a(questions=questions_vec, answers=answers_vec, limit_to_species=False)


pdb.set_trace()
x = 1