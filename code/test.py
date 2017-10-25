import util
import pdb

captions, questions, answers, vocab = util.load_data('/ais/gobi5/atef/VQA-Memnet/birds', 10.0)
memory_size, sentence_size, vocab_size, word_idx = util.calculate_parameter_values(captions=captions, questions=questions,
                                                                                   limit_to_species=False, memory_limit=None,
                                                                                   sentence_limit=None, vocab=vocab)

pdb.set_trace()
x = 1