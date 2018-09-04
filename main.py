from config import config, parseArgs
from model import MACnet
import tensorflow as tf
import time
import sys
import json
import gzip
import pickle
import numpy as np
import nltk
nltk.download('stopwords')
nltk.download('maxent_ne_chunker')
nltk.download('words')
import os.path

from keras.preprocessing.text import Tokenizer


def build_label_to_offset_map(snippets):
    num_snippets = len(snippets)
    one_snippet_len = len(snippets[0])
    label = 0
    label_map = dict()
    # for gram in [1, 2, 3, 4]:
    for gram in [1, 2, 3]: # using only trigrams due to bug in preprocess notebook
        for i in range(num_snippets):
            for j in range(one_snippet_len - gram+1):
                label_map[label] = (i, j, j+gram)
                label += 1
    return label_map

def config_tf():
    NUM_PARALLEL_EXEC_UNITS = 6
    config = tf.ConfigProto(intra_op_parallelism_threads=NUM_PARALLEL_EXEC_UNITS, inter_op_parallelism_threads=2,
                            allow_soft_placement=True, device_count={'CPU': NUM_PARALLEL_EXEC_UNITS})
    import os
    os.environ["OMP_NUM_THREADS"] = "NUM_PARALLEL_EXEC_UNITS"
    os.environ["KMP_BLOCKTIME"] = "30"
    os.environ["KMP_SETTINGS"] = "1"
    os.environ["KMP_AFFINITY"] = "granularity=fine,verbose,compact,1,0"
    return config

def get_pos_for_snippets_dataset(dataset_snippets, index_to_word):
    global INDEX_TO_WORD
    INDEX_TO_WORD = index_to_word
    return list(map(get_pos_for_snippets, dataset_snippets))

def get_ner_for_snippets_dataset(dataset_snippets_pos_tagged):
    return list(map(get_ner_for_snippets, dataset_snippets_pos_tagged))

INDEX_TO_WORD = None
def get_pos_for_snippets(snippets):
    all_snippets_pos_tags = []
    for snippet in snippets:
        snippet_words = []
        for tok in snippet:
            if tok != 0: # skip padding
                snippet_words.append(INDEX_TO_WORD[tok])
        snippet_pos_tags = nltk.pos_tag(snippet_words)
        pos_padding = (len(snippet) - len(snippet_words)) * ["NO_POS"]
        snippet_pos_tags = snippet_pos_tags + pos_padding
        all_snippets_pos_tags.append(snippet_pos_tags)
    return all_snippets_pos_tags

def get_ner_for_snippets(pos_tagged_snippets):
    all_snippets_ner_tags = []
    for snippet in pos_tagged_snippets:
        snippet_tagged_words = []
        for tok in snippet:
            if tok != "NO_POS":  # skip padding
                snippet_tagged_words.append(tok)
        snippet_ner_tags = nltk.ne_chunk(snippet_tagged_words, binary=True)
        snippet_ner_tags_mask = []
        for chunk in snippet_ner_tags:
            if hasattr(chunk, 'label') and chunk.label() == "NE":
                snippet_ner_tags_mask.append(False) # keep label
            else:
                snippet_ner_tags_mask.append(True) # mask the label
        ner_padding = (len(snippet) - len(snippet_tagged_words)) * [True]
        snippet_ner_tags_mask = snippet_ner_tags_mask + ner_padding
        all_snippets_ner_tags.append(snippet_ner_tags_mask)
    return all_snippets_ner_tags

def load_pos(file_name, snippets, index_to_word):
    if not os.path.isfile(file_name):
        print("file {} doesn't exists, compute pos".format(file_name))
        config.makedirs("pos")
        pos = get_pos_for_snippets_dataset(snippets, index_to_word)
        with open(file_name, 'wb+') as f:
            pickle.dump(pos, f)
    else:
        print("file {} exists, use pre computed pos".format(file_name))
        with open(file_name, 'rb') as f:
            pos = pickle.load(f)
    return pos

def load_ner(file_name, pos):
    if not os.path.isfile(file_name):
        print("file {} doesn't exists, compute ner".format(file_name))
        config.makedirs("ner")
        ner = get_ner_for_snippets_dataset(pos)
        with open(file_name, 'wb+') as f:
            pickle.dump(ner, f)
    else:
        print("file {} exists, use pre computed ner".format(file_name))
        with open(file_name, 'rb') as f:
            ner = pickle.load(f)

    # patching missing one padding
    for entry in ner:
        for s in entry:
            # print (s)
            s.append(True)
    return ner

def get_term_freq(q_snippets, label_map):
        token_hist = dict()
        for snippet in q_snippets:
            for i in range(len(snippet)):
                # unigram
                key = snippet[i]
                if key in token_hist:
                    token_hist[key] += 1
                else:
                    token_hist[key] = 1

                if i > 0: # bigram
                    key = tuple(snippet[i - 1:i + 1])
                    if key in token_hist:
                        token_hist[key] += 1
                    else:
                        token_hist[key] = 1
                if i > 1:  # trigram
                    key = tuple(snippet[i - 2:i + 1])
                    if key in token_hist:
                        token_hist[key] += 1
                    else:
                        token_hist[key] = 1
                if i > 2: # fourgram
                    key = tuple(snippet[i - 3:i + 1])
                    if key in token_hist:
                        token_hist[key] += 1
                    else:
                        token_hist[key] = 1
        hist = []
        for label in label_map:
            snippet_index, start_index, end_index = label_map[label]
            term = tuple(q_snippets[snippet_index][start_index:end_index])
            if len(term) == 1:
                term = term[0]
            if term not in token_hist:
                print (term, "ERROR")
            else:
                hist.append(token_hist[term])
        return hist

def contains(small, big):
    for i in range(len(big)-len(small)+1):
        for j in range(len(small)):
            if big[i+j] != small[j]:
                break
        else:
            return True
    return False

def get_filter_tok(word_to_index):
    from nltk.corpus import stopwords
    _filter_tok = stopwords.words("english")
    _filter_tok += [".", ",", "?", "...", "-", ":", ";", "·"]

    filter_tok = [[0], [0, 0], [0, 0, 0], [0, 0, 0, 0]] # padding
    for tok in _filter_tok:
        tokenized = [word_to_index[w] for w in tok.split() if w in word_to_index]
        if tokenized != []:
            filter_tok.append(tokenized)
    return filter_tok

def load_candidates_mask(snippets_dataset, label_map, filter_tok, index_to_word, questions, keep_top_k, is_train=False, prefix=""):
    file_name = "cand_mask/{prefix}_cand_mask_k_{k}.pkl".format(prefix=prefix, k=keep_top_k)
    print("Getting top {} candidates".format(keep_top_k))
    if not os.path.isfile(file_name):
        print("file {} doesn't exists, compute candidates".format(file_name))
        cand_masks = get_candidates_mask(snippets_dataset, label_map, filter_tok, index_to_word, questions, keep_top_k, is_train)
        with open(file_name, 'wb+') as f:
            pickle.dump(cand_masks, f)
    else:
        print("file {} exists, use pre computed candidates".format(file_name))
        with open(file_name, 'rb') as f:
            cand_masks= pickle.load(f)
    return cand_masks

def load_candidates_mask_filter_terms_from_question(snippets_dataset, label_map, filter_tok, index_to_word, questions, keep_top_k, is_train=False, prefix=""):
    file_name = "cand_mask/{prefix}_cand_mask_filter_q_k_{k}.pkl".format(prefix=prefix, k=keep_top_k)
    print("Getting top {} candidates".format(keep_top_k))
    if not os.path.isfile(file_name):
        print("file {} doesn't exists, compute candidates".format(file_name))
        cand_masks = get_candidates_mask_filter_terms_in_question(snippets_dataset, label_map, filter_tok, index_to_word, questions, keep_top_k, is_train)
        with open(file_name, 'wb+') as f:
            pickle.dump(cand_masks, f)
    else:
        print("file {} exists, use pre computed candidates".format(file_name))
        with open(file_name, 'rb') as f:
            cand_masks= pickle.load(f)
    return cand_masks

def get_candidates_mask_filter_terms_in_question(snippets_dataset, label_map, filter_tok, index_to_word, questions, keep_top_k, is_train):
    dataset_masks = []
    from tqdm import tqdm
    for i in tqdm(range(len(snippets_dataset))):
        q_snippets = snippets_dataset[i]
        true_label = questions[i]["answers"] if "answers" in questions[i] else None
        term_freq = get_term_freq(q_snippets, label_map)
        sorted_term_freq_labels = sorted(range(len(term_freq)), key=lambda k: -term_freq[k])

        # build grams of the question for filtering terms that explicitly found in the question
        question_grams = set()
        q_tokens = questions[i]["question"]
        for j in range(len(q_tokens)):
            # unigram
            question_grams.add((q_tokens[j],))
            if j > 0:
                # bigram
                question_grams.add(tuple(q_tokens[j-1:j+1]))
            if j > 1:
                # trigram
                question_grams.add(tuple(q_tokens[j-2:j+1]))

        printed = set()
        mask = [True] * len(label_map)
        for j in range(len(sorted_term_freq_labels)):
            snippet_index, start_index, end_index = label_map[sorted_term_freq_labels[j]]
            answer = q_snippets[snippet_index][start_index:end_index]
            answer_tup = tuple(answer)
            if not any(contains(tok, answer) for tok in filter_tok) and \
                answer_tup not in question_grams: # filter stop words and tokens from question
                if answer_tup not in printed:
                    printed.add(answer_tup)
                    # answer = [index_to_word[k] for k in answer]
                    # answer = u" ".join(answer)
                    # print(answer, term_freq[sorted_term_freq_labels[i]])
                    if len(printed) > keep_top_k:
                        break
                mask[sorted_term_freq_labels[j]] = False
        # print("number of unique answers is {}, number of labels is {}".format(len(printed), len(mask) - sum(mask)))

        # sanity on the mask
        if true_label is not None:
            has_at_least_one_answer = False
            correct_answers_in_label= 0
            for j in range(len(mask)):
                if true_label[j] == 1 and mask[j] == 1:
                    # snippet_index, start_index, end_index = label_map[j]
                    # answer = q_snippets[snippet_index][start_index:end_index]
                    # answer = [index_to_word[k] for k in answer]
                    # answer = u" ".join(answer)
                    # print(answer, term_freq[i])
                    if is_train and not config.manual_cross_entropy:
                        mask[j] = 0
                elif true_label[j] == 1 and mask[j] == 0:
                    has_at_least_one_answer = True
                    correct_answers_in_label += 1

            if not has_at_least_one_answer:
                pass
                # print("NO ANSWERS")

            # print("odds are {} / {} = {}".format(correct_answers_in_label, len(mask) - sum(mask), float(correct_answers_in_label) / (len(mask) - sum(mask))))


        dataset_masks.append(mask)
    return dataset_masks


def get_candidates_mask(snippets_dataset, label_map, filter_tok, index_to_word, questions, keep_top_k, is_train):
    dataset_masks = []
    from tqdm import tqdm
    for i in tqdm(range(len(snippets_dataset))):
        q_snippets = snippets_dataset[i]
        true_label = questions[i]["answers"] if questions is not None else None
        term_freq = get_term_freq(q_snippets, label_map)
        sorted_term_freq_labels = sorted(range(len(term_freq)), key=lambda k: -term_freq[k])

        printed = set()
        mask = [True] * len(label_map)
        for i in range(len(sorted_term_freq_labels)):
            snippet_index, start_index, end_index = label_map[sorted_term_freq_labels[i]]
            answer = q_snippets[snippet_index][start_index:end_index]
            answer_tup = tuple(answer)
            if not any(contains(tok, answer) for tok in filter_tok): # filter stop words:
                if answer_tup not in printed:
                    printed.add(answer_tup)
                    # answer = [index_to_word[k] for k in answer]
                    # answer = u" ".join(answer)
                    # print(answer, term_freq[sorted_term_freq_labels[i]])
                    if len(printed) > keep_top_k:
                        break
                mask[sorted_term_freq_labels[i]] = False
        # print("number of unique answers is {}, number of labels is {}".format(len(printed), len(mask) - sum(mask)))

        # sanity on the mask
        if true_label is not None:
            has_at_least_one_answer = False
            correct_answers_in_label= 0
            for i in range(len(mask)):
                if true_label[i] == 1 and mask[i] == 1:
                    # snippet_index, start_index, end_index = label_map[i]
                    # answer = q_snippets[snippet_index][start_index:end_index]
                    # answer = [index_to_word[k] for k in answer]
                    # answer = u" ".join(answer)
                    # print(answer, term_freq[i])
                    if is_train and not config.manual_cross_entropy:
                        mask[i] = 0
                elif true_label[i] == 1 and mask[i] == 0:
                    has_at_least_one_answer = True
                    correct_answers_in_label += 1

            if not has_at_least_one_answer:
                pass
                # print("NO ANSWERS")

            # print("odds are {} / {} = {}".format(correct_answers_in_label, len(mask) - sum(mask), float(correct_answers_in_label) / (len(mask) - sum(mask))))


        dataset_masks.append(mask)
    return dataset_masks

def pre_process():

    embeddings_init_file = "Data/embedding_matrix.dat"
    embeddings_init = np.load(embeddings_init_file)
    # loading
    with open('Data/tokenizer.pickle', 'rb') as f:
        tokenizer = pickle.load(f)
    word_to_index = tokenizer.word_index
    with open('Data/inverse_word_token_map.pickle', 'rb') as f:
        index_to_word = pickle.load(f)
    index_to_word[0] = "<PADDING>"
    filter_tok = get_filter_tok(word_to_index)

    train_data_questions_file = "Data/" + "final_train_questions.json.gz"
    with gzip.open(train_data_questions_file, "rb") as f:
        train_data_questions_dict = json.load(f)
    train_data_snippets_file = "Data/" + "final_train_snippets.json.gz"
    with gzip.open(train_data_snippets_file, "rb") as f:
        train_data_snippets_dict = json.load(f)

    train_data_questions = [item for item in train_data_questions_dict if item is not None]
    train_data_snippets = [item["snippets"] for item in train_data_snippets_dict if item is not None]

    print("train len before{}".format(len(train_data_questions)))
    train_data_questions_new = []
    train_data_snippets_new = []
    for i in range(len(train_data_questions)):
        if np.count_nonzero(train_data_questions[i]["answers"]) > 0:
            train_data_questions_new.append(train_data_questions[i])
            train_data_snippets_new.append(train_data_snippets[i])
    train_data_questions = train_data_questions_new
    train_data_snippets = train_data_snippets_new
    print("train len after{}".format(len(train_data_questions)))

    print("build label to offset map")
    label_map = build_label_to_offset_map(train_data_snippets[0])  # all the snippets has the same shape, choose one

    # pos were saved after filtering
    train_pos = load_pos("pos/train_skip_pad.pkl", train_data_snippets, index_to_word)
    train_ner = load_ner("ner/train_skip_pad.pkl", train_pos)
    train_candidates_mask = load_candidates_mask_filter_terms_from_question(train_data_snippets, label_map, filter_tok, index_to_word, train_data_questions,
                                                 keep_top_k=config.top_k_candidates, is_train=True, prefix="train")
    train_data = {"questions": train_data_questions, "snippets": train_data_snippets, "pos": train_pos, "ner": train_ner,
                  "candidates_mask": train_candidates_mask}

    dev_data_questions_file = "Data/" + "final_dev_questions.json.gz"
    with gzip.open(dev_data_questions_file, "rb") as f:
        dev_data_questions_dict = json.load(f)
    dev_data_snippets_file = "Data/" + "final_dev_snippets.json.gz"
    with gzip.open(dev_data_snippets_file, "rb") as f:
        dev_data_snippets_dict = json.load(f)

    dev_data_questions = [item for item in dev_data_questions_dict if item is not None]
    dev_data_snippets = [item["snippets"] for item in dev_data_snippets_dict if item is not None]
    # =======================
    # dev_data_questions = dev_data_questions[:1]
    # dev_data_snippets = dev_data_snippets[:1]
    # =======================

    print("dev len before{}".format(len(dev_data_questions)))
    dev_data_questions_new = []
    dev_data_snippets_new = []
    for i in range(len(dev_data_questions)):
        if np.count_nonzero(np.asarray(dev_data_questions[i]["answers"])) > 0:
            dev_data_questions_new.append(dev_data_questions[i])
            dev_data_snippets_new.append(dev_data_snippets[i])
    dev_data_questions = dev_data_questions_new
    dev_data_snippets = dev_data_snippets_new
    print("dev len after{}".format(len(dev_data_questions)))
    dev_pos = load_pos("pos/dev_skip_pad.pkl", dev_data_snippets, index_to_word)
    dev_ner = load_ner("ner/dev_skip_pad.pkl", dev_pos)
    # =============================
    # print("build label to offset map")
    # label_map = build_label_to_offset_map(dev_data_snippets[0])  # all the snippets has the same shape, choose one
    # =============================
    dev_candidates_mask = load_candidates_mask_filter_terms_from_question(dev_data_snippets, label_map, filter_tok, index_to_word, dev_data_questions,
                                               keep_top_k=config.top_k_candidates, prefix="dev")
    dev_data = {"questions": dev_data_questions, "snippets": dev_data_snippets, "pos": dev_pos, "ner": dev_ner,
                "candidates_mask": dev_candidates_mask}

    # ====================================
    # train_data = dev_data
    #=====================================

    test_data_questions_file = "Data/" + "final_test_questions.json.gz"
    with gzip.open(test_data_questions_file, "rb") as f:
        test_data_questions_dict = json.load(f)
    test_data_snippets_file = "Data/" + "final_test_snippets.json.gz"
    with gzip.open(test_data_snippets_file, "rb") as f:
        test_data_snippets_dict = json.load(f)

    test_data_questions = [item for item in test_data_questions_dict if item is not None]
    test_data_snippets = [item["snippets"] for item in test_data_snippets_dict if item is not None]

    # =======================
    # test_data_questions = test_data_questions[:1]
    # test_data_snippets = test_data_snippets[:1]
    # =======================

    test_pos = load_pos("pos/test_skip_pad.pkl", test_data_snippets, index_to_word)
    test_ner = load_ner("ner/test_skip_pad.pkl", test_pos)
    test_candidates_mask = load_candidates_mask_filter_terms_from_question(test_data_snippets, label_map, filter_tok, index_to_word,
                                                                           test_data_questions, keep_top_k=config.top_k_candidates, prefix="test")
    test_data = {"questions": test_data_questions, "snippets": test_data_snippets, "pos": test_pos,
                 "ner": test_ner, "candidates_mask": test_candidates_mask}

    dataset = {"train": train_data, "val": dev_data, "test": test_data}

    config.max_possible_answers = len(train_data["questions"][0]["answers"])
    print(f"using max_possible_answers {config.max_possible_answers}")

    print(f"using classifier dims {config.outClassifierDims}")

    return embeddings_init, index_to_word, word_to_index, dataset, label_map


# Initializes savers (standard, optional exponential-moving-average and optional for subset of variables)
def setSavers(model):
    saver = tf.train.Saver(max_to_keep=config.weightsToKeep)

    subsetSaver = None
    if config.saveSubset:
        isRelevant = lambda var: any(s in var.name for s in config.varSubset)
        relevantVars = [var for var in tf.global_variables() if isRelevant(var)]
        subsetSaver = tf.train.Saver(relevantVars, max_to_keep=config.weightsToKeep, allow_empty=True)

    emaSaver = None
    if config.useEMA:
        emaSaver = tf.train.Saver(model.emaDict, max_to_keep=config.weightsToKeep)

    return {
        "saver": saver,
        "subsetSaver": subsetSaver,
        "emaSaver": emaSaver
    }

def main():
    parseArgs()
    print("Pre-Process")
    embeddings_init, index_to_word, word_to_index, dataset, label_map = pre_process()

    with tf.Graph().as_default():
        print("Building model...")
        start = time.time()
        model = MACnet(embeddings_init, index_to_word, label_map)
        print("took {} seconds".format(time.time() - start))

        init = tf.global_variables_initializer()
        # savers
        savers = setSavers(model)
        saver, emaSaver = savers["saver"], savers["emaSaver"]

        with tf.Session() as sess:
            # save grpah visualization
            if config.visualize_graph:
                summary_dir = "./summary"
                if not os.path.exists(os.path.dirname(summary_dir)):
                    os.makedirs(os.path.dirname(summary_dir))
                tf.summary.FileWriter(summary_dir, sess.graph)

            if config.save_weights:
                print("Saving weights while training")
                model.run(sess, saver, emaSaver, init, dataset)
            else:
                print("Skipping saving weights while training")
                model.run(sess, None, None, init, dataset)

if __name__ == '__main__':
    main()
