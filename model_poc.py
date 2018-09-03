import time
import math
import json
import numpy as np
import tensorflow as tf
import io
import sys
import nltk


import ops
from config import config
from mac_cell import MACCell

def get_seq_actual_length(lst):
    seq_len = 0
    for i in range(len(lst)):
        if lst[i] == 0:
            break
        else:
            seq_len += 1
    return lst, seq_len


'''
The MAC network model. It performs reasoning processes to answer a question over
knowledge base (the image) by decomposing it into attention-based computational steps,
each perform by a recurrent MAC cell.

The network has three main components. 
Input unit: processes the network inputs: raw question strings and image into
distributional representations.

The MAC network: calls the MACcells (mac_cell.py) config.netLength number of times,
to perform the reasoning process over the question and image.

The output unit: a classifier that receives the question and final state of the MAC
network and uses them to compute log-likelihood over the possible one-word answers.       
'''


class MACnet(object):
    '''Initialize the class.

    Args:
        embeddingsInit: initialization for word embeddings (random / glove).
        answerDict: answers dictionary (mapping between integer id and symbol).
    '''

    def __init__(self, embeddingsInit, index_to_word, label_map):
        self.embeddingsInit = embeddingsInit
        self.index_to_word = index_to_word
        self.label_map = label_map

        self.gradientVarsList = []
        self.ist = []

        self.answerLossList = []
        self.correctNumList = []
        self.answerAccList = []
        self.predsList = []

        # CD Cardinal number
        # JJ - Adjective
        # JJR - Adjective, comparative
        # JJS - Adjective, superlative
        # NN - Noun, singular or mass
        # NNS - Noun, plural
        # NNP - Proper noun, singular
        # NNPS - Proper noun, plural
        # RB - Adverb
        # RBR - Adverb, comparative
        # RBS - Adverb, superlative
        self.keep_pos = { "CD",
                          "JJ",
                          "JJR",
                          "JJS",
                          "NN",
                          "NNS",
                          "NNP",
                          "NNPS",
                          "RB",
                          "RBR",
                          "RBS"}

        self.build()

    def add_placeholders(self):
        with tf.variable_scope("Placeholders"):
            ## data
            # questions
            self.questionsIndicesAll = tf.placeholder(tf.int32, shape=(None, None))
            self.questionLengthsAll = tf.placeholder(tf.int32, shape=(None,))

            # snippets
            self.snippetsIndicesAll = tf.placeholder(tf.int32, shape=(None, None, None)) # shape is (batch_size, num_snippets, num_words_in_snippet)
            self.snippetsLengthsAll = tf.placeholder(tf.int32, shape=(None, None)) # shape is (batch_size, num_snippets)

            # answers
            # self.answers_indices_start = tf.placeholder(tf.float32, shape=(None, None))
            # self.answers_indices_end = tf.placeholder(tf.float32, shape=(None, None))
            self.answers = tf.placeholder(tf.float32, shape=(None, None))

            # logits mask
            self.output_logits_mask = tf.placeholder(tf.float32, shape=(None, None))

            ## optimization
            self.lr = tf.placeholder(tf.float32, shape=())
            self.train = tf.placeholder(tf.bool, shape=())

            ## dropouts
            self.dropouts = {
                "encInput": tf.placeholder(tf.float32, shape=()),
                "encState": tf.placeholder(tf.float32, shape=()),
                "question": tf.placeholder(tf.float32, shape=()),
                "snippet": tf.placeholder(tf.float32, shape=()),
                "read": tf.placeholder(tf.float32, shape=()),
                "write": tf.placeholder(tf.float32, shape=()),
                "memory": tf.placeholder(tf.float32, shape=()),
                "output": tf.placeholder(tf.float32, shape=())
            }

            # batch norm params
            self.batchNorm = {"decay": config.bnDecay, "train": self.train}

    '''
        The Question Input Unit embeds the questions to randomly-initialized word vectors,
        and runs a recurrent bidirectional encoder (RNN/LSTM etc.) that gives back
        vector representations for each question (the RNN final hidden state), and
        representations for each of the question words (the RNN outputs for each word). 

        The method uses bidirectional LSTM, by default.
        Optionally projects the outputs of the LSTM (with linear projection / 
        optionally with some activation).

        Args:
            questions: question word embeddings  
            [batchSize, questionLength, wordEmbDim]

            questionLengths: the question lengths.
            [batchSize]

            projWords: True to apply projection on RNN outputs.
            projQuestion: True to apply projection on final RNN state.
            projDim: projection dimension in case projection is applied.  

        Returns:
            Contextual Words: RNN outputs for the words.
            [batchSize, questionLength, ctrlDim]

            Vectorized Question: Final hidden state representing the whole question.
            [batchSize, ctrlDim]
        '''
    def encoder(self, sequences, sequences_lengths, name, projWords=False,
                projQuestion=False, projDim=None):
        if name == "snippets":
            reuse = tf.AUTO_REUSE
        elif name == "questions":
            reuse = False
        with tf.variable_scope("encoder_{}".format(name), reuse=reuse): # pay attention that reuse don't work in multi-threaded environment
            # variational dropout option
            varDp = None
            if config.encVariationalDropout:
                varDp = {"stateDp": self.dropouts["stateInput"],
                         "inputDp": self.dropouts["encInput"],
                         "inputSize": config.wrdEmbDim}
            enc_dim = -1
            if name == "snippets":
                enc_dim = config.snippetEncDim
            elif name == "questions":
                enc_dim = config.encDim

            # rnns
            # num_layers = config.encNumLayers
            # if name == "snippets":
            #     num_layers = 3
            # print(f"using stack bilstm {num_layers}")
            # for i in range(num_layers):
            for i in range(config.encNumLayers):
                sequence_cntx_words, vec_sequences = ops.RNNLayer(sequences, sequences_lengths,
                                                                  enc_dim, bi=config.encBi, cellType=config.encType,
                                                                  dropout=self.dropouts["encInput"], varDp=varDp,
                                                                  name="rnn{}_{}".format(name, i))

            # dropout for the question vector
            if name == "questions":
                vec_sequences = tf.nn.dropout(vec_sequences, self.dropouts["question"])
            elif name == "snippets":
                vec_sequences = tf.nn.dropout(vec_sequences, self.dropouts["snippet"])

            # projection of encoder outputs
            if projWords:
                sequence_cntx_words = ops.linear(sequence_cntx_words, enc_dim, projDim,
                                               name="projCW")
            if projQuestion:
                vec_sequences = ops.linear(vec_sequences, enc_dim, projDim,
                                          act=config.encProjQAct, name="projQ")

        return sequence_cntx_words, vec_sequences

    '''
        Runs the MAC recurrent network to perform the reasoning process.
        Initializes a MAC cell and runs netLength iterations.

        Currently it passes the question and knowledge base to the cell during
        its creating, such that it doesn't need to interact with it through 
        inputs / outputs while running. The recurrent computation happens 
        by working iteratively over the hidden (control, memory) states.  

        Args:
            images: flattened image features. Used as the "Knowledge Base".
            (Received by default model behavior from the Image Input Units).
            [batchSize, H * W, memDim]

            vecQuestions: vector questions representations.
            (Received by default model behavior from the Question Input Units
            as the final RNN state).
            [batchSize, ctrlDim]

            questionWords: question word embeddings.
            [batchSize, questionLength, ctrlDim]

            questionCntxWords: question contextual words.
            (Received by default model behavior from the Question Input Units
            as the series of RNN output states).
            [batchSize, questionLength, ctrlDim]

            questionLengths: question lengths.
            [batchSize]

        Returns the final control state and memory state resulted from the network.
        ([batchSize, ctrlDim], [bathSize, memDim])
        '''

    def MACnetwork(self, kb, snippets_lengths, vecQuestions, questionWords, questionCntxWords,
                   questionLengths, name="", reuse=None):

        with tf.variable_scope("MACnetwork" + name, reuse=reuse):
            self.macCell = MACCell(
                vecQuestions=vecQuestions,
                questionWords=questionWords,
                questionCntxWords=questionCntxWords,
                questionLengths=questionLengths,  # used in the softmax attention on the question
                knowledgeBase=kb,
                snippets_lengths=snippets_lengths,
                memoryDropout=self.dropouts["memory"],
                readDropout=self.dropouts["read"],
                writeDropout=self.dropouts["write"],
                batchSize=self.batchSize,
                train=self.train,
                reuse=reuse)

            state = self.macCell.zero_state(self.batchSize, tf.float32)

            # inSeq = tf.unstack(inSeq, axis = 1)
            none = tf.zeros((self.batchSize, 1), dtype=tf.float32)

            # for i, inp in enumerate(inSeq):
            for i in range(config.netLength):
                self.macCell.iteration = i
                _, state = self.macCell(none, state)


            finalControl = state.control
            finalMemory = state.memory

        return finalControl, finalMemory


    '''
        Output Unit (step 2): Computes the logits for the answers. Passes the features
        through fully-connected network to get the logits over the possible answers.
        Optionally uses answer word embeddings in computing the logits (by default, it doesn't).
    
        Args:
            features: features used to compute logits
            [batchSize, inDim]
    
            inDim: features dimension
    
            aEmbedding: supported word embeddings for answer words in case answerMod is not NON.
            Optionally computes logits by computing dot-product with answer embeddings.
    
        Returns: the computed logits.
        [batchSize, answerWordsNum]
        '''


    def classifier(self, features, inDim):
        with tf.variable_scope("classifier"):
            outDim = config.max_possible_answers
            dims = [inDim] + config.outClassifierDims + [outDim]
            logits = ops.FCLayer(features, dims,
                                 batchNorm=self.batchNorm if config.outputBN else None,
                                 dropout=self.dropouts["output"])
        return logits

    def get_logits_mask_by_snippets_length(self, snippets_length):
        """
        snippets length is a matrix, (batch_size, number_of_snippets)
        :param snippets_length:
        :return:
        """
        sample_mask = [False] * config.max_possible_answers
        for label in range(len(sample_mask)):
            snippet_index, start_index, end_index = self.label_map[label]
            if snippets_length[snippet_index] < end_index:
                sample_mask[label] = True

        sample_mask = np.asarray(sample_mask)
        sample_mask = sample_mask * (-ops.inf)
        return sample_mask

    def get_logits_mask_pos_and_padding(self, snippets_length, snippets_pos, correct_label=None):
        sample_mask = [False] * config.max_possible_answers
        for label in range(len(sample_mask)):
            snippet_index, start_index, end_index = self.label_map[label]
            # check if this is a unigram in the excluded part of speech

            if correct_label is not None and correct_label[label] == 0 and \
               start_index+1 == end_index and snippets_pos[snippet_index][start_index][1] not in self.keep_pos:
                sample_mask[label] = True

            # check if label contains padding
            if snippets_length[snippet_index] < end_index:
                sample_mask[label] = True

        sample_mask = np.asarray(sample_mask)
        sample_mask = sample_mask * (-ops.inf)
        return sample_mask

    def get_logits_mask_ner_and_padding(self, snippets_length, snippets_ner, correct_label=None):
        sample_mask = [False] * config.max_possible_answers
        for label in range(len(sample_mask)):
            snippet_index, start_index, end_index = self.label_map[label]
            # check if this is a unigram in the excluded part of speech

            if correct_label is not None and correct_label[label] == 0:
                for i in range(start_index, end_index):
                    # print(len(snippets_ner), snippet_index, len(snippets_ner[snippet_index]), i)
                    if snippets_ner[snippet_index][i]:
                        sample_mask[label] = True

            # check if label contains padding
            if snippets_length[snippet_index] < end_index:
                sample_mask[label] = True

        sample_mask = np.asarray(sample_mask)
        sample_mask = sample_mask * (-ops.inf)
        return sample_mask

    def get_logits_mask_cand_and_padding(self, snippets_length, cand_mask):
        sample_mask = cand_mask
        for label in range(len(sample_mask)):
            snippet_index, start_index, end_index = self.label_map[label]

            # check if label contains padding
            if snippets_length[snippet_index] < end_index:
                sample_mask[label] = True

        sample_mask = np.asarray(sample_mask)
        if config.sigmoid_cross_entropy:
            sample_mask = 1 - sample_mask
        else:
            sample_mask = sample_mask * (-ops.inf)
        return sample_mask



    # Computes mean cross entropy loss between logits and answers.
    def addAnswerLossOp(self, logits, answers):
        with tf.variable_scope("answerLoss"):
            if config.sigmoid_cross_entropy:
                losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=answers, logits=logits)
            elif config.manual_cross_entropy:
                print("computing manual cross entropy")
                qs = tf.nn.softmax(logits)
                losses = -tf.reduce_sum(answers * tf.log(qs), axis=1)
            else:
                print("using softmax_cross_entropy_with_logits_v2")
                prob_answers = answers / tf.reshape(tf.reduce_sum(answers, axis=1), (-1, 1))
                losses = tf.nn.softmax_cross_entropy_with_logits_v2(labels=prob_answers, logits=logits)
            loss = tf.reduce_mean(losses)
            self.answerLossList.append(loss)

        return loss

    # Computes predictions (by finding maximal logit value, corresponding to highest probability)
    # and mean accuracy between predictions and answers.
    def addPredOp(self, logits, answers):
        with tf.variable_scope("pred"):
            preds = tf.argmax(logits, axis = -1) # tf.nn.softmax(
            # instead of comparing one-hot predicted label to one-hot true label
            # we are going to declare "Correct" if the predicted index is in the possible indices
            # tf.mul is bitwise, hence only intersection bits should be 1
            # since we take only one bit for index prediction, we can count the number of bits
            # after tf.multiply and this is the number of correct answers
            indices = preds
            depth = tf.shape(logits)[1] # number of possible indices
            predicted_answers_one_hot = tf.one_hot(indices, depth)
            corrects = tf.multiply(predicted_answers_one_hot, answers)

            correctNum = tf.reduce_sum(tf.to_int32(corrects))
            acc = tf.reduce_mean(tf.to_float(corrects))
            self.correctNumList.append(correctNum)
            self.answerAccList.append(acc)


        return preds, corrects, correctNum, tf.nn.softmax(logits)

    # Creates optimizer (adam)
    def add_optimizer_op(self):
        with tf.variable_scope("trainAddOptimizer"):
            self.globalStep = tf.Variable(0, dtype = tf.int32, trainable = False, name = "globalStep") # init to 0 every run?
            self.optimizer = tf.train.AdamOptimizer(learning_rate = self.lr)

    '''
    Computes gradients for all variables or subset of them, based on provided loss, 
    using optimizer.
    '''
    def computeGradients(self, optimizer, loss, trainableVars = None): # tf.trainable_variables()
        with tf.variable_scope("computeGradients"):
            if config.trainSubset:
                trainableVars = []
                allVars = tf.trainable_variables()
                for var in allVars:
                    if any((s in var.name) for s in config.varSubset):
                        trainableVars.append(var)

            gradients_vars = optimizer.compute_gradients(loss, trainableVars)
        return gradients_vars

    '''
        Output Unit (step 1): chooses the inputs to the output classifier.

        By default the classifier input will be the the final memory state of the MAC network.
        If outQuestion is True, concatenate the question representation to that.
        If outImage is True, concatenate the image flattened representation.

        Args:
            memory: (final) memory state of the MAC network.
            [batchSize, memDim]

            vecQuestions: question vector representation.
            [batchSize, ctrlDim]

            images: image features.
            [batchSize, H, W, imageInDim]

            imageInDim: images dimension.

        Returns the resulted features and their dimension. 
        '''

    # def outputOp(self, memory, vecQuestions, images, imageInDim):
    def outputOp(self, memory, vecQuestions, vec_snippets, final_control, contextual_words):
        with tf.variable_scope("outputUnit"):
            if not config.baseline:
                features = memory
                dim = config.memDim
            else:
                print("using baseline")

            if config.outQuestion:
                eVecQuestions = ops.linear(vecQuestions, config.ctrlDim, config.memDim, name="outQuestion")
                if config.baseline:
                    features = eVecQuestions
                    dim = config.memDim
                else:
                    features, dim = ops.concat(features, eVecQuestions, config.memDim, mul=config.outQuestionMul)

            if config.outSnippet:
                # skipping dimensionality reduction, since the input dim and output dim are the same for
                # the snippets.
                # dim is the dimension of features, override it since the returned dim here doesn't take into count
                # the fact that vec snippets is not in the same dimension of features
                override_dim = dim + config.num_snippets * config.memDim
                features, dim = ops.concat(features, vec_snippets, config.memDim, mul=config.outSnippetMul)
                dim = override_dim
            else:
                print("not using snippets in output")

            if config.outFinalCtrl and not config.baseline:
                # dim is the dimension of features, override it since the returned dim here doesn't take into count
                # the fact that final_control is not in the same dimension of features
                print("using final control")
                override_dim = dim + config.ctrlDim
                features, dim = ops.concat(features, final_control, config.ctrlDim)
                dim = override_dim
            elif config.baseline:
                print("using baseline")

            if config.out_contextual_words: # currently not in use, memory issues
                # dim is the dimension of features, override it since the returned dim here doesn't take into count
                # the fact that final_control is not in the same dimension of features
                print("using contextual words")
                contextual_words = tf.reshape(contextual_words, shape=(config.batchSize, -1))
                override_dim = dim + config.num_snippets * config.snippets_max_len * config.memDim
                features, dim = ops.concat(features, contextual_words, config.ctrlDim)
                dim = override_dim


        return features, dim

    '''
        Apply gradients. Optionally clip them, and update exponential moving averages 
        for parameters.
        '''

    def addTrainingOp(self, optimizer, gradients_vars):
        with tf.variable_scope("train"):
            gradients, variables = zip(*gradients_vars)
            norm = tf.global_norm(gradients)

            # gradient clipping
            if config.clipGradients:
                clippedGradients, _ = tf.clip_by_global_norm(gradients, config.gradMaxNorm, use_norm=norm)
                gradients_vars = zip(clippedGradients, variables)

            # updates ops (for batch norm) and train op
            updateOps = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(updateOps):
                train = optimizer.apply_gradients(gradients_vars, global_step=self.globalStep)

            # exponential moving average
            if config.useEMA:
                ema = tf.train.ExponentialMovingAverage(decay=config.emaDecayRate)
                maintainAveragesOp = ema.apply(tf.trainable_variables())

                with tf.control_dependencies([train]):
                    trainAndUpdateOp = tf.group(maintainAveragesOp)

                train = trainAndUpdateOp

                self.emaDict = ema.variables_to_restore()

        return train, norm

    def build(self):
        self.add_placeholders()
        self.add_optimizer_op()

        with tf.variable_scope("macModel"):
            # use the actual batch size, it may be less than the configuration
            self.batchSize = tf.shape(self.questionsIndicesAll)[0]

            self.loss = tf.constant(0.0)
            embeddings_var = tf.get_variable("emb", initializer=tf.to_float(self.embeddingsInit), dtype=tf.float32)
            # concatenating row of zeros for padding
            embeddings_var = tf.concat([tf.zeros((1, config.wrdEmbDim)), embeddings_var], axis=0)
            question_embeddings_lookup = tf.nn.embedding_lookup(embeddings_var, self.questionsIndicesAll)
            question_cntx_words, vec_questions = self.encoder(question_embeddings_lookup,
                                                           self.questionLengthsAll, "questions", False, False,
                                                           config.ctrlDim)

            snippets_cntx_words = []
            snippets_last_fw_first_bw = []
            for i in range(config.num_snippets):
                snippets_embeddings_lookup = tf.nn.embedding_lookup(embeddings_var, self.snippetsIndicesAll[:,i,])
                snippet_cntx_words, vec_snippets = self.encoder(snippets_embeddings_lookup,
                                                            self.snippetsLengthsAll[:,i], "snippets", False, False,
                                                            config.memDim) # memDim is the output dimension of vec_snippets and the contextual words
                snippets_cntx_words.append(snippet_cntx_words)
                snippets_last_fw_first_bw.append(vec_snippets)
            # no need to flatten kb this way
            cntx_word_level_kb = tf.concat(snippets_cntx_words, 1)
            snippet_level_kb = tf.concat(snippets_last_fw_first_bw, 1)

            final_control, finalMemory = self.MACnetwork(cntx_word_level_kb, self.snippetsLengthsAll, vec_questions,
                                                        question_embeddings_lookup, question_cntx_words, self.questionLengthsAll)

            # Output Unit - step 1 (preparing classifier inputs)
            output, dim = self.outputOp(finalMemory, vec_questions, snippet_level_kb, final_control, cntx_word_level_kb)

            # Output Unit - step 2 (classifier)
            logits = self.classifier(output, dim)

            # apply mask on logits, some of the labels contains padding, we don't want to predict using them
            if config.sigmoid_cross_entropy:
                logits = logits * self.output_logits_mask
            else:
                logits += self.output_logits_mask

            # compute loss, predictions, accuracy
            loss = self.addAnswerLossOp(logits, self.answers)

            if config.l2 > 0:
                print(f"using l2 regularization {config.l2}")
                loss += ops.L2RegularizationOp()

            self.preds, self.corrects, self.correct_num, self.predict_proba = self.addPredOp(logits, self.answers)

            self.loss += loss # code of multiple gpus

            # compute gradients
            gradient_vars = self.computeGradients(self.optimizer, self.loss, trainableVars=None)
            # self.gradientVarsList.append(gradient_vars)

        self.trainOp, self.gradNorm = self.addTrainingOp(self.optimizer, gradient_vars)
        self.noOp = tf.no_op()

    def buildPredsList(self, preds, snippets_per_batch, attentionMaps, predict_proba, answers, is_test):
        predsList = []
        attList = []
        agg_pred_list = []
        agg_corrects = 0
        for i, snippets in enumerate(snippets_per_batch):
            pred = preds[i]
            snippet_index, start_index, end_index = self.label_map[pred]
            pred_answer = snippets[snippet_index][start_index:end_index]

            pred_answer = [self.index_to_word[i] for i in pred_answer]
            pred_answer = u" ".join(pred_answer)

            # aggregate np attentions of instance i in the batch into 2d list
            attMapToList = lambda attMap: [step[i].tolist() for step in attMap]
            if attentionMaps is not None:
                attentions = {k: attMapToList(attentionMaps[k]) for k in attentionMaps}
                attList.append(attentions)
            predsList.append(pred_answer)

            # ====== Perform prediction according to aggregation of probabilities of the same answers =====
            agg = dict()
            for j in range(len(predict_proba[i])):
                snippet_index, start_index, end_index = self.label_map[j]
                curr_answer = tuple(snippets[snippet_index][start_index:end_index])
                if curr_answer not in agg:
                    agg[curr_answer] = [j, predict_proba[i][j]]
                else:
                    agg[curr_answer][1] += predict_proba[i][j]
            agg_pred_answer, agg_value = max(agg.items(), key=lambda t: t[1][1])
            agg_pred_answer_label = agg_value[0]
            # calc accuracy add to new pred list
            agg_pred_answer = [self.index_to_word[i] for i in agg_pred_answer]
            agg_pred_answer = u" ".join(agg_pred_answer)
            agg_pred_list.append(agg_pred_answer)
            if not is_test:
                if answers[i][agg_pred_answer_label] == 1:
                    agg_corrects += 1

        return predsList, agg_pred_list, agg_corrects, attList

    '''
    Processes a batch of data with the model.

    Args:
        sess: TF session

        data: Data batch. Dictionary that contains numpy array for:
        questions, questionLengths, answers. 
        See preprocess.py for further information of the batch structure.

        images: batch of image features, as numpy array. images["images"] contains
        [batchSize, channels, h, w]

        train: True to run batch for training.

        getAtt: True to return attention maps for question and image (and optionally 
        self-attention and gate values).

    Returns results: e.g. loss, accuracy, running time.
    '''
    def runBatch(self, sess, batch, train, getAtt=False, is_test=False):

        trainOp = self.trainOp if train else self.noOp
        gradNormOp = self.gradNorm if train else self.noOp

        predsOp = (self.preds,            # 0
                   self.correct_num,      # 1
                   self.correctNumList,   # 2
                   self.answerAccList[0], # 3
                   self.predict_proba)    # 4

        attOp = self.macCell.attentions

        time0 = time.time()
        feed = {
            self.questionsIndicesAll: batch["questions"],
            self.questionLengthsAll: batch["questions_length"],
            self.snippetsIndicesAll: batch["snippets"],
            self.snippetsLengthsAll: batch["snippets_length"],
            self.answers: batch["answers"],
            self.output_logits_mask: batch["output_logits_mask"],


            self.dropouts["encInput"]: config.encInputDropout if train else 1.0,
            self.dropouts["encState"]: config.encStateDropout if train else 1.0,
            self.dropouts["question"]: config.qDropout if train else 1.0, #_
            self.dropouts["snippet"]: config.snippetDropout if train else 1.0,  # _
            self.dropouts["memory"]: config.memoryDropout if train else 1.0,
            self.dropouts["read"]: config.readDropout if train else 1.0, #_
            self.dropouts["write"]: config.writeDropout if train else 1.0,
            self.dropouts["output"]: config.outputDropout if train else 1.0,

            self.lr: config.lr,
            self.train: train
        }

        time1 = time.time()
        _, loss, predsInfo, gradNorm, attentionMaps = sess.run(
            [trainOp, self.answerLossList[0], predsOp, gradNormOp, attOp],
            feed_dict=feed)

        time2 = time.time()
        predsList, agg_pred_list, agg_corrects, attList = self.buildPredsList(predsInfo[0], batch["snippets"],
                                                                     attentionMaps if getAtt else None, predsInfo[4], batch["answers"], is_test)

        # debug printouts
        print("argmax_corrects {}, agg_argmax_corrects {}".format(predsInfo[1], agg_corrects))

        return {"loss": loss,
                "correctNum": predsInfo[1],
                "acc": predsInfo[3],
                "preds": predsList,
                "agg_preds": agg_pred_list,
                "agg_correctNum": agg_corrects,
                "attList": attList,
                "gradNorm": gradNorm if train else -1,
                "readTime": time1 - time0,
                "trainTime": time2 - time1}

    # Gets last logged epoch and learning rate
    def lastLoggedEpoch(self):
        with open(config.logFile(), "r") as inFile:
            lastLine = list(inFile)[-1].split(",")
        epoch = int(lastLine[0])
        lr = float(lastLine[-1])
        return epoch, lr

    # Write a line to file
    def writeline(self, f, line):
        f.write(str(line) + u"\n")

    # Write a list to file
    def writelist(self, f, l):
        self.writeline(f, ",".join(map(str, l)))

    # Writes log header to file
    def logInit(self):
        # with open(config.logFile(), "a+") as outFile:
        with io.open(config.logFile(), "a+", encoding="utf-8") as outFile:
            self.writeline(outFile, config.expName)
            headers = ["epoch", "trainAcc", "trainAggAcc", "valAcc", "valAggAcc", "trainLoss", "valLoss"]
            if config.evalTrain:
                headers += ["evalTrainAcc", "evalTrainAggAcc", "evalTrainLoss"]
            if config.extra:
                if config.evalTrain:
                    headers += ["thAcc", "thLoss"]
                headers += ["vhAcc", "vhLoss"]
            headers += ["time", "lr"]

            self.writelist(outFile, headers)
            # lr assumed to be last

    # Restores weights of specified / last epoch if on restore mod.
    # Otherwise, initializes weights.
    def loadWeights(self, sess, saver, init):
        if config.restoreEpoch > 0 or config.restore:
            # restore last epoch only if restoreEpoch isn't set
            if config.restoreEpoch == 0:
                # restore last logged epoch
                config.restoreEpoch, config.lr = self.lastLoggedEpoch()
            print("Restoring epoch {} and lr {}".format(config.restoreEpoch, config.lr))
            print("Restoring weights")
            saver.restore(sess, config.weightsFile(config.restoreEpoch))
            epoch = config.restoreEpoch
        else:
            print("Initializing weights")
            sess.run(init)
            self.logInit()
            epoch = 0

        return epoch

    # Computes exponential moving average.
    def ema_avg(self, avg, value):
        if avg is None:
            return value
        emaRate = 0.98
        return avg * emaRate + value * (1 - emaRate)

    def update_stats(self, stats, res, batch):
        stats["totalBatches"] += 1
        stats["totalData"] += batch["questions"].shape[0]

        stats["totalLoss"] += res["loss"]
        stats["totalCorrect"] += res["correctNum"]
        stats["agg_totalCorrect"] += res["agg_correctNum"]

        stats["loss"] = stats["totalLoss"] / stats["totalBatches"]
        stats["acc"] = stats["totalCorrect"] / stats["totalData"]
        stats["agg_acc"] = stats["agg_totalCorrect"] / stats["totalData"]

        stats["emaLoss"] = self.ema_avg(stats["emaLoss"], res["loss"])
        stats["emaAcc"] = self.ema_avg(stats["emaAcc"], res["acc"])

    # auto-encoder ae = {:2.4f} autoEncLoss,
    # Translates training statistics into a string to print
    def stats_to_str(self, stats, res, epoch, batchNum, startTime, tier):
        formatStr = "stats for {tier:5s}: epoch[{epoch}],batchNum[{batchNum}] dataProcessed[{dataProcessed}]" + \
                    "time = {time} ({loadTime:2.2f}+{trainTime:2.2f}), " + \
                    "lr {lr}, loss = {loss}, acc = {acc}, agg_acc = {agg_acc}, avgLoss = {avgLoss}, " + \
                    "avgAcc = {avgAcc}, gradNorm = {gradNorm:2.4f}, " + \
                    "emaLoss = {emaLoss:2.4f}, emaAcc = {emaAcc:2.4f}; " + \
                    "expname{expname}"

        s_epoch = "{:2d}".format(epoch)
        s_batchNum = "{:3d}".format(batchNum)
        s_dataProcessed = "{:5d}".format(stats["totalData"])
        s_time = "{:2.2f}".format(time.time() - startTime)
        s_loadTime = res["readTime"]
        s_trainTime = res["trainTime"]
        s_lr = config.lr
        s_loss = "{:2.4f}".format(res["loss"])
        s_acc = "{:2.4f}".format(stats["acc"])
        s_agg_acc = "{:2.4f}".format(stats["agg_acc"])
        s_avgLoss = "{:2.4f}".format(stats["loss"])
        s_avgAcc = "{:2.4f}".format(res["acc"])
        s_gradNorm = res["gradNorm"]
        s_emaLoss = stats["emaLoss"]
        s_emaAcc = stats["emaAcc"]
        s_expname = config.expName

        return formatStr.format(tier=tier, epoch=s_epoch, batchNum=s_batchNum, dataProcessed=s_dataProcessed,
                                time=s_time, loadTime=s_loadTime,
                                trainTime=s_trainTime, lr=s_lr, loss=s_loss, acc=s_acc, agg_acc=s_agg_acc,
                                avgLoss=s_avgLoss, avgAcc=s_avgAcc, gradNorm=s_gradNorm,
                                emaLoss=s_emaLoss, emaAcc=s_emaAcc, expname=s_expname)

    # Batches data into a a list of batches of batchSize.
    # Shuffles the data by default.
    def get_batches(self, questions, snippets, snippets_pos, snippets_ner, candidates_mask, batchSize=None, shuffle=True, is_test=False, is_train=False):
        batches = []

        num_questions = len(questions)
        num_snippets = len(snippets)

        assert num_questions == num_snippets

        if batchSize is None or batchSize > num_questions:
            batchSize = num_questions

        indices = np.arange(num_questions)
        if shuffle:
            np.random.shuffle(indices)

        for batchStart in range(0, num_questions, batchSize):
            batch_indices = indices[batchStart: batchStart + batchSize]
            batch = {"questions": [], "questions_length": [], "snippets": [], "snippets_length": [],
                     "output_logits_mask": [], "answers": []}
            for i in batch_indices:
                q_new, q_len = get_seq_actual_length(questions[i]["question"])
                batch["questions"].append(q_new)
                batch["questions_length"].append(q_len)
                new_snippets = []
                new_snippets_lens = []
                for j in range(len(snippets[i])):
                    s_new, s_len = get_seq_actual_length(snippets[i][j])
                    new_snippets.append(s_new)
                    new_snippets_lens.append(s_len)
                batch["snippets"].append(new_snippets)
                batch["snippets_length"].append(new_snippets_lens)
                if config.mask_output_by_padding:
                    batch["output_logits_mask"].append(self.get_logits_mask_by_snippets_length(new_snippets_lens))
                elif config.mask_output_by_pos:
                    batch["output_logits_mask"].append(self.get_logits_mask_pos_and_padding(new_snippets_lens, snippets_pos[i],
                                                                                            questions[i]["answers"] if not is_test else None))
                elif config.mask_output_by_tf_and_stop_words:
                    batch["output_logits_mask"].append(self.get_logits_mask_cand_and_padding(new_snippets_lens, candidates_mask[i]))
                elif config.mask_output_by_ner:
                    batch["output_logits_mask"].append(self.get_logits_mask_ner_and_padding(new_snippets_lens, snippets_ner[i],
                                                         questions[i]["answers"] if is_train else None))
                if not is_test:
                    batch["answers"].append(questions[i]["answers"])
                else:
                    batch["answers"].append([False]*config.max_possible_answers)
            # convert to numpy
            for k, v in batch.items():
                    batch[k] = np.asarray(v)
            batches.append(batch)
        return batches, indices

    def run_epoch(self, sess, questions, snippets, snippets_pos, snippets_ner, candidates_mask, epoch, saver = None, getAtt = False, is_test=False, tier="", is_train=False):
        stats = {
        "totalBatches": 0,
        "totalData": 0,
        "totalLoss": 0.0,
        "totalCorrect": 0,
        "agg_totalCorrect": 0,
        "loss": 0.0,
        "acc": 0.0,
        "agg_acc": 0.0,
        "emaLoss": None,
        "emaAcc": None,
        }
        preds = []
        agg_preds = []
        attList = []

        # indices is the original indices after shuffeling, we will use that to recover
        # the correct order of our predictions
        batches, indices = self.get_batches(questions, snippets, snippets_pos, snippets_ner, candidates_mask, config.batchSize, is_test=is_test, is_train=is_train)

        # train on every batch
        for batchNum, batch in enumerate(batches):
            startTime = time.time()
            res = self.runBatch(sess, batch, is_train, getAtt, is_test)

            # update stats
            self.update_stats(stats, res, batch)
            preds += res["preds"]
            agg_preds += res["agg_preds"]
            attList += res["attList"]
            # print stats
            print(self.stats_to_str(stats, res, epoch, batchNum, startTime, tier))

        # save weight for next runs
        if saver is not None:
            print("")
            print("saving weights")
            saver.save(sess, config.weightsFile(epoch))

        # reorder the predictions in the original order
        sorted_preds = [None] * len(preds)
        sorted_agg_preds = [None] * len(agg_preds)
        sorted_attList = [None] * len(attList)
        for i in range(len(preds)):
            orig_index = indices[i]
            sorted_preds[orig_index] = preds[i]
            sorted_agg_preds[orig_index] = agg_preds[i]
        for i in range(len(attList)):
            orig_index = indices[i]
            sorted_attList[orig_index] = attList[i]


        return {"loss": stats["loss"],
                "acc": stats["acc"],
                "agg_acc": stats["agg_acc"],
                "preds": sorted_preds,
                "agg_preds": sorted_agg_preds,
                "attList": sorted_attList
                }

    # Runs evaluation on train / val / test datasets.
    def runEvaluation(self, sess, dataset, epoch, evalTrain = False, evalTest=False, getAtt=None):
        if getAtt is None:
            getAtt = config.getAtt
        res = {"val": None, "test": None, "evalTrain": None}

        if evalTrain and config.evalTrain:
            res["evalTrain"] = self.run_epoch(sess, dataset["train"]["questions"], dataset["train"]["snippets"],
                                              dataset["train"]["pos"], dataset["train"]["ner"], dataset["train"]["candidates_mask"], is_train = False, epoch = epoch, getAtt = getAtt, tier="evalTrain")

        res["val"] = self.run_epoch(sess, dataset["val"]["questions"], dataset["val"]["snippets"],
                                    dataset["val"]["pos"], dataset["val"]["ner"], dataset["val"]["candidates_mask"], is_train=False, epoch=epoch, getAtt=getAtt, tier="val")

        if evalTest or config.test:
            res["test"] = self.run_epoch(sess, dataset["test"]["questions"], dataset["test"]["snippets"],
                                         dataset["test"]["pos"], dataset["test"]["ner"], dataset["test"]["candidates_mask"], is_train=False, epoch=epoch, getAtt=getAtt, is_test=True, tier="test")

        return res

    # Print results for a tier
    def printTierResults(self, tierName, res):
        if res is None:
            return

        print("{tierName} Loss: {loss}, {tierName} accuracy: {acc} agg accuracy {agg_acc}".format(tierName=tierName,
                                                                           loss=res["loss"],
                                                                           acc=res["acc"],
                                                                           agg_acc = res["agg_acc"]))

    # Prints dataset results (for several tiers)
    def printDatasetResults(self, trainRes, evalRes):
        self.printTierResults("Training", trainRes)
        self.printTierResults("Training EMA", evalRes["evalTrain"])
        self.printTierResults("Validation", evalRes["val"])

    def write_preds(self, res, tier, suffix = ""):
        if res is None:
            return
        sortedPreds = res["preds"]
        # with open(config.predsFile(tier + suffix), "w") as outFile:
        with io.open(config.predsFile(tier + suffix), "w", encoding="utf-8") as outFile:
            outFile.write(json.dumps(sortedPreds))
        # with open(config.answersFile(tier + suffix), "w") as outFile:
        with io.open(config.answersFile(tier + suffix), "w+", encoding="utf-8") as outFile:
            for pred in sortedPreds:
                self.writeline(outFile, pred)

        sortedAggPreds = res["agg_preds"]
        # with open(config.predsFile(tier + suffix), "w") as outFile:
        with io.open(config.predsFile(tier + suffix)+".agg", "w", encoding="utf-8") as outFile:
            outFile.write(json.dumps(sortedAggPreds))
        # with open(config.answersFile(tier + suffix), "w") as outFile:
        with io.open(config.answersFile(tier + suffix)+".agg", "w+", encoding="utf-8") as outFile:
            for pred in sortedAggPreds:
                self.writeline(outFile, pred)

    def write_attentions(self, res, tier, suffix=""):
        with open(f"attention_{tier}_{config.expName}_{suffix}.pkl", "wb+") as f:
            import pickle
            pickle.dump(res["attList"], f)

    # Writes log record to file
    def log_record(self, epoch, epochTime, lr, trainRes, evalRes):
        import io
        with io.open(config.logFile(), "a+", encoding="utf-8") as outFile:
        # with open(config.logFile(), "a+") as outFile:
            record = [epoch, trainRes["acc"], trainRes["agg_acc"], evalRes["val"]["acc"], evalRes["val"]["agg_acc"], trainRes["loss"], evalRes["val"]["loss"]]
            if config.evalTrain:
                record += [evalRes["evalTrain"]["acc"], evalRes["evalTrain"]["agg_acc"], evalRes["evalTrain"]["loss"]]
            record += [epochTime, lr]
            self.writelist(outFile, record)

    def better(self, currRes, bestRes):
        return currRes["val"]["acc"] > bestRes["val"]["acc"]

    def improveEnough(self, curr, prior, lr):
        print("improveEnough was not implemented")

        return True

    def run(self, sess, saver, ema_saver, init, dataset):
        # restore / initialize weights, initialize epoch variable
        epoch = self.loadWeights(sess, saver, init)

        if config.train:
            start0 = time.time()

            bestEpoch = epoch
            bestRes = None
            prevRes = None

            # Continue training since loaded epoch or from beginning
            for epoch in range(config.restoreEpoch + 1, config.epochs + 1):
                print("Training epoch {}...".format(epoch))
                start = time.time()

                # train
                train_res = self.run_epoch(sess, dataset["train"]["questions"], dataset["train"]["snippets"],
                                           dataset["train"]["pos"], dataset["train"]["ner"], dataset["train"]["candidates_mask"], is_train=True, epoch=epoch,
                                         saver=saver, tier="train")
                # save weights
                if saver is not None:
                    saver.save(sess, config.weightsFile(epoch))

                # load EMA weights
                if config.useEMA:
                    print("Restoring EMA weights")
                    ema_saver.restore(sess, config.weightsFile(epoch))

                # evaluation on dev set
                eval_res = self.runEvaluation(sess, dataset, epoch)

                # restore standard weights
                if config.useEMA:
                    print("Restoring standard weights")
                    saver.restore(sess, config.weightsFile(epoch))

                print("")
                epochTime = time.time() - start
                print("Epoch took {:.2f} seconds".format(epochTime))

                # print results
                self.printDatasetResults(train_res, eval_res)

                # stores predictions and optionally attention maps
                if config.getPreds:
                    print("Writing predictions...")
                    self.write_preds(eval_res["val"], "val")
                    if config.test:
                        print("Writing test predictions...")
                        self.write_preds(eval_res["test"], "test")

                #write attentions to pickle
                if config.getAtt:
                    print("writing attentions")
                    self.write_attentions(eval_res["val"], "val")
                    if config.test:
                        print("Writing test attentions...")
                        self.write_attentions(eval_res["test"], "test")

                self.log_record(epoch, epochTime, config.lr, train_res, eval_res)

                # update best result
                # compute curr and prior
                currRes = {"train": train_res, "val": eval_res["val"]}
                curr = {"res": currRes, "epoch": epoch}

                if bestRes is None or self.better(currRes, bestRes):
                    bestRes = currRes
                    bestEpoch = epoch

                prior = {"best": {"res": bestRes, "epoch": bestEpoch},
                         "prev": {"res": prevRes, "epoch": epoch - 1}}

                # lr reducing - use this if you want to reduce learning rate use this configuration
                if config.lrReduce:
                    if not self.improveEnough(curr, prior, config.lr):
                        config.lr *= config.lrDecayRate
                        print("Reducing LR to {}".format(config.lr))

                # early stopping
                if config.earlyStopping > 0:
                    if epoch - bestEpoch > config.earlyStopping:
                        break

                # update previous result
                prevRes = currRes

            # Perform the final test on test set
            if config.finalTest:
                print("Testing on epoch {}...".format(epoch))

                start = time.time()
                if epoch > 0:
                    if config.useEMA:
                        ema_saver.restore(sess, config.weightsFile(epoch))
                    else:
                        saver.restore(sess, config.weightsFile(epoch))

                # evaluation on dev set
                test_res = self.runEvaluation(sess, dataset, epoch, evalTest=True)

                print("Test took {:.2f} seconds".format(time.time() - start))
                self.printDatasetResults(None, test_res)

                print("Writing test predictions...")
                self.write_preds(test_res["test"], "test")

            print("Done!!!")
