# MACforCQA
Applying MAC model on complexWebQuestions dataset.

This code is based on the code from: https://github.com/stanfordnlp/mac-network
of Drew A. Hudson & Christopher D. Manning.

## How to run the code
To achieve our results run the command:

python ./main.py --expName exp1 --train --epochs 2 --netLength 5 --outSnippet --writeSelfAtt --outFinalCtrl --top_k_candidates 50 --l2 1 --test --getAtt @configs/args.txt

## Introduction
Humans are often required to answer complex questions that involve reasoning over multiple pieces of
information, e.g., “Who is the author of the book currently at the top of the NY Times best seller list?”.

Answering such questions in broad domains can be quite involved for humans, because it demands integrating
information from multiple sources. Recently, Talmor and Berant (2018) suggested a model for answering
complex questions through question decomposition , while also introducing the ComplexWebQuestions
dataset.
Answering such questions requires reasoning, the capacity for consciously making sense of things, establishing
and verifying facts, applying logic, and changing or justifying practices, institutions and beliefs
based on new or existing information (definition borrowed from Wikipedia - Reasoning). It is considered
a distinguishing ability of an intellegent mind. In recent years, the scientific community has been seeking
to advance neural nets beyond sensory perception towards tasks that require more thinking, cognition
and intellect, with the goal of giving them the ability to use facts to draw conclusions. To this end,
compositional attention networks were introduced, including the MAC architecture (Hudson and Manning,
2018), which provided state-of-the-art performance for visual question answering. To achieve
this in the context of complex QA, we consider here a way to bypass the explicit need to decompose a
complex question, seeking a network that internally performs the structured and iterative reasoning necessary
for complex QA.

In the paper, we present a framework for QA that combines the two aforementioned approaches, adapting
the MAC architecture for complex question an swering, thus providing an end-to-end architecture
that bypasses the need for question decomposition. Our thesis is that answering complex questions can
be addressed by an attention based network, that encapsulates the decomposition process and integrates
the information internally. Our model gets as input a complex question and snippets from the web, containing
answers for the corresponding simple questions. It outputs an answer for the complex question,
having decomposed it and reasoned over the knowledge base internally. To evaluate our framework we
need a dataset of complex questions that calls for reasoning over multiple pieces of information. For this
reason, we use the ComplexWebQuestions dataset, introduced by Talmor and Berant. We evaluate our
model on ComplexWebQuestions and get a precision @1 of 11.61. It worth noting that upon running the
network with just the output unit, without the recurrent attention cells, and with with just the question
representation as input, we get a similar 11.84 p@1, which suggests that our compositionality based implementation
does not improve compared to a baseline mapping from questions to inputs. In light of
the results, in subsection 5.2 we offer a few ideas as to why our model does not improve upon a simple
baseline.
