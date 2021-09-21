
from helpers.helpers import *
import spacy_universal_sentence_encoder


class Summarization:
    def __init__(self, nlp = spacy_universal_sentence_encoder.load_model('en_use_lg')):
        self.nlp = nlp

    def weightedSummary(self, weightedSentences, max_selected=None, aggresive_uniqueness=True):
        """
        weightedSentences: is a map from "text" to double
        """
        sentenceDocs = {}
        simMap = {}
        sentenceUniquness = {}

        if max_selected is None:
            max_selected = len(weightedSentences)

        for sent in weightedSentences:
            sentenceDocs[sent] = self.nlp(sent)
            simMap[sent] = {}
            sentenceUniquness[sent] = 1

        for sent1 in sentenceDocs:
            for sent2 in sentenceDocs:
                if sent1 in simMap[sent2]:
                    continue
                curr_sim = sentenceDocs[sent1].similarity(sentenceDocs[sent2])
                simMap[sent1][sent2] = curr_sim
                simMap[sent2][sent1] = curr_sim
        # Normalize similarity
        norm_sim = normalizeSimMatrix(simMap)

        done_sentences = set()
        res = []
        i = 0
        while i < min(max_selected, len(weightedSentences)):
            i = i+1
            sentence_scores = {}
            candidates = [sentence for sentence in sentenceUniquness if sentence not in done_sentences]
            if aggresive_uniqueness and len(candidates) > 1:
                candidates = sorted(candidates, key=lambda k: sentenceUniquness[k], reverse=True)
                candidates = candidates[0:int((len(candidates)+1)/2)]

            if len(candidates) < 1:
                break

            for sentence in candidates:
                if sentence in done_sentences:
                    continue
                curr_score = 0
                for other_sentence in norm_sim[sentence]:
                    if other_sentence in done_sentences:
                        continue
                    curr_score = curr_score+(weightedSentences[other_sentence]*sentenceUniquness[other_sentence]*norm_sim[sentence][other_sentence])
                sentence_scores[sentence] = curr_score
            best_sentence = sorted(sentence_scores, key=lambda k: sentence_scores[k], reverse=True)[0]
            res.append({"sentence": best_sentence, "score": sentence_scores[best_sentence]})
            for sentence in sentenceUniquness:
                sentenceUniquness[sentence] = sentenceUniquness[sentence]*(1-norm_sim[sentence][best_sentence])
        return res








