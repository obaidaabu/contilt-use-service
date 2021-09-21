import spacy
from spacy.matcher import Matcher
from helpers.helpers import *



class DescriptionPhrases:
    def __init__(self, spacy_model= spacy.load('en_core_web_lg')):
        self.nlp = spacy_model
        self.PhrasesMatcher = Matcher(self.nlp.vocab)
        pattern1 = [{"POS": "ADJ"}, {"POS": "NOUN"}]
        pattern2 = [{"POS": "ADJ"}, {"POS": "NOUN"}, {"POS": "NOUN"}]
        pattern3 = [{"POS": "NOUN"}, {"POS": "NOUN"}, {"LEMMA": "be"}, {"POS": "ADJ"}]
        pattern4 = [{"POS": "NOUN"}, {"LEMMA": "be"}, {"POS": "ADJ"}]
        pattern5 = [{"POS": "ADJ"}, {"POS": "ADJ"}, {"POS": "NOUN"}, {"POS": "NOUN"}]
        pattern7 = [{"POS": "ADJ"}, {"POS": "ADJ"}, {"POS": "NOUN"}]
        #pattern6 = [{"POS": "ADJ"}, {"POS": "NOUN"}, {"LOWER": "and"}, {"POS": "NOUN"}]
        self.PhrasesMatcher.add("ADJNOUN", [pattern2, pattern1, pattern5, pattern7], greedy="LONGEST")
        # self.PhrasesMatcher.add("ADJNOUNNOUN", [pattern2])
        self.PhrasesMatcher.add("NOUNADJ", [pattern3, pattern4], greedy="LONGEST")
        # self.PhrasesMatcher.add("NOUNNOUNADJ", [pattern3])

        self.AdverbMatcher = Matcher(self.nlp.vocab)
        self.AdverbMatcher.add("ADVERB", [[{"POS": "ADV"}, {"POS": "ADJ"}]])
        self.penalty = 0.8

    def extract(self, docs):
        adjectiveDocFrequency = {}
        nameDocFrequency = {}
        phraseDocFreq = {}
        phraseWeight = {}
        for doc in docs:
            doc_names = set()
            doc_adjs = set()
            doc_phrases = set()
            for sentence in doc:
                sentence_phrases = self.getPhrases(sentence["text"])
                for phrase in sentence_phrases:
                    name = " ".join(phrase[1:])
                    adj = phrase[0]
                    doc_names.add(name)
                    doc_adjs.add(adj)
                    phrase_key = adj + " " + name
                    doc_phrases.add(phrase_key)

                    if phrase_key in phraseWeight:
                        phraseWeight[phrase_key] = phraseWeight[phrase_key]+sentence["score"]
                    else:
                        phraseWeight[phrase_key] = sentence["score"]

            for doc_name in doc_names:
                addVToMap(nameDocFrequency, doc_name, 1)
            for doc_adj in doc_adjs:
                addVToMap(adjectiveDocFrequency, doc_adj, 1)
            for doc_phrase in doc_phrases:
                addVToMap(phraseDocFreq, doc_phrase, 1)
        final_unsum = {}
        for doc_phrase in phraseDocFreq:
            phrase = doc_phrase.split()
            name = " ".join(phrase[1:])
            adj = phrase[0]
            if len(phrase) > 2 or (adjectiveDocFrequency[adj] > 1 and nameDocFrequency[name] > 1):
                final_unsum[doc_phrase] = phraseWeight[doc_phrase]

        return final_unsum

    def extract_old(self, docs):
        adjectiveTotalFrequency = {}
        adjectiveDocFrequency = {}
        nameTotalFrequency = {}
        nameDocFrequency = {}
        phraseTotalFreq = {}
        phraseDocFreq = {}
        for doc in docs:
            doc_names = set()
            doc_adjs = set()
            doc_phrases = set()
            for sentence in doc:
                sentence_phrases = self.getPhrases(sentence["text"])
                for phrase in sentence_phrases:
                    name = ""
                    adj = ""
                    if len(phrase) > 2:
                        name = phrase[1]+" "+phrase[2]
                        adj = phrase[0]
                    else:
                        name = phrase[1]
                        adj = phrase[0]
                    addVToMap(adjectiveTotalFrequency,adj,sentence["score"])
                    addVToMap(nameTotalFrequency,name,sentence["score"])
                    addVToMap(phraseTotalFreq,adj+" "+name,sentence["score"])
                    doc_names.add(name)
                    doc_adjs.add(adj)
                    doc_phrases.add(adj+" "+name);

            for doc_name in doc_names:
                addVToMap(nameDocFrequency,doc_name,1)
            for doc_adj in doc_adjs:
                addVToMap(adjectiveDocFrequency,doc_adj,1)
            for doc_phrase in doc_phrases:
                addVToMap(phraseDocFreq,doc_phrase,1)
        
        nameUniquness = {}
        for doc_name in nameDocFrequency:
            nameUniquness[doc_name] = 1.0

        max_phrase_frequency = max(phraseTotalFreq.values())
        phrase_frequency_score = dict((k,v/max_phrase_frequency) for (k,v) in phraseTotalFreq.items())

        phrase_static_score = {}
        phrase_to_name = {}
        for doc_phrase in phraseDocFreq:
            phrase_parts = doc_phrase.split()
            adj = phrase_parts[0]
            name = phrase_parts[1]
            if len(phrase_parts) > 2:
                    name = phrase_parts[1]+" "+phrase_parts[2]
            phrase_to_name[doc_phrase] = name

            df_score = 1.0

            if len(docs) > 4:
                phrase_df_score = 0.6
                if phraseDocFreq[doc_phrase] > 1:
                    phrase_df_score = 0.8
                if phraseDocFreq[doc_phrase] > 2:
                    phrase_df_score = 1.0
                
                name_df_score = 0.0
                if nameDocFrequency[name] > 1:
                    name_df_score = 0.7
                if nameDocFrequency[name] > 2:
                    name_df_score = 1.0

                adj_df_score = 0.0
                if adjectiveDocFrequency[adj] > 1:
                    adj_df_score = 0.7
                if adjectiveDocFrequency[adj] > 2:
                    adj_df_score = 1.0

                df_score = adj_df_score*name_df_score*phrase_df_score
            
            phrase_static_score[doc_phrase] = phrase_frequency_score[doc_phrase]*df_score
            if len(doc_phrase.split()) > 2:
                phrase_static_score[doc_phrase] = phrase_static_score[doc_phrase] ** 0.3

        res = []
        done_phrases = set()
        while len(res) < len(phrase_static_score):
            phrases_scores = dict((k,v*nameUniquness[phrase_to_name[k]]) for (k,v) in phrase_static_score.items() if k not in done_phrases)
            max_phrase = max(phrases_scores, key=phrases_scores.get)
            if phrases_scores[max_phrase] == 0.0:
                break
            nameUniquness[phrase_to_name[max_phrase]] = nameUniquness[phrase_to_name[max_phrase]]*(1-self.penalty)
            done_phrases.add(max_phrase)
            res.append(max_phrase)
        
        return res

    def getPhrases(self, sentence):
        analyzed_sent = self.nlp(sentence)
        adverbs = self.AdverbMatcher(analyzed_sent)
        for match_id, start, end in adverbs:
            sentence = sentence.replace(analyzed_sent[start:end][0].text, "").replace("  ", " ")

        analyzed_sent = self.nlp(sentence)
        matches = self.PhrasesMatcher(analyzed_sent)
        phrases = []
        for match_id, start, end in matches:
            string_id = self.nlp.vocab.strings[match_id]  # Get string representation
            span = analyzed_sent[start:end]
            if string_id == "ADJNOUN":
                if len(span) == 3:
                    currAdj = span[0].text.lower()
                    currName1 = span[1].text.lower()
                    currName2 = span[2].text.lower()
                    phrases.append([currAdj, currName1, currName2])
                else:
                    if len(span) == 4:
                        currAdj = span[0].text.lower()
                        currAdj2 = span[1].text.lower()
                        currName1 = span[2].text.lower()
                        currName2 = span[3].text.lower()
                        phrases.append([currAdj, currAdj2, currName1, currName2])
                    else:
                        currAdj = span[0].text.lower()
                        currName = span[1].text.lower()
                        phrases.append([currAdj, currName])

            if string_id == "NOUNADJ":
                if len(span) > 3:
                    currAdj = span[3].text.lower()
                    currName1 = span[0].text.lower()
                    currName2 = span[1].text.lower()
                    phrases.append([currAdj,currName1, currName2])
                else:
                    currAdj = span[2].text.lower()
                    currName = span[0].text.lower()
                    phrases.append([currAdj,currName])
                
            # if string_id == "NOUNNOUNADJ":
            #     currAdj = span[3].text
            #     currName1 = span[0].text
            #     currName2 = span[1].text
            #     phrases.append([currAdj,currName1, currName2])
        return phrases

    def getDescriptiveSentences(self, sentences):
        res = []
        for sentence in sentences:
            analyzed_sent = self.nlp(sentence)
            matches = self.PhrasesMatcher(analyzed_sent)
            if len(matches) > 0:
                res.append(sentence)
        return res

