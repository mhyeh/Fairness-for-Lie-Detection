import re
import os
import nltk
from pathlib import Path

DIC_EN = "dic/LIWC2015_English.dic"
DIC_ZH = "dic/Traditional_Chinese_LIWC2015_Dictionary.dic"
DIC_RO = "dic/Romanian_LIWC2015_Dictionary.dic"
DIC_NL = "dic/Dutch_LIWC2015_Dictionary.dic"

CAT_DELIM = "%"

class Liwc():
    def __init__(self, language):
        dirname = Path(__file__).parent
        self.core = None
        if language == "en":
            self.core = Liwc_EN(dirname)
        elif language == "zh":
            self.core = Liwc_ZH(dirname)
        elif language == "ro":
            self.core = Liwc_RO(dirname)
        elif language == "nl":
            self.core = Liwc_NL(dirname)

    def cal_liwc(self, sent, max_token=None):
        return self.core.cal_liwc(sent, max_token)

    def cal_dominant(self, sent, dominant_classes, max_token=None):
        return self.core.cal_dominant(sent, dominant_classes, max_token)

class Liwc_Super():
    def __init__(self):
        self._word_list = None

    def __cal_word_per_sent(self, sent, max_token):
        sents = nltk.sent_tokenize(sent)
        return sum([self._preprocess(s, max_token)[2] for s in sents]) / len(sents)

    def __cal_six_plus_word(self, token_seq):
        return sum([1 for t in token_seq if len(t) >= 6]) / len(token_seq)

    def _construct_word_list(self, filename):
        f = open(filename, "r")
        cats_section  = False
        cats_to_words = {}
        id_to_cat     = {}
        for l in f:
            l = l.strip()
            if l == CAT_DELIM:
                cats_section = not cats_section
                continue

            if cats_section:
                try:
                    i, cat = l.split("\t")
                    cat    = cat.split()[0]
                    id_to_cat[int(i)] = cat
                except:
                    pass
            else:
                w, cats = l.split("\t")[0], l.split("\t")[1:]

                if "(" in w and ")" in w: 
                    w = w.replace("(", "").replace(")", "")
                w = w.replace("(", r"\(").replace(")", r"\)")

                w = r"(?<!\S)" + w
                if "*" in w:
                    w = w.replace("*", r"\w*(?!\S)")
                    if w[-6:] != r"(?!\S)":
                        w += r"(?!\S)"
                else:
                    w += r"(?!\S)"

                for c in cats:
                    c = id_to_cat[int(c)]
                    if c not in cats_to_words: 
                        cats_to_words[c] = set()
                    cats_to_words[c].add(w)

        return cats_to_words

    def _preprocess(self, sent, max_token):
        sent = sent.lower().replace("kind of", "kindof")
        sent = re.sub(r"[^a-z0-9'\/-]", " ", sent)

        token_seq = sent.split()
        if max_token: 
            token_seq = token_seq[:max_token]

        sent = " ".join(token_seq)
        return sent, token_seq, len(token_seq)

    def _cal_coverage(self, sent, wc, word_set):
        reg_string = "|".join(word_set)
        r = re.compile(reg_string)

        matched  = r.findall(sent)
        matched  = " ".join(matched).split()
        coverage = len(matched) / wc
        assert coverage <= 1, print(matched, sent)

        return coverage, matched

    def cal_liwc(self, sent, max_token=None):
        proc_sent, token_seq, wc = self._preprocess(sent, max_token)
        if wc == 0: return None

        res = dict()
        res["words_per_sentence"] = self.__cal_word_per_sent(sent, max_token)
        res["six_plus_words"]     = self.__cal_six_plus_word(token_seq)
        res["word_count"]         = wc

        for cat, words in self._word_list.items():
            res[cat] = self._cal_coverage(proc_sent, wc, words)[0]

        return res

    def cal_dominant(self, sent, dominant_classes, max_token=None):
        proc_sent, token_seq, wc = self._preprocess(sent, max_token)
        if wc == 0: return None, [], []

        word_set = set()
        for c in dominant_classes:
            word_set |= self._word_list[c]

        coverage, matched = self._cal_coverage(proc_sent, wc, word_set)
        return coverage, matched, token_seq

class Liwc_EN(Liwc_Super):
    def __init__(self, dirname):
        super(Liwc_EN, self).__init__()


        self._word_list = self._construct_word_list(os.path.join(dirname, DIC_EN))

class Liwc_ZH(Liwc_Super):
    def __init__(self, dirname):
        super(Liwc_ZH, self).__init__()

        self._word_list = self._construct_word_list(os.path.join(dirname, DIC_EN))

    def _construct_word_list(self, filename):
        f = open(filename, "r")
        cats_section = False
        cats_to_words = {}
        id_to_cat = {}
        for l in f:
            l = l.strip()
            if l == CAT_DELIM:
                cats_section = not cats_section
                continue

            if cats_section:
                try:
                    i, cat = l.split("\t")
                    cat = cat.split()[0]
                    id_to_cat[int(i)] = cat
                except:
                    pass
            else:
                w, cats = l.split("\t")[0], l.split("\t")[1:]

                w = w.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)").replace("$", r"\$").replace("+", r"\+").replace("\"", r"\"").replace(".", r"\.").replace("^", r"\^")

                w = r"(?<!\S)" + w
                if "*" in w:
                    w = w.replace("*", r"\w*(?!\S)")
                    if w[-6:] != r"(?!\S)":
                        w += r"(?!\S)"
                else:
                    w += r"(?!\S)"

                for c in cats:
                    c = id_to_cat[int(c)]
                    if c not in cats_to_words: cats_to_words[c] = set()
                    cats_to_words[c].add(w)

        return cats_to_words

    def _preprocess(self, sent, max_token):
        token_seq = sent.split()
        if max_token: 
            token_seq = token_seq[:max_token]

        sent = " ".join(token_seq)
        return sent, token_seq, len(token_seq)

    def _cal_coverage(self, sent, wc, word_set):
        reg_string = "|".join(word_set)
        r = re.compile(reg_string.encode("utf-8"))
        matched = r.findall(sent.encode("utf-8"))
        matched = [m.decode("utf-8") for m in matched]
        matched = " ".join(matched).split()
        coverage = len(matched) / wc
        assert coverage <= 1, print(len(matched), wc)

        return coverage, matched

class Liwc_RO(Liwc_Super):
    def __init__(self, dirname):
        super(Liwc_RO, self).__init__()

        self._word_list = self._construct_word_list(os.path.join(dirname, DIC_EN))

    def _preprocess(self, sent, max_token):
        token_seq = sent.split()
        if max_token: 
            token_seq = token_seq[:max_token]

        sent = " ".join(token_seq)
        return sent, token_seq, len(token_seq)

    def _cal_coverage(self, sent, wc, word_set):
        reg_string = "|".join(word_set)
        r = re.compile(reg_string.encode("utf-8"))
        matched = r.findall(sent.encode("utf-8"))
        matched = [m.decode("utf-8") for m in matched]
        matched = " ".join(matched).split()
        coverage = len(matched) / wc
        assert coverage <= 1, print(len(matched), wc)

        return coverage, matched

class Liwc_NL(Liwc_Super):
    def __init__(self, dirname):
        super(Liwc_NL, self).__init__()

        self._word_list = self._construct_word_list(os.path.join(dirname, DIC_EN))

    def _preprocess(self, sent, max_token):
        token_seq = sent.split()
        if max_token: 
            token_seq = token_seq[:max_token]

        sent = " ".join(token_seq)
        return sent, token_seq, len(token_seq)

    def _cal_coverage(self, sent, wc, word_set):
        reg_string = "|".join(word_set)
        r = re.compile(reg_string.encode("utf-8"))
        matched = r.findall(sent.encode("utf-8"))
        matched = [m.decode("utf-8") for m in matched]
        matched = " ".join(matched).split()
        coverage = len(matched) / wc
        assert coverage <= 1, print(len(matched), wc)

        return coverage, matched
