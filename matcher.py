import string
from nltk.translate.bleu_score import sentence_bleu
from nltk.corpus import stopwords
import spacy

nlp = spacy.load('en')

class Matcher:
    @staticmethod
    def bowMatch(ref, ex, ignorePunctuation, ignoreCase):
        s1 = ref.bow()
        s2 = ex.bow()
        if ignoreCase:
            s1 = s1.lower()
            s2 = s2.lower()
        
        s1Words = s1.split(' ')
        s2Words = s2.split(' ')
        
        if ignoreStopwords:
            s1Words = Matcher.removeStopwords(s1Words)
            s2Words = Matcher.removeStopwords(s2Words)
            
        return sorted(s1Words) == sorted(s2Words)
    
    @staticmethod
    def bleuMatch(ref, ex, ignorePunctuation, ignoreCase):
        sRef = ref.bow()
        sEx = ex.bow()

        if ignoreCase:
            sRef = sRef.lower()
            sEx = sEx.lower()

        sRef = sRef.split(' ')
        sEx = sEx.split(' ')

        if ignorePunctuation:
            sRef = Matcher.removePunctuation(sRef)
            sEx = Matcher.removePunctuation(sEx)


        bleu = sentence_bleu(references = [sRef], hypothesis = sEx)
        return bleu > Matcher.BLEU_THRESHOLD
    
    @staticmethod
    def spacyMatch(ref, ex, ignorePunctuation, ignoreCase):
        sRef = ref.bow()
        sEx = ex.bow()

        if ignoreCase:
            sRef = sRef.lower()
            sEx = sEx.lower()

        spacyRef = nlp(sRef)
        spacyEx = nlp(sEx)
        coverage = spacyRef.similarity(spacyEx)
        return coverage > Matcher.SPACY_THRESHOLD

    @staticmethod
    def complexMatch(ref, ex, ignorePunctuation, ignoreCase):
        sRef = ref.get_tuple()
        sEx = ex.get_tuple()
        if ex.confidence > 0.4:
            print("ref ", ref.get_tuple())
            print("ex ", ex.get_tuple())
        if sRef == None or sEx == None or len(ex.bow().split(' ')) > len(ref.bow().split(' ')) * 2: # Tuple length less than 2
            return False

        arg1 = nlp(sRef[0])
        exArg1 = nlp(sEx[0])

        pred = nlp(sRef[1])
        exPred = nlp(sEx[1])

        arg2 = nlp(sRef[2])
        exArg2 = nlp(sEx[2])


        arg1Sim = arg1.similarity(exArg1)
        predSim = pred.similarity(exPred)
        arg2Sim = arg2.similarity(exArg2)

        bools = arg1Sim > Matcher.SPACY_THRESHOLD and predSim > Matcher.PRED_THRESHOLD and arg2Sim > Matcher.SPACY_THRESHOLD
        toret = bools and Matcher.lexicalMatch(ref, ex, ignorePunctuation, ignoreCase)
        return toret

    @staticmethod
    def lexicalMatch(ref, ex, ignorePunctuation, ignoreCase):
        sRef = ref.bow()
        sEx = ex.bow()


        if ignoreCase:
            sRef = sRef.lower()
            sEx = sEx.lower()

        sRef = sRef.split(' ')
        sEx = sEx.split(' ')

        if ignorePunctuation:
            sRef = Matcher.removePunctuation(sRef)
            sEx = Matcher.removePunctuation(sEx)
            
        count = 0
        
        for w1 in sRef:
            for w2 in sEx:
                if w1 == w2:
                    count += 1
                    
        # We check how well does the extraction lexically cover the reference
        # Note: this is somewhat lenient as it doesn't penalize the extraction for
        #       being too long
        coverage = float(count) / (len(sRef))

        return coverage > Matcher.LEXICAL_THRESHOLD
    
    @staticmethod
    def removeStopwords(ls):
        return [w for w in ls if w.lower() not in Matcher.stopwords]

    @staticmethod
    def removePunctuation(ls):
        return [w for w in ls if w.lower() not in list(string.punctuation)]
    
    # CONSTANTS
    BLEU_THRESHOLD = 0.4
    LEXICAL_THRESHOLD = 0.3 # Note: changing this value didn't change the ordering of the tested systems
    SPACY_THRESHOLD = 0.2
    PRED_THRESHOLD = 0.2
    stopwords = stopwords.words('english') + list(string.punctuation)





