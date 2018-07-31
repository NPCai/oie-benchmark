from oie_readers.oieReader import OieReader
from oie_readers.extraction import Extraction
import ast

class NeuralReader(OieReader):
    
    def __init__(self):
        self.name = 'OpenNMT_Format'
    
    def read(self, fn):
        d = {}
        currentSent = None
        with open(fn) as fin:
            for line in fin:
                data = line.strip()
                if data.startswith("SENT"):
                    tmp = ast.literal_eval(data[data.index(":") +1:].strip())
                    currentSent = ''.join(i + " " for i in tmp).strip()
               
                if data.startswith("[") and currentSent != None:
                    confidence = float(data[1: data.index("]")])
                    tup = ast.literal_eval(str(data[data.index(" ") :]).strip())
                    tup = ''.join(i + " " for i in tup).strip()
                    tup = tup.split("<>")
                    if len(tup) != 3:
                        continue
                    arg1, rel, arg2 = tup[0], tup[1], tup[2]
                    curExtraction = Extraction(pred = rel, sent = currentSent, confidence = float(confidence))
                    curExtraction.addArg(arg1)
                    curExtraction.addArg(arg2)
                    d[currentSent] = d.get(currentSent, []) + [curExtraction]
        self.oie = d
        self.normalizeConfidence()
    
    def normalizeConfidence(self):
        ''' Normalize confidence to resemble probabilities '''        
        EPSILON = 1e-3
        
        self.confidences = [extraction.confidence for sent in self.oie for extraction in self.oie[sent]]
        maxConfidence = max(self.confidences)
        minConfidence = min(self.confidences)
        
        denom = maxConfidence - minConfidence + (2*EPSILON)
        
        for sent, extractions in list(self.oie.items()):
            for extraction in extractions:
                extraction.confidence = ( (extraction.confidence - minConfidence) + EPSILON) / denom

    