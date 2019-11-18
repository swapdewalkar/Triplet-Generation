import spacy

nlp = spacy.load('en_core_web_sm')
print("swapnil")

doc = nlp(u"Autonomous cars shift insurance liability toward manufacturers")
for token in doc:
    print(token.text, token.dep_, token.head.text)