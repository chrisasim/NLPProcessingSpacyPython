import re
text = "Paul Newman was an American actor, but Paul Hollywood is a British TV Host. The name Paul is quite common."
#that is the text we are going to work with. 
pattern = r"Paul [A-Z]\w+"
matches = re.find(pattern, text)
for match in matches:
 print(match)
matches = re.finditer(pattern, text)
for match in matches:
 print(match)
#going forward in custom space pipe
#write the code
#import spacy dot token import span
import spacy
from spacy.tokens import Span
nlp =spacy.blank("en")
doc = nlp(text)
original_ents = list(doc.ents)
mwt_ents = []
for match in re.finditer(pattern, doc.text):
 start, end = match.span()
#reverse engineer this to get the character span
 span = doc.char_span(start, end)
 print(span)
 if span is not None:
   mwt_ents.append((span.start, span.end, span.text))
for ent in mwt_ents:
 start, end, name = ent
 per_ent = Span(doc, start, end, label="PERSON")
 original_ents.append(per_ent)
doc.ents = original_ents
for ent in doc.ents:
 print(ent.tex, ent.label_)

from spacy.language import Language
#@Language.componet("paul_ner")
#def paul_ner(doc):

print(mwt_ents)
nlp2 = spacy.blank("en")
nlp2.add_pipe("paul_ner")
doc2 = nlp2(text)
print(doc2.ents)
spacy.blank("en")
nlp3 = spacy.load("en_core_web_sm")
nlp3.add_pipe("cinema_ner")
doc3 = nlp3(text)
for ent in doc3.ents:
 print(ent.tex, ent.label_)

