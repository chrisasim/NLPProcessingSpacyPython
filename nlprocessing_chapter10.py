import spacy
import pandas as pd
df = pd.read_csv('data/stocks.tsv')
symbols = df.Symbol.tolist()
companies = df.CompanyName.tolist()
print(symbols[:10])
nlp = spacy.blank("en")
ruler = nlp.add_pipe("entity_ruler")
patterns = []
for symbol in symbols:
 patterns.append({"label": "STOCK", "pattern": symbol})
for company in companies:
 patterns.append({"label":"COMPANY", "pattern": company})
doc = nlp(text)
for ent in doc.ents:
 print(ent.text, ent.label_)
ruler.add_patterns(patterns)
#false positive machine learning model.
from spacy import displacy
doc = nlp(text)
displacy.render(doc, style="ent")

