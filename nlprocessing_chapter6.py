import spacy
from spacy.matcher import Matcher
nlp = spacy.load("en_core_web_sm")
matcher = Matcher(nlp.vocab)
pattern = [{"LIKE_EMAIL" : True}]
#we wanna extract everything that looks like an email. 
#a label we want to extract.
matcher.add("EMAIL_ADDRESS", [pattern])
doc = nlp("This is an email address: wmattingly@aol.com")
matches = matcher(doc)
print(matches)
print(nlp.vocab[matches[0][0]].text)
#the lower is when the patter is in lowe case. if we use a pattern in lowe case
#we look for a pattern that matches. The length is when the characters are in alphabetical order 
#is now replaced. If a token is a digit then it counts in the patter. 
#If the token is a title then capitalize. 
#or the entity. If the token is like a number or email we shall extract it. 
#any other string matchin any other token text 
#lemma and shape matching the linguistic of sequences. 
#all provided by a verb with a regex framework. 
#we leverage all the types of spaces in morphological analysis that have got into pipeline. 
#we make really cool patterns and robust techniques. 
#it turns how often this happens in zero times. 
#make the pattern optional with a questionmark.
#I am gonna work with another dataset open a script on jupyter and print text 
#extract very specific patterns that is the task ahead of us. 
#we are going to do all more.
with open("data/wiki_mlk.txt", "r") as f:
 text = f.read()
print(text)
nlp = spacy.load("en_core_web_sm")
speak_lemmas = ["think", "say"]
matcher = Matcher(nlp.vocab)
#pattern = [{"POS": "PROPN", "OP": "+"}]
pattern = [{"ORTH": "'"}, 
    {"IS_ALPHA": True, "OP": "+"},
      {"IS_PUNCT": True, "OP": "*"},
        {"ORTH": "'"},
         {"POS": "VERB", "LEMMA": {"IN": speak_lemmas}},
         {"POS": "PROPN", "OP": "+"}
]
matcher.add("PROPER_NOUN", [pattern], greedy='LONGEST')
doc = nlp(text)
matches = matcher(doc)
matches.sort(key=lambda x : x[1])
print(len(matches))
for match in matches[:10]:
 print(match, doc[match[1]:match[2]])
import json
with open("data/alice.json", "r") as f:
 data = json.load(f)

#text = text.replace("'", "'")
#print(text)
for text in data[0][2]:
 text = text.replace("'", "'")
 doc = nlp(text)
 print(len(matches))
 matches = matcher(doc)
 matches.sort(key = lambda x :x[1])
 for match in matches[:10]:
   print(match, doc[match[1]:match[2]])

matcher = Matcher(nlp.vocab)
pattern1 = [{'ORTH': "'" }, {'IS_ALPHA': True, "OP": "+"}, {'IS_PUNCT': True, "OP": "*"}]
for text in data[0][2]:
 text = text.replace("'", "'")
 doc = nlp(text)
 matches = matcher(doc)
 matches.sort(key = lambda x:x[1])
 print(len(matches))
 for match in matches[:10]:
   print(match, doc[match[1]:match[2]])

import spacy
nlp = spacy.load("en_core_web_sm")
doc = nlp("Britain is a place. Mary is a doctor.")
for ent in doc.ents:
 print(ent.text, ent.label_)

#doc = nlp("Britain is a place. Mary is a doctor.")


from spacy.language import Language
@Language.component("remove_gpe")
def remove_gpe(doc):
 original_ents = list(doc.ents)
 for ent in doc.ents:
   if ent.label_ == "GPE":
    original_ents.remove(ent)
 doc.ents = original_ents
 return (doc)
nlp.add_pipe("remove_gpe")
nlp.analyze_pipes()
doc  = nlp("Britain is a place. Mary is a doctor.")
for ent in doc.ents:
 print(ent.tex, ent.label_)
nlp.to_disk("data/new_en_core_web_sm")

#special microcomponent know how to handle particular data. In this textbook. 
#regular component just a moment with chapter 9 
#I spent five hours 
#really robust strings. 
#how to use regex in spacy. 
#in this example, not to extract the sequence pattern it tells us a sequence of character. 
#followed by dash. execute the whole code. 
#it is built for one important reason. 
#I cannot regex. It means
#engange REGEX in python.
#Extract multi word tokens with Regex. 
#indpependent with lemma and if working with linguistics. 
#if a sequence of string is not dependent with that. 
#capitalize letter then you are going to use of linguistic 
#much more robust thing component. 
#remove now. lets see regex and lets go ahead try and see regex multi word tokens for us. 
#the first thing I am going to do. 

