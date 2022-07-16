import spacy
nlp = spacy.load("en_core_web_sm")
#Spacy features why they are useful
#containers within spacy are large quantity of data.
#example, span, spangroup, toek, spangroup, docbin, doc
#spacy understands sentences as each word separately span span span and categorizes them into groups
#into a document. In case I iterate each, punctuation marks, self-contained importan values syntactically or semantically. 
#Mrtin arthur the king will be a person whose a collection of sequence of individual tokens.
#object I create that contains access eventually improve in python in spacy.
#quite similar to text itself and it is much more powerful. We are getting used to in depth the features into spacy. 
#Follow along as we explore coding.
#we are loading up the model so we are working on the container that stores a lot of features about the text 
#lets go ahead that we created the nlp project. within this we have got a data sheet folder. 
#we have got one in the states lets use this on the operator.
with open("data/wiki_us.txt") as f:
 text = f.read()

print(text)
#in case I am wroking with multiple documents I would name them after doc1, doc2, doc3
doc = nlp(text)
print(len(text))
print(len(doc))
#lets examine the case in which the lengths of text and documents are not similar.
for token in text[0:10]:
 print(token)
for token in doc[:10]:
 print(token)
#The profoundly hugely difference betweek text and document is the fact that the first one 
#creates a token with letters. The second is much more valuable because it creates tokens with words
#The first one counts each character whereas the second one pronts each word as a token.
#in order to split up the text, I dont need anything fancy from spacy. lets see a toke. 
#I am splitting light spaces iterate the first ten again. 
for token in text.split()[:10]:
 print(token)

#that looks good, up until down the line of the brackets. The problem is the fact that
#spacy automatically removes individual tokens that are not relavant with the token. 
#whereas without spacy it does not do that. 
#This is the power of spacy spend a few monutes and start talking about token features and attributes. 
#Identification of sentences within text. Senetence boundary detection is the term.
#we are trying to split a text within an individual sentence and the punctuation of the texts
#for example USA represetna an abbreviated word. 
for sent in doc.sents:
 print(sent)
#sentence1 = doc.sents[0]
#print(sentence1)
#We get an error. That is we iterate because it is a generator. There is a solution for this. In fact, there is. 
#we need to convert it into list to convert the sents into a list. 
sentence1 = list(doc.sents)[0]
print(sentence1)
#as we move forward we will ne talking about individual tokesn. Go ahead and play around a little bit
#convert into a list and we will be talking about token. 
#Under token attributes on chapter 2. we are going to see and provid ethe most importa ones. 
#lets explain with documentation. lets go ahead and start talking the doc object has a sequence of objects. 
#each individual token what we dont see here has a bunch of metadata are things that we call attributes. 
#lets work with token2 

token2 = sentence1[2]
print(token2)
token2.text
#id we are working with text states that works like that. corresponds tothe work states. behind there is a bunch of metadata 
#token2 left edge has multiple components and this is the most left token edge component in the states print that off and we get the word america. 
#we also learn a lot about is. From the beginning and left and right edge. token2. ent_type. 
print(token2.left_edge)
print(token2.right_edge)
print(token2.ent_type)
print(token2.ent_iob)
#I means it is inside of an entity , o means it is outside of the entity. 
#wwe can also say dot lemma. 
print(token2.lemma_)
print(sentence1[12].lemma_)
print(sentence1[12])
print(sentence1[12].morph)
print(token2.pos_)
print(token2.dep_)
print(token2.lang_)
text = "Mike enjoys playing fottball."
doc2 = nlp(text)
for token in doc2:
 print(token.text, token.pos_, token.dep_)
from spacy import displacy
displacy.render(doc2, style="dep")
#visualizae how words and sentences are connected with regards parts of speech. In the ext section we will be 
#talking about data visualization and information extraction from text. 
#books recommended. 
#access the pieces of information we will be dealing with a lot of ai. financial analyses. 
#GPE Geopolitical entity. simply continent. 
#LOC location
#five cardinal number 
#India NORP religious analysis 
#16th centrury means date 
#the american event is recorded 
#world waar II was recorded as event 
#for the most part, this is what is expected. 
#This is how I access really vital information about tokens ane enities. 
#to visualize and render 
#awesome
for ent in doc. ents:
 print(ent.text, ent.label_)
displacy.render(doc, style="ent")
#we get this really nice visualization these entities appear within the text. 
#even change the max length to get the characeters long. 
#nevertheles,, we really see very good results. there is a reason for this. 
#machine learning models typically wikipedia because it is included in the training process. 
#this chapter 2 ends. we understand what the doc container is, the documents are, the sentences and entities within the text. 
#linguistic features for token attributes. 
#become familiar with. 
#we are moving in chapter 3 in spacy python. 
#we understand the larger documents and entities and be ffamiliar with specifically with reagrds with language and transform models on machine learning models. 
#this means that is going to be larger and slowly depending on size. 
#we will be working on machine leanrin and how it works with regards with text. 

