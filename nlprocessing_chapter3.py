import spacy
import numpy as np
#word vectors or word embeddings are numerical respresentation of words in multidimensional space through matrices. 
#The purpose of the word vector is to get a computer system to understand a word. Computers cannot understand text efficintly. They can however, process numbers quickly and well. For this reason, it is important, to convert a word into a number. 
#Initial methods for creating word vectors in a pipleine take all words in a corpus and convert them into a single, unique number. These words are then stored in a dictionary that would like this {"this": 1, "a":2}
#known as a bag of words. This approach to representing words numerically, however, only allow a computer to understand words numberically to identify unique words. 
#computationally is less expensive than the lists in python. 
#To a cimputer system, means tactible and semantic meaning, which gives a cimputer system a differ larger collection of words. in relation to other wwords. 
#figure out meaning.
nlp = spacy.load("en_core_web_md")
with open("data/wiki_us.txt", "r") as f:
 text = f.read()

doc = nlp(text)
sentence1 = list(doc.sents)[0]
print(sentence1)
your_word = "country"
ms = nlp.vocab.vectors.most_similar(np.asarray([nlp.vocab.vectors[nlp.vocab.strings[your_word]]]), n=10)
words = [nlp.vocab.strings[w] for w in ms[0][0]]
distances = ms[2]
print(words)
doc1 = nlp("I like salty fries and hamburgers.")
doc2 = nlp("Fast food tastes very good")

print(doc1, "<->", doc2, doc1.similarity(doc2))
doc3 = nlp("The Empire State Building is in New York")
print(doc1, "<->", doc3, doc1.similarity(doc3))
doc4 = nlp("I enjoy oragnes")
doc5 = nlp("I enjoy apples.")
print(doc4, "<->", doc5, doc4.similarity(doc5))
doc6 = nlp("i enjoy burgers.")
print(doc4, "<->", doc6, doc4.similarity(doc6))
#similarity difference in meaning as a langugae in whole. 
#calculate other things as well. 
#caluclate salty fries as well. 
#play around with the code over here spacy can help you find similarities between documents and texts as well. 
#why median and large are so bigger and larger.
#pipline is very common expression in data asience. a sequence of differnt pipes. 
#as each pipe in addition the late pipes is very common when you think code. 
#This input sentence and enter spacy pipeline. They tokenize it of different vectors and words. 
#That might be an entity ruler. will be a series of rule-based in ai inti the token o the entity. 
#an individual token now receive a bunch of dot token. 
#The next pipeline tries to find out which they are some of wiki data for example a series of bunch of things might have the connenction. 
#This entity linker is out of the scipr. kepp in mind this pipeline might be doing something else on the entity ruler. 
#a sequence of pipes that act on your data. really consiencious.
#Span categorizer the text categorizer is when you train the machin elarning task which is assigned into word embeddings. 
#all the text and indivudial tokesn theere are also matchers to make things less confusing 
#add a phrase matcher add it in the github repo. 
#a big question is how to add.
french_fries = doc1[2:4]
burgers = doc1[5]
print(french_fries, "<->", burgers, french_fries.similarity(burgers))
nlp = spacy.blank("en")
#neglish tokenizer. add pipe into the token in order to add sencecizer
nlp.add_pipe("sentencizer")
#the smaller the better the sentencizer did not have extra adda, if it had we would have better ressults. 
#we simply add a sentencizer different ones into the pipelines.
#the general understanding how this might work.
#we can analyze our pipeline
#if we look on the nlp pipeline ignore summary and we really have got syntencizer 
#into spacy load.
#start leveragin for my own uses. Model english model. make an entity more entities based on the list that we have
#find specific sequences within text. Find components and features within sentences. 

#in part 3 we start applying all these skills and we are going to be moving in part 5. 
#this is the bread and butter. the umbrealla nd entity spacy into linguistic entity rules into custom componetns into spacy. 
#more robut and sophisticated trained model so that we can put into pipeline into advance machine learning
#
